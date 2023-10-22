# Copyright (C) Robert Bosch GmbH 2023.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# This program and the accompanying materials are made available under
# the terms of the Bosch Internal Open Source License v4
# which accompanies this distribution, and is available at
# http://bios.intranet.bosch.com/bioslv4.txt

from enum import Enum
from typing import Callable, Optional
from itertools import compress
from bolt.applications.scl9hi.preprocessing.datasets import Datasets
from bolt.applications.scl9hi.compute.model import IMG_SIZE
from bolt.applications.scl9hi.compute.pose3d import Pose3D

import numpy as np
import random

from bolt.applications.pose_estimation_3d.data.keypoint.pose_keypoints import PoseKeypoints
from bolt.applications.mfsd.data import MfsdDataPipe
from bolt.applications.object_detection.v0.data import ObjectDetectionListing
from bolt.applications.object_detection.v0.data.annotations import MSCOCOLoader
from bolt.applications.object_detection.v0.data.processing.box_processing import affine_box_transform
from bolt.applications.object_detection.v1.data.processing import assemble_objdet_batch, limit_box_num
from bolt.data import GenericGenerator
from bolt.data.containers import BoxAttribute2D, FloatAttribute, KeypointsAttribute, OneHotAttribute
from bolt.data.processing.geometry import geometric_image_transformation

from bolt.applications.object_detection.v0.data.processing.box_processing import (
    DeleteBoxes,
    IgnoreBoxes,
)
from bolt.applications.scl9hi.preprocessing.datasets import Dataset

from bolt.applications.keypoint_detection.data.keypoint import convert_to_pers
from bolt.applications.object_detection.v0.visualization import BoundingBoxPlotter
from bolt.applications.keypoint_detection.postprocessing.drawing_utils import print_keypoints_on_image

from bolt.applications.keypoint_detection.data.keypoint import KeypointsDefinition
from bolt.data.augment.affine import AffineTransformationSampler


class DataPipe(MfsdDataPipe):
    """Simple loading of mnist data but extended with keypoints."""

    def __init__(self, name, annotations, curr_dataset=None, data_format_dataset=None, validation=False, uwe=False):
        super().__init__(
            out_dims=list([int(axis) for axis in IMG_SIZE]) if validation else IMG_SIZE,
            img_specifier_key="file",
            file=annotations,
            validation=name == "validation",
            gap_params=None,
        )
        self.uwe_counter = 0
        self.uwe = uwe

        self.sampler = AffineTransformationSampler(target_shape=self.out_dims)

        self.data_format_dataset = data_format_dataset
        self.data_curr_dataset = curr_dataset
        self.is_validation = (name == "validation")
    def helperPlot(self, label, img):
        painter = BoundingBoxPlotter(top=None, bottom="{occlusion}")
        bb_img = painter(boxes=label, image=img)
        if self.data_curr_dataset.kp_df:
            personen = list(convert_to_pers(person, self.data_curr_dataset.kp_df, IMG_SIZE) for person in label.keypoints)
            kp_img = print_keypoints_on_image(keypoints=personen, image=bb_img)
            return kp_img
        # else:
        return bb_img

    def kp_visible(self, kp_array):
        absolute_kp = sum([not np.isnan(kp) for kp in kp_array])/2 # since we are looking at 2d Keypoints
        should = self.data_format_dataset.kp_df.num_keypoints
        return absolute_kp/should

    def __call__(self, sample):
        file = self.file(sample)
        # the sample is used to determine which information should be read by the mnist reader
        file_name = sample["reader_args"][self.img_specifier_key]
        if self.data_curr_dataset == Datasets.KI_A_S:
            file_name = file_name.split('/')[-1]
        img_org = self.reader(file, file_name)

        if self.data_curr_dataset == Datasets.AGORA:
            # Check if the image has an alpha channel
            if img_org.shape[2] == 4:
                # Extract only the RGB channels
                img_org = img_org[:, :, :3]

        if img_org.shape[-1] == 1:
            img_org = np.repeat(img_org, 3, axis=-1)
        label_org = sample["label"]

        # adapt data to target dataset
        #label.keypoints = self.transfrom_kp(label.keypoints)

        aug_matrix = self.sampler(img_org, translation_params={"image": img_org})
        img = geometric_image_transformation(img_org, aug_matrix, output_shape=self.out_dims)
        label = affine_box_transform(
            label_org,
            aug_matrix,
            output_shape=self.out_dims,
            clip_to_output_boundaries=True,
            delete_cropped_margin=None,
        )
        # x min: 10, y min: 20
        if self.is_validation:
            self.box_deleter = DeleteBoxes(min_height=5, min_width=5)
        else:
            self.box_deleter = DeleteBoxes(min_height=10, min_width=10)
        if self.data_curr_dataset == Datasets.JTA:
            self.box_deleter = DeleteBoxes(min_height=15, min_width=15)


        label = self.box_deleter(label)

        # # Used for debugging
        # occlusion: 80%

        box_ignorer = (
            self.box_ignorer_synth if self.mixed_training and not sample.get("real", False) else self.box_ignorer
        )
        label_ignored = box_ignorer(label)

        # ignore every person that is more than 80 occluded or if the occlusion is not known
        # ignore every person that has less than 30% of the visible keypoints

        # for each dataset handle the filtering different:
        if self.data_curr_dataset == Datasets.JTA:
            mask = list(filter(lambda i: label_ignored[i].occlusion < 0.6, range(len(label_ignored))))
        elif self.data_curr_dataset == Datasets.AGORA:
            mask = list(filter(lambda i: label_ignored[i].occlusion < 0.95, range(len(label_ignored))))
        elif self.data_curr_dataset == Datasets.MOTS:
            mask = list(filter(lambda i:  label_ignored[i].occlusion[0] < 0.8, range(len(label_ignored))))
        else:
            mask = list(filter(lambda i: label_ignored[i].occlusion[0][0] < 0.8 or
                            np.isnan(label_ignored[i].occlusion[0][0]) and (sum(np.isnan(label_ignored[i].keypoints[0])) / len(label_ignored[i].keypoints[0])) < 0.7 or
                            label_ignored[i].vis_ratio[0][0] > 0.2,
                            range(len(label_ignored))))
        # apply visible filter
        label_limited = label_ignored[mask]

        # ignore the two above rules for shift, since they dont have keypoints "yet"
        # and also they dont have accurate vis ratios.
        if self.data_curr_dataset == Datasets.SHIFT:
            label_limited = label_ignored

        if self.data_curr_dataset == Datasets.MOTS:
            mask = list(filter(lambda i:  label_ignored[i].occlusion[0] < 0.8, range(len(label_ignored))))
            label_ignored = label_ignored[mask]

        label_limited = limit_box_num(label_limited, 300, False)

        # just for debugging purpose
        # personen = list(convert_to_pers(person, self.data_curr_dataset.kp_df, IMG_SIZE) for person in label_limited.keypoints)
        # painter = BoundingBoxPlotter(top="", bottom="{occlusion}")
        # bb_img = painter(boxes=label_ignored, image=img)
        # kp_img = print_keypoints_on_image(keypoints=personen, image=bb_img)

        # transform not annotated coordinates (0,0) to nan values,
        # so they will be ignored in loss
        keypoints_reshape = label_limited.keypoints.reshape(-1, 2)
        valid_mask = (keypoints_reshape > 0).sum(axis=1).astype(bool)
        keypoints_reshape[np.logical_not(valid_mask), :] = np.nan
        label_limited.keypoints = keypoints_reshape.reshape(label_limited.keypoints.shape)

        # same for 3d loss
        keypoints_reshape = label_limited.keypoints_3d.reshape(-1, 3)
        valid_mask = (keypoints_reshape != 0).sum(axis=1).astype(bool)
        keypoints_reshape[np.logical_not(valid_mask), :] = np.nan
        label_limited.keypoints_3d = keypoints_reshape.reshape(label_limited.keypoints_3d.shape)

        # transform keypoints to root relative to make training a lot easier ...

        # make ground truth already ready for evaluation ...
        if len(label_ignored.keypoints_3d) != 0 and not np.isnan(label_ignored.keypoints_3d).all():
            kp_3d = label_ignored.keypoints_3d.reshape((len(label_ignored.keypoints_3d),int(len(label_ignored.keypoints_3d[0])/3),3))
            kp_evalable = PoseKeypoints(kp_3d, self.data_curr_dataset.kp_df, self.data_curr_dataset.kp_df.root_kp_name)
        else:
            kp_evalable = None

        #img_ignored = self.helperPlot(label_org, img_org)
        #img_limited = self.helperPlot(label_limited, img)
        if self.uwe:
            self.uwe_counter += 1
            path = f'/home/scl9hi/uwe_test/frame_{self.uwe_counter:04d}.png'
            import cv2
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # label_limited.bbox_coords = []
            # label_limited.classes = []
            # label_limited.keypoints = []
            # label_limited.keypoints_3d = []
            # label_limited.abs_3d = []
            # label_ignored = []
            # kp_evalable = []
            # label_ignored.abs_3d = []

        return {
            "net_in": {
                "input": img,
            },
            "target": {
                "bbox_coords": label_limited.bbox_coords,
                "classes": label_limited.classes,
                "keypoints": label_limited.keypoints,  # added this line to support keypoints
                "keypoints_3d": label_limited.keypoints_3d, # added this line to support 3d keypoints
                "abs_3d": label_limited.abs_3d
            },
            # as ground truth give only those back which are visible by our critea, otherwise it is hard to evalutate each dataset
            # against each other
            "callback_data": {"original_img": img, "label": label_ignored, "keypoints": label_ignored.keypoints, "keypoints_3d": kp_evalable,
                              "abs_3d": label_ignored.abs_3d},
        }
class ExtendedObjDetListing(ObjectDetectionListing):
    def __init__(self, annotation_loader, num_samples, seed, kp_df):
        super().__init__(annotation_loader=annotation_loader, num_samples=num_samples, seed=seed)
        self.kp_df = kp_df
    def __call__(self):
        annotation_base = super(ExtendedObjDetListing, self).__call__()
        for annotation in annotation_base:
            annotation["root_keypoint_name"] = self.kp_df.root_kp_name
            annotation["dataset_keypoint_definition"] = self.kp_df
            annotation["master_keypoint_definition"] = self.kp_df
            annotation["gt_camera"] = None
        return annotation_base



def mfsd_listing(
    annotation_path: str,
    kp_df: KeypointsDefinition,
    skip: int = 1,
    seed: int = 0,
    num_keypoints: int = 19,
    num_samples: Optional[int] = None,
) -> ObjectDetectionListing:
    """Listing for KIA and Citypersons dataset used in the MFSD activity.

    Parameters
    ----------
    annotation_path : str
        path to annotations
    skip : int, optional
        use only every "skip"-th image, by default 1
    num_samples : Optional[int], optional
        number of samples to keep of the listing, by default None which will use all
    seed : int, optional
        seed used for shuffeling when num_samples is not None

    Returns
    -------
    ObjectDetectionListing
        annotation listing for object detection
    """

    def vis_and_occ(boxes):
        for box in boxes:
            if box["vis_ratio"] is not None:
                box.update(occlusion=1 - box["vis_ratio"])
            elif box["vis_ratio"] is not None:
                box.update(vis_ratio=1 - box["occlusion_est"])
            else:
                box.update(vis_ratio=1)
        return boxes

    attributes = {
        "label": OneHotAttribute(hierarchy="person", has_bg_class=True),
        "pos": BoxAttribute2D(),
        "vis_ratio": FloatAttribute(name="vis_ratio", ignore_value=0.0, length=1),
        "occlusion": FloatAttribute(name="occlusion", ignore_value=1.0, length=1),
        "truncation": FloatAttribute(name="truncation", ignore_value=1.0, length=1),
        "occlusion_est": FloatAttribute(name="occlusion_est", ignore_value=1.0, length=1),
        "keypoints": KeypointsAttribute(name="keypoints", num_of_keypoints=num_keypoints, use_unvisible=False),
        "keypoints_3d": Pose3D(kp_df, "keypoints_3d"),
    }
    anno_loader = MSCOCOLoader(
        annotation_path=annotation_path,
        attribute_representations=attributes,
        used_annotations={
            "pos": None,
            "keypoints": None,
            "keypoints_3d": None,
            "label": {
                "included": ["person"],
                "label_mapping": [["pedestrian", "person"],
                                   ["pedestrain", "person"],
                                   ["bdd_rider", "person"],
                                   ["bdd_person", "person"],
                                   ["bdd_bike", "person"],
                                   ["bdd_motor", "person"]],
                "ignored": ["ignore", "group", "rider", "sitting_person", "other"],
            },
            "vis_ratio": None,
            "occlusion": None,
            "truncation": None,
            "occlusion_est": None,
        },
        modify_attributes=vis_and_occ,
        skip=skip,
    )
    return ExtendedObjDetListing(annotation_loader=anno_loader, num_samples=num_samples, seed=seed, kp_df=kp_df)


def filtered_sampling(data_list, steps=None, batch_size=None):
    # this functions filters out every image that does not contain keypoint annotations or bounding box
    training_list = [
        i
        for i, item in enumerate(data_list)
        if len(item["label"].keypoints) != 0 and not np.all(np.isnan(item["label"].keypoints)) or
            len(item["label"].bbox_coords) != 0 and not np.all(np.isnan(item["label"].bbox_coords))
    ]
    random.shuffle(training_list)
    return training_list

def filtered_sampling_val(data_list, steps=None, batch_size=None):
    # this functions filters out every image that does not contain keypoint annotations or bounding box
    training_list = [
        i
        for i, item in enumerate(data_list)
        if len(item["label"].keypoints) != 0 and not np.all(np.isnan(item["label"].keypoints)) or
            len(item["label"].bbox_coords) != 0 and not np.all(np.isnan(item["label"].bbox_coords))
    ]
    return training_list


def build_data(dataset: Dataset, trained_on: Dataset, debug, batchsize=6, uwe=False) -> Callable[[str], GenericGenerator]:
    """Builds the factory for the dataset which is loaded in the model.

    Parameters
    ----------
    dataset : Dataset
        the dataset to load
    trained_dataset: Dataset
        the dataset on which the model was trained on,
        if it differs from the dataset which is loaded, the data has to be transformed.

    Returns
    -------
    Callable[[str], GenericGenerator]
        the Dataset Generator as a Callable
    """

    def factory(name: str) -> GenericGenerator:
        # The sampling generator is responsible for selecting a batch of
        # samples from the sample listing to output next. This can either
        # be a user-defined function or one of the following predefined
        # sampling generators: ordered, random, random_fixed_seed

        # there is no test split used so we always use the val split
        validation = False
        if name == "test":
            name = "validation"
        if name == "validation":
            validation = True
        if debug:
            sampling_generator = {"train": "random", "validation": "random", "test": "random"}

        else:
            sampling_generator = {"train": "random", "validation": "ordered", "test": "ordered"}



        splits = {"validation": dataset.splits.val,
                "train": dataset.splits.train,
                "test": dataset.splits.test}

        anno_path = splits[name].annotation_path
        if dataset.has_debug and debug:
            anno_path += ".small"

        input_dszip = splits[name].input_dszip

        return GenericGenerator(
            batch_size=batchsize,
            sample_listing=mfsd_listing(
                annotation_path=anno_path, num_keypoints=trained_on.kp_df.num_keypoints if trained_on.kp_df else None, skip=1,
                kp_df=trained_on.kp_df
            ),
            sampling_generator=sampling_generator[name],
            batch_assembler=assemble_objdet_batch,
            # name, annotations, curr_dataset=None, data_format_dataset=None
            graph=DataPipe(name, input_dszip, dataset, trained_on, validation, uwe=uwe),
        )

    return factory
