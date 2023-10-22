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

from typing import Any
import numpy as np
from bolt.applications.scl9hi.compute.model import IMG_SIZE

import bolt.applications.keypoint_detection.data.keypoint
import bolt.applications.keypoint_detection.postprocessing.drawing_utils
import bolt.applications.object_detection.v0.core
import bolt.applications.object_detection.v0.post_processing
import bolt.applications.object_detection.v0.visualization
import bolt.applications.object_detection.v1.backends.torch.encodings.rcnn
import bolt.applications.object_detection.v1.data.encoding
import bolt.applications.object_detection.v1.post_processing
import bolt.data.containers.attributes
from bolt.applications.pose_estimation_3d.data.keypoint.pose_keypoints import PoseKeypoints
from bolt.applications.keypoint_detection.data.keypoint import KeypointsDefinition
from bolt.applications.object_detection.v0.visualization import DrawStyle
from bolt.applications.scl9hi.constants import THRESHOLD_CLASS

class PreprocessPoints:
    """processes network output.

    this class reprocess the net output to make it available to the callbacks,
    which then evaluate the data from the network.
    """

    def __init__(self, kp_2d, kp_3d, output_for_object_evaluation, kp_df: KeypointsDefinition) -> None:
        self.output_for_object_evaluation = output_for_object_evaluation
        self.kp_2d = kp_2d
        self.kp_3d = kp_3d
        self.kp_df = kp_df
        self.decode = bolt.applications.object_detection.v1.post_processing.OutputDecoder(
            bgscore_threshold=0.975,
            decoders=self._build_decoders(),
            attributes=self._build_attributes(),
        )
        self.nms = bolt.applications.object_detection.v0.post_processing.NMS(
            include_background=False,
            threshold=0.4,
            obj_num=100,
            overlap_fcn=bolt.applications.object_detection.v0.core.intersection_over_union,
        )

    def _build_decoders(self):
        decoders = {
            "classes": bolt.applications.object_detection.v1.data.encoding.logits_to_confidences,
        }
        if self.kp_2d:
            decoders["keypoints"] = bolt.applications.object_detection.v1.backends.torch.encodings.rcnn.point_decode
        return decoders

    def _build_attributes(self):
        attributes = [
            bolt.data.containers.attributes.BoxAttribute2D(),
            bolt.data.containers.attributes.OneHotAttribute(hierarchy="person", has_bg_class=True),
        ]
        if self.kp_2d:
            attributes.append(bolt.data.containers.attributes.KeypointsAttribute(
                    name="keypoints", num_of_keypoints=len(self.kp_df.keypoint_names), use_unvisible=True
                ),
            )
        if self.kp_3d:
            attributes.append(bolt.data.containers.attributes.KeypointsAttribute(
                    name="keypoints_3d", num_of_keypoints=len(self.kp_df.keypoint_names), use_unvisible=True
                ),
            ),
            attributes.append(bolt.data.containers.attributes.KeypointsAttribute(
                    name="abs_3d", num_of_keypoints=len(self.kp_df.keypoint_names), use_unvisible=True
                ),
            )

        return attributes

    # 'callback_data', 'sample_info', 'net_in', 'target', and 'sample_weight'
    def __call__(
        self, net_out, callback_data=None, sample_info=None, net_in=None, target=None, sample_weight=None, epoch=None
    ) -> Any:
        decoded = self.decode(net_output=net_out)
        nms = self.nms(bboxes=decoded)
        if self.output_for_object_evaluation:
            output = {
                "info": sample_info["label"].frame_name,
                "annotations": callback_data["label"],
                "predictions": nms,
            }
        else:
            output = {
                "detections": nms,
                "predictions": nms,
                "img_size": IMG_SIZE,
                "kp_df": self.kp_df,
                "info": "None" if sample_info is None else sample_info["label"].frame_name,
                "annotations": "None" if callback_data is None else callback_data["label"],
            }

            if self.kp_2d:
                persons = []
                persons = list(
                    bolt.applications.keypoint_detection.data.keypoint.convert_to_pers(skeleton, self.kp_df, IMG_SIZE)
                    for skeleton in nms.keypoints
                )
                output["pred_keypoints"] = persons
            if self.kp_3d:
                kp_3d = nms.keypoints_3d

                # Numpy array of shape (B, J, D), where D is 3.
                # Batch, Joints, Dimensions
                if len(kp_3d) != 0:
                    # calculate absolute 3d pose

                    num_kp = int(len(kp_3d[0])/3)

                    abs_3d = np.array([np.array(array) for array in nms.abs_3d])
                    abs_repeated = np.array([np.repeat(np.array([array]), axis=0, repeats=num_kp) for array in abs_3d])

                    kp_3d = kp_3d.reshape((len(kp_3d),num_kp,3))
                    kp_3d = np.array([arr for arr in kp_3d if not np.all(np.isnan(arr))])

                    abs_kp = np.nan_to_num(kp_3d) + abs_repeated
                    abs_kp = np.array([arr for arr in abs_kp if not np.all(np.isnan(arr))])

                    kp_evalable = PoseKeypoints(kp_3d, self.kp_df, self.kp_df.root_kp_name)
                    kp_abs =  PoseKeypoints(abs_kp, self.kp_df, self.kp_df.root_kp_name)
                    output["pred_keypoints_3d"] = kp_evalable
                    output["pred_keypoints_3d_abs"] = kp_abs

                    # convert ground truth
                    abs_3d = np.array([np.array(array) for array in callback_data['abs_3d']])
                    abs_repeated = np.array([np.repeat(np.array([kp]), axis=0, repeats=num_kp) for kp in abs_3d])

                    if callback_data['keypoints_3d']:
                        kp_3d = callback_data['keypoints_3d'].arr
                        # num_kp = int(len(kp_3d[0])/3)
                        # kp_3d = kp_3d.reshape((len(kp_3d),num_kp,3))
                        abs_kp = np.nan_to_num(kp_3d) + abs_repeated
                        abs_kp = np.array([arr for arr in abs_kp if not np.all(np.isnan(arr))])
                        gt_kp_abs =  PoseKeypoints(abs_kp, self.kp_df, self.kp_df.root_kp_name)

                        output["grth_3d_abs"] = gt_kp_abs
                    else:
                        output["grth_3d_abs"] = None
                else:
                    # empty keypoints
                    output["pred_keypoints_3d"] = None
                    output["pred_keypoints_3d_abs"] = None
                    output["grth_3d_abs"] = None
        return output

    def outputs(self):
        if self.output_for_object_evaluation:
            keys = ["info", "annotations", "predictions"]
        else:
            keys = [
                "detections",
                "img_size",
                "kp_df",
                "info",
                "annotations",
                "predictions"
            ]

            if self.kp_2d:
                keys.append("pred_keypoints")
            if self.kp_2d:
                keys.append("pred_keypoints_3d")
                keys.append("pred_keypoints_3d_abs")
                keys.append("grth_3d_abs")
        return keys


class DrawImage:
    """takes post_process Networkoutput and draws a image based on the kp (if present) and bboxes.

    Parameters
    ----------
    train_with_kp: bool, whether the training is including keypoints
    """

    def __init__(self) -> None:
        self.paint = bolt.applications.object_detection.v0.visualization.BoundingBoxPlotter(
            top="{classes:label}", bottom="{classes:score}",
            draw_style=DrawStyle(line_width=3)
        )
        # self.kp_image = bolt.applications.keypoint_detection.postprocessing.drawing_utils.print_keypoints_on_image

    def __call__(self, callback_data) -> Any:
        # filter detections
        matching = bolt.applications.object_detection.v0.visualization.BoxMatchingPlotter(plotter=self.paint)
        filter_indexes = [i for i, det in enumerate(callback_data['detections']) if det.classes[0][1] > THRESHOLD_CLASS]
        matched = matching(callback_data["original_img"], callback_data["detections"][filter_indexes], callback_data['label'])
        #person_indexes = [i for i, det in enumerate(callback_data['label']) if callback_data['label'].classes[i].__format__(params="label") == "person"]


        #ignored_ = self.paint(boxes=callback_data['label'], image=callback_data["original_img"], colors=[(255, 0, 0) for i in range(0, 150)])
        #target = self.paint(boxes=callback_data['label'][person_indexes], image=ignored_, colors=[(0, 255, 0) for i in range(0, 150)])
        #predictions = self.paint(boxes=callback_data["detections"][filter_indexes], image=target, colors=[(0, 0, 255) for i in range(0, 150)])
        # if self.kp_2d:
        #     kp_image = self.kp_image(keypoints=callback_data["keypoints"][filter_indexes], image=paint)
        #     ret_output = {"bp_kp": kp_image}
        # else:
        ret_output = {"bp_kp": matched}
        return ret_output

    def outputs(self):
        return ["bp_kp"]
