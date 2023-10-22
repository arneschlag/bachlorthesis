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

import os

from dataclasses import dataclass
from typing import Union
from bolt.applications.keypoint_detection.data.keypoint import KeypointsDefinition
from bolt.applications.scl9hi.preprocessing.definitions import *

ANNOTATION_SUBFOLDER = '02_stud/01_schlag/01_datasets/'

@dataclass
class Split:
    """a split is a part of an dataset.
        it can have one of the following keys:
        annotation_path: coco encoded json file
        input_dszip: packed dszip file, which are the input images
        instance_mask: packed dszip file with instance masks
    """
    annotation_path: Union[str, None] = None
    input_dszip: Union[str, None] = None
    instance_mask_dszip: Union[str, None] = None
    def __eq__(self,other):
        """Overrides the default implementation"""
        if isinstance(other, Split):
            return self.annotation_path == other.annotation_path and self.input_dszip == other.input_dszip and self.instance_mask_dszip == other.instance_mask_dszip
        return False


@dataclass
class Splits:
    """ A dataset can three different splits:
        train
        validation
        test
    """
    train: Union[None, Split] = None
    val: Union[None, Split] = None
    test: Union[None, Split] = None
    def __eq__(self,other):
        """Overrides the default implementation"""
        if isinstance(other, Splits):
            return self.train == other.train and self.val == other.val and self.test == other.test
        return False

@dataclass
class Dataset:
    """
        a dataset consistes of splits and a keypoint definition.
        splits: the data is split up into subsets of the dataset
        kp_df: the dataformat how the keypoints are encoded
        has_debug: whether it has debug data (small subset of each split) to allow for faster debugging
                   the json files will then end with .small
    """
    splits: Splits
    kp_df: Union[KeypointsDefinition, None]
    has_debug: bool
    def __eq__(self,other):
        """Overrides the default implementation"""
        if isinstance(other, Dataset):
            return self.splits == other.splits and self.kp_df == other.kp_df and self.has_debug == other.has_debug
        return False

@dataclass
class Datasets:
    """all possible datasets ..."""
    KI_A = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_ki_a/3d/all_train_3d.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/SynPeDS/20230202_kia_mv.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_main/kia_anno/kia_instance_seg.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_ki_a/3d/all_validation_3d.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/SynPeDS/20230202_kia_mv.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_main/kia_anno/kia_instance_seg.dszip'
            ),
            test=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_ki_a/3d/all_test_3d.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/SynPeDS/20230202_kia_mv.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_main/kia_anno/kia_instance_seg.dszip'
            )
        ),
        has_debug=True,
        kp_df=KEYPOINT_DF_KI_A
    )
    KI_A_S = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_ki_a/3d/all_train_3d.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/SynPeDS/AWADA_MV2City_03222023_142008.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_main/kia_anno/kia_instance_seg.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_ki_a/3d/all_validation_3d.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/SynPeDS/AWADA_MV2City_03222023_142008.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_main/kia_anno/kia_instance_seg.dszip'
            ),
            test=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_ki_a/3d/all_test_3d.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/SynPeDS/AWADA_MV2City_03222023_142008.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}01_main/kia_anno/kia_instance_seg.dszip'
            )
        ),
        has_debug=True,
        kp_df=KEYPOINT_DF_KI_A
    )
    MSCOCO = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}02_mscoco/person_keypoints_train2017_fixed.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MSCOCO17/images/train2017.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}02_mscoco/person_keypoints_val2017_fixed.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MSCOCO17/images/val2017.dszip'
            ),
        ),
        has_debug=True,
        kp_df=KEYPOINT_DF_MSCOCO
    )
    PEDX = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}03_pedx/train_pedx.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}{ANNOTATION_SUBFOLDER}03_pedx/uncompressed.zip' # CAUTION: It has been moved to hi-c-001p2
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}03_pedx/vald_pedx.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}{ANNOTATION_SUBFOLDER}03_pedx/uncompressed.zip' # CAUTION: It has been moved to hi-c-001p2
            ),
            test=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}03_pedx/test_pedx.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}{ANNOTATION_SUBFOLDER}03_pedx/uncompressed.zip' # CAUTION: It has been moved to hi-c-001p2
            ),
        ),
        has_debug=False,
        kp_df=KEYPOINT_DF_PEDX
    )
    JTA = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_DATASETS"]}Incoming/JTA/annotations/jta_train_fixed.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/JTA/JTA_fixed.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_DATASETS"]}Incoming/JTA/annotations/jta_val_fixed.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/JTA/JTA_fixed.dszip'
            ),
            test=Split(
                annotation_path=f'{os.environ["RB_DATASETS"]}Incoming/JTA/annotations/jta_test_fixed.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/JTA/JTA_fixed.dszip'
            )
        ),
        has_debug=True,
        kp_df=KEYPOINT_DF_JTA
    )
    CITYPERSONS = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}01_main/annotations/cityperson/Annotation_citypersons_train.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/Cityscapes_leftImg8bit_fine.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}01_main/annotations/cityperson/Annotation_citypersons_val.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/Cityscapes_leftImg8bit_fine.dszip'
            )
        ),
        has_debug=False,
        kp_df=None
    )
    MOT17 = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}05_MOT17/train_half.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MOT17/MOT17.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}05_MOT17/val_half.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MOT17/MOT17.dszip'
            ),
        ),
        has_debug=False,
        kp_df=None
    )
    BDD100K = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}06_bdd100k/bdd100k_train_with_all.json',
                input_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}06_bdd100k/bdd100k.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}06_bdd100k/bdd100k_instance.dszip',
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}06_bdd100k/bdd100k_val_with_all.json',
                input_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}06_bdd100k/bdd100k.dszip',
                instance_mask_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}06_bdd100k/bdd100k_instance.dszip',
            )
        ),
        has_debug=False,
        kp_df=KEYPOINT_DF_BDD100K
    )
    MOTS = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}07_MOTS/mot_synth_train_v2.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MOTSynth/MOT_Synth_Sensor.dszip',
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}07_MOTS/mot_synth_val_v2.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MOTSynth/MOT_Synth_Sensor.dszip',
            ),
            test=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}07_MOTS/mot_synth_test_v2.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/MOTSynth/MOT_Synth_Sensor.dszip',
            ),
        ),
        has_debug=True,
        kp_df=KEYPOINT_DF_JTA
    )

    SHIFT = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}08_shift/shift_train.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/shift/shift_sensor.dszip',
                instance_mask_dszip=f'{os.environ["RB_DATASETS"]}Incoming/shift/shift_instance.dszip',
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}08_shift/shift_val.json',
                input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/shift/shift_sensor.dszip',
                instance_mask_dszip=f'{os.environ["RB_DATASETS"]}Incoming/shift/shift_instance.dszip',
            ),
        ),
        has_debug=True,
        kp_df=KEYPOINT_DF_MSCOCO
    )
    NUIMAGES = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}09_NuImages/nuimages_train.json',
                input_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}09_NuImages/front.dszip'
            ),
            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}09_NuImages/nuimages_val.json',
                input_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}09_NuImages/front.dszip'
            ),
            test=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}09_NuImages/nuimages_test.json',
                input_dszip=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}09_NuImages/front.dszip'
            )
        ),
        has_debug=False,
        kp_df=None
    )

    AGORA = Dataset(splits=Splits(
            train=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}10_AGORA/agora_train_all_n.json',
                    input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/AGORA/AGORA.dszip',
            ),

            val=Split(
                annotation_path=f'{os.environ["RB_MFSD_PATH"]}{ANNOTATION_SUBFOLDER}10_AGORA/agora_validation_all_n.json',
                    input_dszip=f'{os.environ["RB_DATASETS"]}Incoming/AGORA/AGORA.dszip',
            ),
            test=None
        ),
        has_debug=False,
        kp_df=KEYPOINT_AGORA
    )
