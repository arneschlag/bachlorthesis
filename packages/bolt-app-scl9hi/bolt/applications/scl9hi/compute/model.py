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

# Parameter
#IMG_SIZE = [640, 960] # old
IMG_SIZE = [630, 1120] # new 16:9

import functools
from typing import Any

from bolt.applications.keypoint_detection.data.keypoint import KeypointsDefinition
from bolt.applications.scl9hi.preprocessing.datasets import Dataset
import torch, einops
import torch.nn as nn
import torchvision.models.resnet
import torchvision.models


from bolt.applications.object_detection.v1.backends.torch.models.building import classification_head, regression_head
from bolt.applications.scl9hi.preprocessing.expr_builder import ExperimentEnv, ExperimentBuilder
import bolt.applications.object_detection.v1.backends.torch.models.one_shot
import bolt.backends.torch.augment
import bolt.backends.torch.models.building
import bolt.backends.torch.models.aggregation

class Model:
    def __init__(self, exp: ExperimentEnv, kp_df) -> None:
        self.kp_df = kp_df
        self.exp: ExperimentEnv = exp

    def build_heads(self):
        depth = 4
        filters = 256
        n_classes = 2

        prediction_layers = [
            functools.partial(nn.Conv2d, kernel_size=3, stride=1, padding="same", out_channels=filters),
            nn.ReLU,
        ] * depth

        self.heads = [
            regression_head(prediction_layers=prediction_layers, shared=True, name="bbox_coords"),  # Bounding Box
            classification_head(prediction_layers=prediction_layers, n_classes=n_classes, shared=True, name="classes"),
        ]

        # 2D Keypoints
        if self.exp.kp_2d:
            self.heads.append(
                regression_head(prediction_layers=prediction_layers, shared=not self.exp.no_shared_weighting, n_values=2 * self.kp_df.num_keypoints , name="keypoints")
            )

        # 3D Keypoints
        if self.exp.kp_3d:
            self.heads.append(
                regression_head(prediction_layers=prediction_layers, shared=not self.exp.no_shared_weighting, n_values=3 * self.kp_df.num_keypoints, name="keypoints_3d")
            )
            if self.exp.abs_kp3d:
                self.heads.append(
                    regression_head(prediction_layers=prediction_layers, shared=not self.exp.no_shared_weighting, n_values=3, name="abs_3d")
                )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.exp.model == "v1":
            # Resnet 50
            feature_extr = bolt.backends.torch.models.building.get_feature_extractor_layers(
                infer_channels=bolt.backends.torch.models.building.infer_resnet50_out_channels,
                layers=["layer2", "layer3", "layer4"],
                feature_extractor=torchvision.models.resnet.resnet50(weights=self.exp.weights),
            )
        elif self.exp.model == "v2":
            # Swin Transformer V2
            feature_extr = bolt.backends.torch.models.building.SwinPyramidGetter(
                swin=torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1),
                layers=[3, 5, 7]
            )
        elif self.exp.model == "v3":
            feature_extr = bolt.backends.torch.models.building.SwinPyramidGetter(
                swin=torchvision.models.swin_v2_t(weights=torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1),
                layers=[3, 5, 7]
            )
        elif self.exp.model == "v4":
            feature_extr = bolt.backends.torch.models.building.ConvNextPyramidGetter(
                feature_extractor=torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            )

        self.build_heads()

        if self.exp.freeze_backend:
            feature_extr.freeze()

        return nn.modules.container.Sequential(
            bolt.backends.torch.augment.ConvertAndScaleImage(),
            bolt.backends.torch.augment.object_detection_augmentation(),
            torchvision.transforms.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            bolt.applications.object_detection.v1.backends.torch.models.one_shot.RetinaNet(
                detector_heads=self.heads,
                input_resolution=IMG_SIZE,
                n_classes=2,
                pyramid_fn=functools.partial(
                    bolt.backends.torch.models.aggregation.fpn,
                    add_extension_to_pyramid=True,
                    pyramid_channels=256,
                ) if self.exp.model  == "v1" else None,
                feature_extractor=feature_extr
            ),
        )
