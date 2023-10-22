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

import torch.nn.modules.container

import bolt.applications.object_detection.v1.backends.torch.encodings.rcnn
from bolt.applications.object_detection.v1.losses import AttributeLoss
import bolt.backends.torch.losses.mask


def mse_loss_with_nans(input_tensor, target, dims=2):
    """Ignores keypoints that are missing for the loss.

    Parameters
    ----------
    input : torch tensor
        the output of the NN
    target : torch tensor
        the target

    Returns
    -------
    torch tensor
        the mse loss
    """
    # Missing data are 0's
    # target.reshape((-1, 17))
    # mask = target == 0
    # valid_mask = torch.logical_not(target.isnan()) Simon suggestion
    target_reshaped = target.reshape((target.shape[0], -1, dims))
    valid_mask = torch.logical_not(target_reshaped.isnan().sum(axis=-1))
    valid_mask = torch.repeat_interleave(valid_mask, repeats=dims, dim=1)

    out = (input_tensor[valid_mask] - target[valid_mask]) ** 2
    if out.nelement() == 0:
        out = torch.tensor([0], dtype=torch.float32).to(device="cuda")

    loss = out.mean()

    return loss

def l1_loss_with_nans(input_tensor, target, dims=3):
    """Ignores keypoints that are missing for the loss.

    Parameters
    ----------
    input : torch tensor
        the output of the NN
    target : torch tensor
        the target

    Returns
    -------
    torch tensor
        the mse loss
    """
    # Missing data are 0's
    # target.reshape((-1, 17))
    # mask = target == 0
    # valid_mask = torch.logical_not(target.isnan()) Simon suggestion
    target_reshaped = target.reshape((target.shape[0], -1, dims))
    valid_mask = torch.logical_not(target_reshaped.isnan().sum(axis=-1))
    valid_mask = torch.repeat_interleave(valid_mask, repeats=dims, dim=1)

    out = abs(input_tensor[valid_mask] - target[valid_mask])
    if out.nelement() == 0:
        out = torch.tensor([0], dtype=torch.float32).to(device="cuda")

    loss = out.mean()

    return loss

def mse_loss_with_nans_3d(input_tensor, target):
    """Ignores keypoints that are missing for the loss.

    Parameters
    ----------
    input : torch tensor
        the output of the NN
    target : torch tensor
        the target

    Returns
    -------
    torch tensor
        the mse loss
    """
    return mse_loss_with_nans(input_tensor, target, dims=3)

def build_losses(pose_2d: bool = False, pose_3d: bool = False, abs_3d: bool = False, custom_weights: bool = False):
    """Builds the losses, depended on whether.

    keypoints are also used for training.

    Parameters
    ----------
    train_with_kp : bool
        are keypoints used?

    Returns
    -------
    dict with losses
    """
    losses = dict(
        bbox_coords=AttributeLoss(
            loss_function=bolt.backends.torch.losses.mask.MaskEmptyInput(loss=torch.nn.modules.loss.SmoothL1Loss()),
            # this transforms them to center width height oriented
            transform_true=bolt.applications.object_detection.v1.backends.torch.encodings.rcnn.encode,
            weight=1.0
        ),
        classes=AttributeLoss(
            loss_function=torch.nn.modules.loss.CrossEntropyLoss(),
            needs_negatives=True,
            weight=0.1 if custom_weights else 1.0
        ),
    )
    if pose_2d:
        losses["keypoints"] = AttributeLoss(
            loss_function=bolt.backends.torch.losses.mask.MaskEmptyInput(loss=mse_loss_with_nans),
            transform_true=bolt.applications.object_detection.v1.backends.torch.encodings.rcnn.point_encode,
            weight=1.0
        )
    if pose_3d:
        losses["keypoints_3d"] = AttributeLoss(
            loss_function=bolt.backends.torch.losses.mask.MaskEmptyInput(loss=mse_loss_with_nans_3d),
            #transform_true=bolt.applications.object_detection.v1.backends.torch.encodings.rcnn.point_encode_3d,
            #needs_negatives=True
            weight=1.0
        )
        if abs_3d:
            losses["abs_3d"] = AttributeLoss(
                loss_function=bolt.backends.torch.losses.mask.MaskEmptyInput(loss=l1_loss_with_nans),
                #transform_true=bolt.applications.object_detection.v1.backends.torch.encodings.rcnn.point_encode_3d,
                #needs_negatives=True
                weight=1.0
            )
    return losses
