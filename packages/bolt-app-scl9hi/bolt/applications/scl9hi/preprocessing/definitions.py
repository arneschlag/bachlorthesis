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

from bolt.applications.keypoint_detection.data.keypoint import KeypointsDefinition

SAME_KP = {
    "Left Eye": {"color": [150, 255, 150], "constant": 0.025},
    "Right Eye": {"color": [50, 255, 50], "constant": 0.25},
    "Left Shoulder": {"color": [100, 150, 100], "constant": 0.079},
    "Right Shoulder": {"color": [250, 0, 0], "constant": 0.079},
    "Left Elbow": {"color": [255, 255, 0], "constant": 0.072},
    "Right Elbow": {"color": [175, 0, 0], "constant": 0.072},
    "Left Wrist": {"color": [80, 0, 0], "constant": 0.062},
    "Right Wrist": {"color": [175, 175, 0], "constant": 0.062},
    "Right Hip": {"color": [80, 80, 0], "constant": 0.107},
    "Left Hip": {"color": [0, 0, 250], "constant": 0.107},
    "Left Knee": {"color": [0, 250, 0], "constant": 0.087},
    "Right Knee": {"color": [0, 0, 175], "constant": 0.087},
    "Left Ankle": {"color": [0, 175, 0], "constant": 0.089},
    "Right Ankle": {"color": [0, 0, 80], "constant": 0.089},
}
ki_a_names = {
    0: "pelvis",
    1: "clavicle_midpoint",
    2: "Right Hip",
    3: "Right Knee",
    4: "Right Ankle",
    5: "toe_r",
    6: "Left Hip",
    7: "Left Knee",
    8: "Left Ankle",
    9: "toe_l",
    10: "neck",
    11: "Left Shoulder",
    12: "Left Elbow",
    13: "Left Wrist",
    14: "Right Shoulder",
    15: "Right Elbow",
    16: "Right Wrist",
    17: "Right Eye",
    18: "Left Eye",
}
coco_names = {
    0: "nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "l_ear",
    4: "r_ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}
pedx_names = {
    0: "nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "head",
    4: "Left Knee",
    5: "neck",
    6: "Left Wrist",
    7: "Left Ankle",
    8: "Left Hip",
    9: "mouth",
    10: "Right Wrist",
    11: "Right Knee",
    12: "Left Elbow",
    13: "Left Shoulder",
    14: "Right Hip",
    15: "Right Ankle",
    16: "Right Elbow",
    17: "Right Shoulder",
}

jta_names = {
    0: "Head_Top",
    1: "head_center",
    2: "neck",
    3: "right_clavicle",
    4: "Right Shoulder",
    5: "Right Elbow",
    6: "Right Wrist",
    7: "left_clavicle",
    8: "Left Shoulder",
    9: "Left Elbow",
    10: "Left Wrist",
    11: "spine0",
    12: "spine1",
    13: "spine2",
    14: "spine3",
    15: "spine4",
    16: "Right Hip",
    17: "Right Knee",
    18: "Right Ankle",
    19: "Left Hip",
    20: "Left Knee",
    21: "Left Ankle",
}
bd100k_names = {
    0: "head",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "right_hip",
    9: "right_knee",
    10: "right_ankle",
    11: "left_hip",
    12: "left_knee",
    13: "left_ankle",
    14: "right_hand",
    15: "left_hand",
    16: "right_foot",
    17: "left_foot",
}
agora_names = {
    0: "spine_0",
    1: "left_pelvis",
    2: "right_pelvis",
    3: "spine_1",
    4: "left_knee",
    5: "right_knee",
    6: "spine_2",
    7: "left_ankle_1",
    8: "right_ankle_1",
    9: "9",
    10: "10",
    11: "11",
    12: "neck",
    13: "13",
    14: "14",
    15: "bottom_head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
    22: "left_hand_0",
    23: "right_hand_0",
    24: "nose",
    25: "left_eye",
    26: "26",
    27: "right_ear",
    28: "left_ear",
    29: "left_feet_0",
    30: "left_feet_1",
    31: "left_ankle_0",
    32: "right_feet_1",
    33: "right_feet_0",
    34: "34",
    35: "35",
    36: "36",
    37: "37",
    38: "38",
    39: "39",
    40: "40",
    41: "41",
    42: "42",
    43: "43",
    44: "44",
}


def generate_color(names):
    """Will lookup the color of the keypoint.

    Parameters
    ----------
    names (str): names of the keypoints

    Returns
    -------
    dict: with the colors of the keypoints
    """
    color_dict = {}
    i = 0
    for kp in names.values():
        if kp in SAME_KP:
            # TODO
            color_dict[i] = [255, 0, 0]
        else:
            color_dict[i] = [255, 0, 0]
        i += 1
    return color_dict


def generate_constants(names):
    """Will generate constants for evaluation.

    Parameters
    ----------
    names (str): names of keypoints

    Returns
    -------
    list: list of constants
    """
    constants = []
    for kp in names.values():
        if kp in SAME_KP:
            constants.append(SAME_KP[kp]["constant"])
        else:
            constants.append(0.026)  # apply default like nose
    return constants


KEYPOINT_DF_KI_A = KeypointsDefinition(
    keypoint_names=ki_a_names,
    color_scheme=generate_color(ki_a_names),
    limbs=[
        [0, 1],
        [0, 2],
        [0, 6],
        [2, 3],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [8, 9],
        [1, 10],
        [10, 14],
        [10, 11],
        [10, 18],
        [10, 17],
        [11, 12],
        [12, 13],
        [14, 15],
        [15, 16],
    ],
    # see https://cocodataset.org/#keypoints-eval 2.2
    constants=generate_constants(ki_a_names),
    root_kp_name="neck"
)

KEYPOINT_DF_MSCOCO = KeypointsDefinition(
    keypoint_names=coco_names,
    color_scheme=generate_color(coco_names),
    limbs=[
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [6, 12],
        [5, 11],
        [6, 8],
        [8, 10],
        [5, 7],
        [7, 9],
        [13, 15],
        [14, 16],
        [12, 14],
        [13, 11],
        [11, 12],
        [5, 3],
        [4, 6],
    ],
    constants=generate_constants(coco_names),
    root_kp_name="nose"
)
KEYPOINT_DF_PEDX = KeypointsDefinition(
    keypoint_names=pedx_names,
    color_scheme=generate_color(pedx_names),
    limbs=[
        [3, 1],
        [3, 2],
        [3, 0],
        [9, 5],
        [3, 13],
        [3, 17],
        [12, 13],
        [12, 6],
        [17, 16],
        [10, 16],
        [13, 8],
        [17, 14],
        [8, 14],
        [8, 4],
        [7, 4],
        [11, 14],
        [11, 15],
        [13, 17],
    ],
    constants=generate_constants(pedx_names),
    root_kp_name="neck"
)

KEYPOINT_DF_JTA = KeypointsDefinition(
    keypoint_names=jta_names,
    color_scheme=generate_color(jta_names),
    limbs=[
        (0, 1),  # head_top -> head_center
        (1, 2),  # head_center -> neck
        (2, 3),  # neck -> right_clavicle
        (3, 4),  # right_clavicle -> right_shoulder
        (4, 5),  # right_shoulder -> right_elbow
        (5, 6),  # right_elbow -> right_wrist
        (2, 7),  # neck -> left_clavicle
        (7, 8),  # left_clavicle -> left_shoulder
        (8, 9),  # left_shoulder -> left_elbow
        (9, 10),  # left_elbow -> left_wrist
        (2, 11),  # neck -> spine0
        (11, 12),  # spine0 -> spine1
        (12, 13),  # spine1 -> spine2
        (13, 14),  # spine2 -> spine3
        (14, 15),  # spine3 -> spine4
        (15, 16),  # spine4 -> right_hip
        (16, 17),  # right_hip -> right_knee
        (17, 18),  # right_knee -> right_ankle
        (15, 19),  # spine4 -> left_hip
        (19, 20),  # left_hip -> left_knee
        (20, 21),  # left_knee -> left_ankle
    ],
    constants=generate_constants(jta_names),
    root_kp_name="neck"
)

KEYPOINT_DF_BDD100K = KeypointsDefinition(
    keypoint_names=bd100k_names,
    color_scheme=generate_color(bd100k_names),
    limbs=[(0,1),
           (1,2),
           (2,3),
           (3,4),
           (4,14),
           (1,5),
           (5,6),
           (6,7),
           (7,15),
           (1,11),
           (1,8),
           (12,11),
           (8,11),
           (8,9),
           (12,13),
           (13,17),
           (9,10),
           (10,16)], # TODO
    constants=generate_constants(bd100k_names),
    root_kp_name="neck"
)

KEYPOINT_AGORA = KeypointsDefinition(
    keypoint_names=agora_names,
    color_scheme=generate_color(agora_names),
    limbs=[(1,2),
           (1, 4),
           (4, 31),
           (31, 30),
           (2, 5),
           (5, 8),
           (32, 8),
           (1, 0),
           (2, 0),
           (3, 0),
           (3, 6),
           (12, 6),
           (12, 15),
           (28, 15),
           (27, 15),
           (24, 15),
           (25, 15),
           (26, 15),
           (12, 17),
           (12, 16),
           (18, 16),
           (18, 20),
           (17, 19),
           (21, 19)], # TODO
    constants=generate_constants(agora_names),
    root_kp_name="neck"
)
