# -*-coding:utf-8 -*-

"""
# File   : util.py
# Time   : 2022/7/8 16:35
# Author : Xu Jiajian
# version: python 3.6
"""

import os
import math
import json

import numpy as np
import pandas as pd

current_dir = os.path.split(os.path.abspath(__file__))[0]

FACEGOOD_BS_CONUNT = 116
# the sort of bs name correspond to UE input sort
bs_name_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105,
                 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 1, 115]
label_name_list = pd.read_csv(
    os.path.join(current_dir, "doc", "bsname.txt"), encoding="utf-8").values.transpose()[0].tolist()

STANDARD_ARKIT_BS_NAME = ["BlendShapeCount", "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
                          "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight",
                          "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
                          "JawForward", "JawRight", "JawLeft", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker",
                          "MouthRight", "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
                          "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
                          "MouthStretchRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
                          "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
                          "MouthUpperUpLeft", "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
                          "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
                          "NoseSneerLeft", "NoseSneerRight", "TongueOut", "HeadYaw", "HeadPitch", "HeadRoll",
                          "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll", ]

VALID_BS_NAME = [
    "JawForward",
    "JawLeft",
    "JawRight",
    "JawOpen",
    "MouthFunnel",
    "MouthPucker",
    "MouthLeft",
    "MouthRight",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight"
]

SELECT_VALID_BS_NAME = [
    "JawOpen",
    "MouthFunnel",
    "MouthPucker",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
]


def add_noise(origin_signal, snr):
    """
    添加高斯白噪声，固定信噪比
    """
    noise = np.random.normal(0, 1, len(origin_signal))

    # 计算语音信号功率Ps和噪声功率Pn1
    Ps = np.sum(origin_signal ** 2) / len(origin_signal)
    Pn1 = np.sum(noise ** 2) / len(noise)

    # 计算k值
    k = math.sqrt(Ps / (10 ** (snr / 10) * Pn1))

    # 将噪声数据乘以k,
    random_values_we_need = noise * k

    new_signal = origin_signal.astype(np.float64) + random_values_we_need

    return new_signal


def add_other_noise(origin_signal, noise):
    """
    添加指定噪声
    """
    if len(origin_signal) / len(noise) > 1:
        new_noise = np.concatenate([noise] * int(len(origin_signal) / len(noise) + 1))
        new_noise = new_noise[:len(origin_signal)]
    else:
        upper = len(noise) - len(origin_signal)
        start = np.random.randint(0, upper - 1)
        new_noise = noise[start:start + len(origin_signal)]
    new_signal = origin_signal + new_noise
    return new_signal


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    return profile


def rectangle_wav(wav):
    """
    将波形信号变为矩形波信号，
    主要用于将时序BlendShape数值进行增强
    """
    rect_wav = np.zeros_like(wav)
    extremum_indices = []
    for t in range(1, len(wav) - 2):
        # 趋势是否改变
        is_change_slope = (wav[t + 1] - wav[t] + 1e-16) / (wav[t] - wav[t - 1] + 1e-16)
        if is_change_slope < 0:
            extremum_indices.append(t)

    # 常量信号，无波形
    if not extremum_indices:
        rect_wav[:] = wav[:]
        return rect_wav

    # 每个极值区间进行赋值
    for i, ind in enumerate(extremum_indices):
        if i == 0:
            start = 0
        else:
            start = int((ind + extremum_indices[i - 1]) / 2)

        if i == len(extremum_indices) - 1:
            end = wav.shape[0]
        else:
            end = int((ind + extremum_indices[i + 1]) / 2)
        rect_wav[start:end] = wav[ind]

    return rect_wav


def facegood_bs_label_to_valid_arkit(label_temp):
    """
    FACEGOOD样例数据转换成标准ARKITS表情

    :param label_temp:
    :return:
    """
    _label = np.zeros((label_temp.shape[0], FACEGOOD_BS_CONUNT))
    for i in range(len(bs_name_index)):
        _label[:, i] = label_temp[:, bs_name_index[i]]

    num_valid_bs = 26
    new_label = np.zeros((_label.shape[0], num_valid_bs), dtype=np.float32)
    new_label[:, 0] = _label[:, label_name_list.index("jaw_thrust_c")]
    new_label[:, 1] = _label[:, label_name_list.index("jaw_sideways_l")]
    new_label[:, 2] = _label[:, label_name_list.index("jaw_sideways_r")]
    new_label[:, 3] = _label[:, label_name_list.index("mouth_stretch_c")]
    # new_label[:, 4] = _label[:, label_name_list.index("mouth_chew_c")]
    new_label[:, 4] = np.max(_label[:, [label_name_list.index(n)
                                        for n in ["mouth_funnel_dl", "mouth_funnel_dr", "mouth_funnel_ul",
                                                  "mouth_funnel_ur"]]], axis=1)
    new_label[:, 5] = np.max(
        _label[:, [label_name_list.index(n) for n in ["mouth_pucker_l", "mouth_pucker_r"]]], axis=1)
    new_label[:, 6] = _label[:, label_name_list.index("mouth_sideways_l")]
    new_label[:, 7] = _label[:, label_name_list.index("mouth_sideways_r")]
    new_label[:, 8] = _label[:, label_name_list.index("mouth_lipCornerPull_l")]
    new_label[:, 9] = _label[:, label_name_list.index("mouth_lipCornerPull_r")]
    new_label[:, 10] = np.max(_label[:, [label_name_list.index(n) for n in ["mouth_lipCornerDepress_l",
                                                                            "mouth_lipCornerDepressFix_l"]]],
                              axis=1)
    new_label[:, 11] = np.max(_label[:, [label_name_list.index(n) for n in ["mouth_lipCornerDepress_r",
                                                                            "mouth_lipCornerDepressFix_r"]]],
                              axis=1)
    new_label[:, 12] = _label[:, label_name_list.index("mouth_dimple_l")]
    new_label[:, 13] = _label[:, label_name_list.index("mouth_dimple_r")]
    new_label[:, 14] = _label[:, label_name_list.index("mouth_lipStretch_l")]
    new_label[:, 15] = _label[:, label_name_list.index("mouth_lipStretch_r")]
    new_label[:, 16] = np.max(
        _label[:, [label_name_list.index(n) for n in ["mouth_suck_dl", "mouth_suck_dr"]]], axis=1)
    new_label[:, 17] = np.max(
        _label[:, [label_name_list.index(n) for n in ["mouth_suck_ul", "mouth_suck_ur"]]], axis=1)
    new_label[:, 18] = _label[:, label_name_list.index("mouth_chinRaise_d")]
    new_label[:, 19] = _label[:, label_name_list.index("mouth_chinRaise_u")]
    new_label[:, 20] = _label[:, label_name_list.index("mouth_press_l")]
    new_label[:, 21] = _label[:, label_name_list.index("mouth_press_r")]
    new_label[:, 22] = _label[:, label_name_list.index("mouth_lowerLipDepress_l")]
    new_label[:, 23] = _label[:, label_name_list.index("mouth_lowerLipDepress_r")]
    new_label[:, 24] = _label[:, label_name_list.index("mouth_upperLipRaise_l")]
    new_label[:, 25] = _label[:, label_name_list.index("mouth_upperLipRaise_r")]

    return new_label


def standard_arkit_bs_to_valid(label_temp):
    """
    标准ARKITS表情，抽取有效的嘴部动作

    :param label_temp:
    :return:
    """
    num_valid_bs = len(VALID_BS_NAME)
    indices = [STANDARD_ARKIT_BS_NAME.index(bs) for bs in VALID_BS_NAME]
    # indices = [STANDARD_ARKIT_BS_NAME.index(bs) for bs in SELECT_VALID_BS_NAME]
    new_label = np.zeros((label_temp.shape[0], num_valid_bs), dtype=np.float32)
    new_label[:] = label_temp[:, indices]
    return new_label
