# -*-coding:utf-8 -*-

"""
# File   : features.py
# Time   : 2022/8/6 14:33
# Author : Lu Zeng
# version: python 3.7
"""

import os
import re
import tqdm
import json

import cv2
import skvideo.io
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from mocap4face.mocap4face import MediapipeFaceDetection
#
MOUTH_RELATED_BLENDSHAPE_LIST = [
    "jawOpen",
    "mouthFunnel",
    "mouthPucker",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
]


def call_mocapface(model, img_src):
    """
    调用mocapface模型，输出blendshape结果
    只选取嘴型有效的blendshape

    :param model: mocapface模型
    :param img_src: 输入图像，BGR通道
    :return:
    """
    try:
        # 先进行人脸的检测, 检测不到人脸
        result, _ = model.MediapipeRun(img_src)
    except Exception as e:
        result = None

    if result is None:
        return [0.] * len(MOUTH_RELATED_BLENDSHAPE_LIST)

    face_json = model.jsonFormat(result)
    # 从mocapface中将得到的结果从里面拿出来
    # gt_label的顺序和MOUTH_RELATED_BLENDSHAPE_LIST中的顺序是一样的
    gt_label = [face_json.get(name, 0.) for name in MOUTH_RELATED_BLENDSHAPE_LIST]
    return gt_label


@DeprecationWarning
def makevideo2(video_file, _model: MediapipeFaceDetection):
    """
    已弃用
    cv2读取视频会跳过部分重复帧或者失败帧，导致最后帧数与音频时长对应不上，无法对齐数据
    :param video_file:
    :param _model:
    :return:
    """
    capture = cv2.VideoCapture(video_file)

    video_profile_info = {
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "channel": capture.get(cv2.CAP_PROP_CHANNEL),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "num_frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
    }
    print(video_profile_info)

    def image_iterator(cap):
        while True:
            _ret, _img = cap.read()
            if not _ret:
                break
            yield _img

    frame_id = 0
    ground_truth_list = []
    if capture.isOpened():
        for img_src in tqdm.tqdm(image_iterator(capture)):
            frame_id += 1
            gt_label = call_mocapface(_model, img_src)
            ground_truth_list.append(gt_label)
    else:
        print('视频打开失败！')

    gt_matrix = np.array(ground_truth_list) / 100.0
    if gt_matrix.shape[0] != video_profile_info["num_frames"]:
        video_profile_info["num_frames"] = gt_matrix.shape[0]

    return gt_matrix, video_profile_info


def makevideo3(video_file, _model: MediapipeFaceDetection):
    """
    cv2 获取视频基本信息
    skvideo 读取视频每一帧(注意: 不会漏帧)

    :param video_file:
    :param _model:
    :return:
    """
    capture = cv2.VideoCapture(video_file)
    video_profile_info = {
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "channel": capture.get(cv2.CAP_PROP_CHANNEL),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "num_frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
    }
    print(video_file, video_profile_info)
    capture.release()

    videogen = skvideo.io.vreader(video_file)

    frame_id = 0
    ground_truth_list = []
    for img_src in tqdm.tqdm(videogen):
        frame_id += 1
        # skvideo读取数据为RGB
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        # 没有人脸的话这里直接返回0, 但是不会丢弃帧
        gt_label = call_mocapface(_model, img_src)
        ground_truth_list.append(gt_label)

    gt_matrix = np.array(ground_truth_list) / 100.0
    if gt_matrix.shape[0] != video_profile_info["num_frames"]:
        video_profile_info["num_frames"] = gt_matrix.shape[0]

    return gt_matrix, video_profile_info


def get_duplicated_name(root_dir, output_name, output_suffix):
    """
    重名文件 加数字编号后缀

    :param root_dir:
    :param output_name:
    :param output_suffix:
    :return:
    """
    # 防止重名
    output_file = os.path.join(root_dir, output_name + output_suffix)
    dup_ind = 1
    while os.path.exists(output_file):
        output_file = os.path.join(root_dir, output_name + "." + str(dup_ind) + output_suffix)
        dup_ind += 1
    return output_file


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    video_data_dir_list = [
        "E:/datasets/audio2face/wanghong_short_video",
    ]
    # 创建生成数据的时候要保存的目录
    wav_data_dir = "E:/datasets/audio2face/wav_base"
    profile_data_dir = "E:/datasets/audio2face/profile_base"
    gt_data_dir = "E:/datasets/audio2face/ground_truth_base"

    make_dir(wav_data_dir)
    make_dir(profile_data_dir)
    make_dir(gt_data_dir)

    def get_sub_files(path):
        """
        这是一个通过递归来收集一个文件夹中的文件的函数(遇到文件夹就继续递归调用, 遇到文件就将其加入到list中)
        """
        sub_files = os.listdir(path)
        all_files = []
        for file in sub_files:
            abs_file = os.path.join(path, file)
            if os.path.isfile(abs_file):
                all_files.append(abs_file)
            if os.path.isdir(abs_file):
                _files = get_sub_files(abs_file)
                all_files.extend(_files)
        return all_files

    # 建立mocapface的model用于打标注
    model = MediapipeFaceDetection(
        tflite_path="./mocap4face/2001161359.tflite",
        json_path="./mocap4face/2001161359.json")

    # 这里说话的人和他的语音是一一对应的
    speakers_mapping = {}
    for video_dir in video_data_dir_list:
        for i, file in enumerate(get_sub_files(video_dir)):

            # 1. 得到各种文件的后缀
            abs_prefix, suffix = os.path.splitext(file)
            prefix, _ = os.path.splitext(os.path.split(file)[-1])
            if suffix != ".avi" and suffix != ".mpg" and suffix != ".mp4":
                continue
            output_prefix = "_".join(file.removeprefix(video_dir).split(os.path.sep)[1:-1]) + "_" + prefix

            # 音频分离保存，采样率降采样为16kHz，单通道
            # 防止重名并更改后缀
            output_wav_file = get_duplicated_name(wav_data_dir, output_prefix, ".wav")
            # 这里统一将音频的采样率设置为了16k
            os.system("ffmpeg -i {} -f wav -ar 16000 -ac 1 {} -y".format(file, output_wav_file))

            # mocapface 识别结果
            # gt_matrix, video_profile_info = makevideo2(file, model)
            gt_matrix, video_profile_info = makevideo3(file, model)

            # 保存GT数据
            gt_output_file = get_duplicated_name(gt_data_dir, output_prefix, ".npy")
            np.save(gt_output_file, gt_matrix)

            # 保存json
            profile_output_file = get_duplicated_name(profile_data_dir, output_prefix, ".json")
            with open(profile_output_file, "w", encoding="utf-8") as f:
                json.dump(video_profile_info, f)

            # 说话人ID
            speaker_id = prefix.split("_")[0].lower()
            speaker_id = re.sub(r"\d+", "", speaker_id) \
                if re.match(r"^[a-z\u4e00-\u9fa5]+\d+$", speaker_id) \
                else speaker_id
            speakers_mapping[output_prefix] = speaker_id

    with open(os.path.join(profile_data_dir, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump(speakers_mapping, f, ensure_ascii=False, indent=4)
