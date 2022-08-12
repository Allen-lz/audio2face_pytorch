# -*-coding:utf-8 -*-

import os
import math
import json

import librosa
import webrtcvad
import numpy as np
import scipy.io.wavfile as wf
import sys
sys.path.append(".")
sys.path.append("..")
from features.vad import VoiceActivityDetector

def voice_activate_indices_detect_2(audio_file):
    """
    使用 py-webrtcvad 进行音频VAD处理

    计算每个子带的对数能量，如果大于阈值，则对当前帧进行处理；否则直接将vad_flag置为0。
    计算每个子带对应的高斯概率，并与子带的权重相乘作为语音/噪声最终的概率。
    计算每个子带的对数似然比，
    每个子带的似然比会和阈值进行比较作为一个局部结果
    所有子带的对数加权似然比之和与阈值比较作一个全局的结果。当全局或局部其中一个为TRUE则认定当前帧是语音帧。
    使用hangover对结果进行平滑

    超过6个连续音频窗都不是语音，才视为静音片段

    :param audio_file:
    :return:
    """
    sample_window = 0.03
    sample_overlap = 0.03

    v = webrtcvad.Vad(3)
    rate, data = wf.read(audio_file)

    sample_start = 0
    detected_windows = np.array([])
    sample_window = int(rate * sample_window)
    sample_overlap = int(rate * sample_overlap)
    # 识别每个音频窗是否为语音
    while (sample_start < (len(data) - sample_window)):
        sample_end = sample_start + sample_window
        if sample_end >= len(data):
            sample_end = len(data) - 1
            sample_start = sample_end - sample_window - 1
        data_window = data[sample_start:sample_end]
        detected_windows = np.append(detected_windows, [sample_start, v.is_speech(data_window.tobytes(), rate)])
        sample_start += sample_overlap
    detected_windows = detected_windows.reshape(int(len(detected_windows) / 2), 2)

    indices = []
    viol_start, viol_end = -1, -1
    act_start, act_end = -1, -1
    interval_frames_threshold = 6
    for i, (_, flag) in enumerate(detected_windows):
        if flag == 0:
            viol_start = i if viol_start == -1 else viol_start
            viol_end = i
        elif viol_end - viol_start >= interval_frames_threshold:
            if act_start != -1:
                act_end = viol_start + 1
                indices.append((int(detected_windows[act_start, 0]), int(detected_windows[act_end, 0])))
            act_start = i - 2
            viol_start = -1
            viol_end = -1
        else:
            if act_start == -1:
                act_start = i
            act_end = i
            viol_start = -1
            viol_end = -1

    indices.append((int(detected_windows[act_start, 0]), int(detected_windows[act_end, 0])))

    return indices


def vad4():
    """
    音频文件VAD处理，静音片段识别，
    静音片段的ground truth置零，切分训练集时会将全零的样本去除
    置零前先进行标签平滑处理

    :return:
    """

    # 公开数据集
    raw_dir = "E:/datasets/audio2face/wav_base"
    gt_dir = "E:/datasets/audio2face/ground_truth_base"
    profile_dir = "E:/datasets/audio2face/profile_base"

    output_gt_dir = "E:/datasets/audio2face/clean_gt_base"

    rate = 16000

    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)

    total_duration = 0
    skip_video_count = 0
    process_video_count = 0
    for file in os.listdir(raw_dir):
        file_id, suffix = os.path.splitext(file)

        # 跳过非wav格式的文件
        if suffix != ".wav":
            continue

        # 文件路径
        _gt_file = os.path.join(gt_dir, file_id + ".npy")
        _profile_file = os.path.join(profile_dir, file_id + ".json")
        if not os.path.exists(_gt_file) or not os.path.exists(_profile_file):
            continue

        # 加载数据，gt标签，视频基本信息
        with open(_profile_file, "r", encoding="utf-8") as f:
            profile = json.load(f)
        gt_label = np.load(_gt_file)
        audio, sr = librosa.load(os.path.join(raw_dir, file), sr=rate)

        fps = profile["fps"]
        frames_step = rate / fps

        # 总帧数无法和音频长度对齐，丢弃
        if int(math.fabs(len(audio) / frames_step - profile["num_frames"])) > 2:
            skip_video_count += 1
            print("Skip {:<15s}, num_frames: {:>8d} , calculate frames: {:>8.2f}".format(
                file_id, int(profile["num_frames"]), len(audio) / frames_step))
            continue

        print("Processing {:<15s}, fps: {:<4.2f}, num_frames: {:<8d}".format(file_id, fps, int(profile["num_frames"])))
        process_video_count += 1

        # 去除空白音频信号
        # internal_clean_ind = voice_activate_indices_detect(os.path.join(raw_dir, file))
        internal_clean_ind = voice_activate_indices_detect_2(os.path.join(raw_dir, file))

        # label加hamming窗，平滑
        win_size = 5
        if gt_label.shape[0] >= win_size:
            win = np.hamming(win_size) / np.sum(np.hamming(win_size))
            for i in range(gt_label.shape[1]):
                gt_label[:, i] = np.convolve(gt_label[:, i], win, mode="same")

        # 重新构造gt标签，静音片段的gt置零
        new_gt_label = np.zeros_like(gt_label)
        for start, end in internal_clean_ind:
            frame_start_ind = round(start / frames_step)
            frame_end_ind = int(end // frames_step)
            frame_end_ind = min(frame_end_ind, len(gt_label) - 1)

            new_gt_label[frame_start_ind: frame_end_ind + 1] = gt_label[frame_start_ind: frame_end_ind + 1]

            duration = (frame_end_ind - frame_start_ind) / fps
            total_duration += duration

        # 输出保存数据
        np.save(os.path.join(output_gt_dir, file_id + ".npy"), new_gt_label)

    # 时长统计
    hours = int(total_duration // 3600)
    minutes = int((total_duration - hours * 3600) // 60)
    seconds = int(total_duration - hours * 3600 - minutes * 60)

    print("Total count: {}, Skip count: {}, Process count: {}, Duration: {:>3d}h {:>2d}m {:>2d}s".format(
        skip_video_count + process_video_count, skip_video_count, process_video_count, hours, minutes, seconds))


if __name__ == '__main__':
    vad4()
