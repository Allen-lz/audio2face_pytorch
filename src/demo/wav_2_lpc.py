# -*-coding:utf-8 -*-

"""
File    : wav_2_lpc.py
Time    : 2022/8/11 16:24
Author  : luzeng
"""
import os
from ctypes import *
import numpy as np

current_path = os.path.abspath(__file__)
dll_path_win = os.path.join(
    os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),
    'LPC.dll')
dll_path_linux = dll_path_win.replace('\\', '/')
dll = cdll.LoadLibrary(dll_path_linux)
LPC = dll.LPC


def get_audio_frames(audio_data, rate=16000, frames_per_second=30, chunks_length=260, audio_frameNum: int=None):
    # 取得音频文件采样频率
    # 取得音频数据
    signal = audio_data
    # signal = list(signal)

    # 视频fps frames_per_second
    # 音频分割，520ms chunks_length
    if audio_frameNum is None:
        audio_frameNum = int(len(signal) / rate * frames_per_second)  # 计算音频对应的视频帧数

    chunks_signal_samples = chunks_length * rate // 1000

    # 前后各添加260ms音频
    a = np.zeros(chunks_length * rate // 1000, dtype=np.int16)
    signal = np.hstack((a, signal, a))

    # 归一化
    # signal = signal / (2. ** 15)
    frames_step = 1000.0 / frames_per_second  # 计算每个视频帧多少毫秒，视频每帧的时长间隔33.3333ms
    rate_kHz = int(rate / 1000)  # 计算每毫秒多少个音频赫兹，采样率：16kHz

    # 分割音频
    audio_frames = [
        signal[round(i * frames_step * rate_kHz): round((i * frames_step * rate_kHz) + 2 * chunks_signal_samples)] \
            if int((i * frames_step * rate_kHz) + 2 * chunks_signal_samples) < len(signal) \
            else signal[len(signal) - 2 * chunks_signal_samples:]
        for i in range(audio_frameNum)
    ]

    return audio_frames


def c_lpc(audio_frames_data, rate=16000, frames_per_second=30, chunks_length=260, overlap_frames_apart=0.008,
          numberOfFrames=64):
    input_data_list = []
    for audio_frame in audio_frames_data:
        overlap = int(rate * overlap_frames_apart)
        frameSize = int(rate * overlap_frames_apart * 2)
        numberOfFrames = (len(audio_frame) - frameSize) // overlap + 1

        # 构造音频帧
        frames = np.ndarray((numberOfFrames, frameSize))
        for j in range(0, numberOfFrames):
            frames[j] = audio_frame[j * overlap: j * overlap + frameSize]

        frames *= np.hanning(frameSize)

        frames_lpc_features = []
        # a = (c_double * frameSize)()
        b = (c_double * 32)()
        for k in range(0, numberOfFrames):
            a = (c_double * frameSize)(*frames[k])
            LPC(pointer(a), frameSize, 32, pointer(b))
            frames_lpc_features.append(list(b))

        image_temp1 = np.array(frames_lpc_features)  # list2array
        image_temp2 = np.expand_dims(image_temp1, axis=0)  # 升维
        input_data_list.append(image_temp2)

    inputData_array = np.concatenate(input_data_list, axis=0)
    inputData_array = inputData_array.transpose((0, 2, 1))

    # 扩展为4维:9000*32*64*1
    inputData_array = np.expand_dims(inputData_array, axis=3)
    return inputData_array
