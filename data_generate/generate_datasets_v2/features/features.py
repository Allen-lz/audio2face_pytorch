# -*-coding:utf-8 -*-

"""
音频处理的可以看看这个博客: https://www.cnblogs.com/LXP-Never/p/11561355.html
天池零基础入门音频的教程: https://pythontechworld.com/article/detail/BCcNjuLDVYa6
分帧函数: https://blog.csdn.net/qq_37653144/article/details/89045363
"""

import os
from ctypes import *

import tqdm
import librosa
import numpy as np

current_dir = os.path.split(os.path.abspath(__file__))[0]
lpc_dll_file = os.path.join(current_dir, "LPC.dll")
lpc_dll = cdll.LoadLibrary(lpc_dll_file)


def lpc(audio_frames, sample_rate=16000):
    input_data_list = []
    for audio_frame in tqdm.tqdm(audio_frames):
        # 8ms帧移， 16ms帧长
        overlap_frames_apart = 0.008
        overlap = int(sample_rate * overlap_frames_apart)
        frameSize = int(sample_rate * overlap_frames_apart * 2)
        numberOfFrames = (len(audio_frame) - frameSize) // overlap + 1

        # 构造音频帧
        # print(numberOfFrames, frameSize)
        frames = np.ndarray((numberOfFrames, frameSize))
        for j in range(0, numberOfFrames):
            frames[j] = audio_frame[j * overlap: j * overlap + frameSize]

        # 加窗
        frames *= np.hanning(frameSize)

        # LPC
        frames_lpc_features = []
        b = (c_double * 32)()
        for fr in frames:
            a = (c_double * frameSize)(*fr)
            # LPC(float *in, int size, int order, float *out)
            lpc_dll.LPC(pointer(a), frameSize, 32, pointer(b));
            frames_lpc_features.append(list(b))
            del a

        del b

        image_temp1 = np.array(frames_lpc_features)
        image_temp2 = np.expand_dims(image_temp1, axis=0)  # 升维
        input_data_list.append(image_temp2)

    if not input_data_list:
        return None

    inputData_array = np.concatenate(input_data_list, axis=0)
    inputData_array = inputData_array.transpose((0, 2, 1))

    # 扩展为4维:(,32,64,1)
    inputData_array = np.expand_dims(inputData_array, axis=3)

    return inputData_array


def zero_crossing_feat(_wav, win_length, hop_length):
    """
    过零率 帧变负数负数变正数的时候要通过0这条线,
    :param _wav: [-1, 1536]
    :param win_length: 256
    :param hop_length: 128(这个应该也同时做为位移)
    :return:
    """
    padding = [(0, 0) for _ in range(_wav.ndim)]  # 不需要padding的维度
    padding[-1] = (hop_length, hop_length)  # 只有最后一个维度才需要padding
    y = np.pad(_wav, padding, mode="constant")

    # sum --> / win_lenght  就是求个平均
    zc = np.sum(
        librosa.zero_crossings(
            np.transpose(
                         # 分帧函数: 将时间序列分割成重叠的帧(所以应该是256帧叠在一起？, 13是特征的维度)
                         librosa.util.frame(y, frame_length=win_length, hop_length=hop_length),  # shape=(-1, 256, 13)
                         [0, 2, 1]),
            pad=False
        ),
        axis=-1
    ) / win_length

    return zc


def fbank(_wav, sample_rate, win_length, hop_length, n_mels, window="hann"):
    """
    sample_rate: 16000
    win_lenght: 256
    hop_length: 128
    n_mels: 32
    window: hann
    这里连续调用了两个音频处理库的包librosa(制作训练的数据集的时候会使用, inference的时候也会使用)
    """
    # 如果提供了时间序列输入y，sr，则首先计算其幅值频谱S，然后通过mel_f.dot（S ** power）将其映射到mel scale上 。
    # 默认情况下，power= 2在功率谱上运行。
    # 这个东西就类似于LPC.dll的功能
    mel_spec_feat = librosa.feature.melspectrogram(
        y=_wav,
        sr=sample_rate,  # 160000
        win_length=win_length,
        hop_length=hop_length,
        n_fft=win_length,
        window=window,
        n_mels=n_mels,
    )

    # 再转换到对数刻度
    db_feat = librosa.core.power_to_db(
        mel_spec_feat, ref=1.0, amin=1e-10, top_db=None,
    )
    feat = (db_feat + 100) / 130.
    return feat


def mfcc(_wav, sample_rate, win_length, hop_length, n_mels, n_mfcc, window="hann"):
    feat = librosa.feature.mfcc(
        y=_wav,
        sr=sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=win_length,
        window=window,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
    )
    return feat
