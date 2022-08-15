# -*-coding:utf-8 -*-

"""
File    : lpc_2_weight.py
Time    : 2022/5/11 16:15
Author  : luzeng
"""

import os
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("../train")

import threading
from queue import Queue
import numpy as np
import tensorflow as tf
from src.demo.wav_2_lpc import c_lpc
from src.train.model import WavBlendShapeModel
from src.train.features.features import zero_crossing_feat, fbank

class WeightsAnimation:
    """
    加载已训练的模型，进行inference
    """

    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model = WavBlendShapeModel(
            16, 1., speakers_class_num=118, conv_encoder_type="zc_micro", build_adverse_model=False)
        # input shape: (-1, 32, 13, 1)
        # 这里初始先输入了一些数据
        self.model(tf.ones(shape=(16, 32, 13, 1)), input_zc=tf.ones(shape=(16, 13)), training=False)
        self.model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)

    def get_weight(self, data, zc_feat=None):
        # print(data.shape, zc_feat.shape)
        weight = self.model(data, input_zc=zc_feat, training=False, adverse_training=False)
        return weight


class SoundAnimation:
    def __init__(self):
        self.init_multiprocessing()
        self.flag_start = False
        self.flag_nums = 0

    def __del__(self):
        if self.flag_start:
            self.stop_multiprocessing()

    def init_multiprocessing(self):
        """
        使用线程安全的队列
        """
        self.q_input = Queue()  # 输入队列
        self.q_output = Queue()  # 输出队列
        self.process = threading.Thread(
            target=inference_multithread_worker,
            args=(self.q_input, self.q_output, True)
        )

    def start_multiprocessing(self):
        """
        开启线程
        """
        self.flag_start = True
        self.process.setDaemon(True)
        self.process.start()

    def stop_multiprocessing(self):
        """
        终止线程
        """
        self.process.terminate()

    def input_frames_data(self, input_data):
        """
        将待测数据放入到输入队列中去, 等待生成bs系数
        """
        self.q_input.put([input_data])

    def yield_output_data(self):
        flag_end = True
        while flag_end:
            data_output = self.q_output.get()
            # 这其实是一个异步的生成器, 这里挂起self.q_output.get()并等待数据返回
            # 但是在等待数据返回的时候还可以做其他的事情, 防止阻塞
            yield data_output


def get_model_output(_wav, without_lpc=False):
    if without_lpc:
        """
        音频的处理
        2. 处理_wav得到音频的特征
        1. 通过_wav计算得到过零率
        """
        feat = fbank(_wav, sample_rate=16000, win_length=256, hop_length=128, n_mels=32, window="hann")
        zc_feat = zero_crossing_feat(_wav, 256, 128).astype(np.float32)
        feat = np.expand_dims(feat, axis=-1)
    else:
        feat = c_lpc(_wav)
        zc_feat = None
    # 将处理好的数据输入到模型之中
    _data = get_weight(data=feat, zc_feat=zc_feat)
    return _data


def inference_multithread_worker(q_input, q_output, without_lpc=False):
    while True:
        # 从输出队列里面取出数据
        input_data = q_input.get()
        # start = time.time()
        for _wav in input_data:
            """
            对数据进行类似于wav文件的处理
            但是我的应用场景中, 倾向于使用wav输入的方式, 以标点符号为间隔输入一段话
            """
            # 512 + 512 + 512 = 1536, 这里是用来三帧, 只是为了满足输入的格式, 主要的是中间的帧
            output = get_model_output(np.array([_wav]), without_lpc=without_lpc)
            # 将预测的结果输入输出队列中去
            q_output.put(output)

def inference(input_data, without_lpc=False):
    q_output = []
    for _wav in input_data:
        output = get_model_output(_wav, without_lpc)
        q_output.append(output)
    return q_output


def inference_batch(input_data, without_lpc=False, batch_size=1024):
    num_batches = (input_data.shape[0] - 1) // batch_size + 1
    q_output = []
    for i in range(num_batches):
        output = get_model_output(input_data[i * batch_size: (i + 1) * batch_size], without_lpc)
        q_output.append(output)
    return q_output

# 使用cpu就行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ckpt_path = "checkpoints/dataset44_var_epoch150_20220714231019/model_epoch149.h5"
pb_weights_animation = WeightsAnimation(ckpt_path)
get_weight = pb_weights_animation.get_weight
