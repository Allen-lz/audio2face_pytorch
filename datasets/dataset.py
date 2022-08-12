import os

from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import sys
sys.path.append(".")
sys.path.append("..")
import random
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from PIL import Image
import math

class AudioDataset(Dataset):
    def __init__(self, target_root, data_root):
        """
        :param window: 音频序列的长度为3
        """
        self.target_root = target_root
        self.data_root = data_root

        self.all_data = []
        self.all_gt = []

        self.pre_process()


    def vector_transforms(self, data):
        # option(1) 这个是全局的mean和std
        # data_mean = np.mean(data)
        # data_std = np.std(data)

        # option(2) 这个是针对每个特征的mean和std
        num_length = data.shape[-1]
        data_mean = np.mean(data.reshape(-1, num_length), axis=0, keepdims=True)[np.newaxis, ...]
        data_std = np.std(data.reshape(-1, num_length), axis=0, keepdims=True)[np.newaxis, ...]

        # 数据标准化
        data = (data - data_mean) / data_std

        return data

    def pre_process(self):
        """
        对数据进行预处理，收集数据
        :return:
        """
        data_list = os.listdir(self.data_root)
        # target_list = os.listdir(self.target_root)

        # for index, item in enumerate(data_list):
        #     assert item == target_list[index]

        for index, data_name in enumerate(data_list):
            data_path = os.path.join(self.data_root, data_name)
            target_path = os.path.join(self.target_root, data_name)
            data = np.load(data_path)
            gt = np.load(target_path)

            # 无口型的片段全部去除，可能是没有人脸，或者噪声数据
            # 静音片段的gt也置零
            gt_sum = gt.sum(axis=1)
            zero_index = np.where(gt_sum == 0)[0]
            # 按概率将0标签的输入也置零
            # option(2)
            if len(zero_index) > 0:
                data[zero_index] = 0

            # # option(1)
            # select_data = []
            # select_gt = []
            # for i in range(data.shape[0]):
            #     if i not in zero_index:
            #         select_data.append(data[i][np.newaxis, ...])
            #         select_gt.append(gt[i][np.newaxis, ...])
            # data = np.concatenate(select_data, axis=0)
            # gt = np.concatenate(select_gt, axis=0)


            data = self.vector_transforms(data)

            padding_data = np.zeros(data[0].shape)[np.newaxis, ...]
            padding_gt = np.zeros(gt[0].shape)[np.newaxis, ...]
            self.all_data.append(data)
            self.all_data.append(padding_data)
            self.all_gt.append(gt)
            self.all_gt.append(padding_gt)

        # 第一个vector是过零率
        self.all_data = np.concatenate(self.all_data, axis=0)[:, np.newaxis, :, :]
        self.all_gt = np.concatenate(self.all_gt, axis=0)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return torch.FloatTensor(np.array(self.all_data[index], dtype=np.float32)), torch.FloatTensor(np.array(self.all_gt[index], dtype=np.float32))

if __name__ == "__main__":
    target_root = "E:/datasets/audio2face/train_gt"
    data_root = "E:/datasets/audio2face/train_data"
    trainsets = AudioDataset(target_root, data_root)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=8, shuffle=True, num_workers=0)

    for batch_idx, (datas, targets) in enumerate(trainloader):
        pass




