# -*-coding:utf-8 -*-

import os
import sys
import math
import librosa
import numpy as np
import pandas as pd
sys.path.append(".")
sys.path.append("..")
import features.util as util
from features import features


def wav_signal_to_feat(signal,
                       ind_path,
                       num_frames,
                       fps,
                       sample_rate,
                       win_length,
                       hop_length,
                       half_chunks_length,
                       feat_func,
                       ):
    """
    音频转成输入特征，特征转换函数作为参数传入

    音频切分成每个样本 -> 每个样本转特征 -> concat -> 返回结果
                    ↘ 每个样本的过零率 ↗

    :param signal: 音频时域信号，直接从wav中读取出来的
    :param ind_path: 图像帧的时域位置索引文件
    :param num_frames: 图像帧总数量
    :param fps:
    :param sample_rate: 音频采样率
    :param win_length: 音频分帧窗口大小
    :param hop_length: 音频分帧的帧移
    :param half_chunks_length: 单个训练样本的时长的一半，单位：ms
    :param feat_func: 特征提取函数，主要是lambda表达式

    :return: numpy.Array
    """

    ### =================== 这个和我的数据处理的唯一区别是这里的rate是设置死的 160000 ====================

    signal_length = len(signal)
    chunks_signal_samples = half_chunks_length * sample_rate // 1000
    # 前后各添加 空白 音频
    a = np.zeros(chunks_signal_samples, dtype=np.int16)
    signal = np.hstack((a, signal, a))

    if ind_path is not None:
        img_frame_ind = np.load(ind_path)
        audio_frames = [signal[ind: ind + chunks_signal_samples * 2] for ind in img_frame_ind]
    else:
        frames_step = 1000.0 / fps  # 视频每帧的时长间隔
        rate_kHz = sample_rate // 1000  # 1ms的采样数

        # 帧数不一致，无法对齐，丢弃数据，返回None
        if math.fabs(int(signal_length / (frames_step * rate_kHz)) - num_frames) > 2:
            print("calculate num frames: {}, actual num frames{}".format(
                signal_length / (frames_step * rate_kHz), num_frames))
            print("different frames count, skip data.")
            return None

        # 按图像帧的位置，切分每个图像对应的输入音频样本
        audio_frames = [
            signal[round(i * frames_step * rate_kHz): round((i * frames_step * rate_kHz) + 2 * chunks_signal_samples)]
            if round((i * frames_step * rate_kHz) + 2 * chunks_signal_samples) < len(signal)
            else signal[-int(2 * chunks_signal_samples):]
            for i in range(num_frames)
        ]

    audio_frames = np.array(audio_frames, dtype=np.float32)

    # 音频特征
    feat = feat_func(audio_frames)

    # 过零率特征
    zc_feat = features.zero_crossing_feat(audio_frames, win_length, hop_length)

    # 这里是包括过零率的
    # 这里过零率和特征的长度是一样的, cat在feat的前面
    feat = np.concatenate([zc_feat[:, np.newaxis, :], feat], axis=1)
    return feat


def preprocess(wav_path,
               ind_path=None,
               num_frames: int = None,
               fps: float = 30,
               sample_rate=16000,
               is_add_noise=True,
               add_thick_noise=True,
               add_env_noise=True,
               env_noise=None,
               win_length=256,
               hop_length=128,
               half_chunks_length=48,
               feat_func=features.fbank,
               ):
    """
    预处理

    :param wav_path: 音频文件路径
    :param ind_path: 图像帧的时域位置索引文件
    :param num_frames:
    :param fps:
    :param sample_rate:
    :param is_add_noise: 是否添加轻量级噪声
    :param add_thick_noise: 是否添加重量级噪声
    :param add_env_noise: 是否添加环境音噪声
    :param env_noise: 环境音噪声，时域信号
    :param win_length:
    :param hop_length:
    :param half_chunks_length:
    :param feat_func:
    :return:
    """

    feat, feat_wgn_light, feat_wgn_thick, feat_env_noise = None, None, None, None

    # 读取文件
    signal, rate = librosa.load(wav_path, sr=sample_rate)
    signal_length = len(signal)
    print("length: ", signal_length, "rate: ", rate)

    # 空文件，返回None
    if signal_length == 0:
        return None, None, None, None

    # 特征提取
    feat = wav_signal_to_feat(
        signal, ind_path, num_frames=num_frames, fps=fps, sample_rate=sample_rate,
        win_length=win_length, hop_length=hop_length, half_chunks_length=half_chunks_length, feat_func=feat_func,
    )

    # 帧数不一致，返回None
    if feat is None:
        return None, None, None, None

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<下面这些都不用看了先, 本任务不会有噪声<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 添加高斯白噪声，信噪比分别为12和6
    if is_add_noise:
        signal_wgn_light = util.add_noise(signal, 12.)
        feat_wgn_light = wav_signal_to_feat(
            signal_wgn_light, ind_path, num_frames=num_frames, fps=fps, sample_rate=sample_rate,
            win_length=win_length, hop_length=hop_length, half_chunks_length=half_chunks_length, feat_func=feat_func,
        )
    if add_thick_noise:
        signal_wgn_thick = util.add_noise(signal, 6.)
        feat_wgn_thick = wav_signal_to_feat(
            signal_wgn_thick, ind_path, num_frames=num_frames, fps=fps, sample_rate=sample_rate,
            win_length=win_length, hop_length=hop_length, half_chunks_length=half_chunks_length, feat_func=feat_func,
        )
    # 添加真实环境音噪声
    if add_env_noise:
        signal_env_noise = util.add_other_noise(signal, noise=env_noise)
        feat_env_noise = wav_signal_to_feat(
            signal_env_noise, ind_path, num_frames=num_frames, fps=fps, sample_rate=sample_rate,
            win_length=win_length, hop_length=hop_length, half_chunks_length=half_chunks_length, feat_func=feat_func,
        )

    return feat, feat_wgn_light, feat_wgn_thick, feat_env_noise


def selfmade_arkit_preprocess(wav_dir, profile_dir, gt_dir, save_dir, env_noise):
    # 文件映射字典
    data_files = {os.path.splitext(file)[0]: os.path.join(wav_dir, file)
                  for file in os.listdir(wav_dir) if file.split(".")[-1] == "wav"}
    profiles_dict = {os.path.splitext(file)[0]: os.path.join(profile_dir, file) for file in os.listdir(profile_dir)}
    gt_dict = {os.path.splitext(file)[0]: os.path.join(gt_dir, file) for file in os.listdir(gt_dir)}

    for file_id, path in data_files.items():
        if os.path.exists(os.path.join(save_dir, file_id + ".npy")):
            print("Exist file {}".format(os.path.join(save_dir, file_id + ".npy")))
            continue

        if file_id not in profiles_dict or file_id not in gt_dict:
            continue

        profile = util.load_json_file(profiles_dict[file_id])
        gt_label = np.load(gt_dict[file_id])

        num_frames = gt_label.shape[0]
        # num_frames = int(profile["num_frames"])
        fps = profile["fps"]

        print("Processing {:<15s}, path: {}...".format(file_id, path))
        lpc_feat, lpc_feat_wgn_s, _, feat_env_noise = preprocess(
            path, ind_path=None, num_frames=num_frames, fps=fps,
            sample_rate=SAMPLE_RATE, add_thick_noise=False, env_noise=env_noise,
            win_length=WIN_LENGTH, hop_length=HOP_LENGTH, half_chunks_length=HALF_CHUNKS_LENGTH, feat_func=FEAT_FUNC
        )
        if lpc_feat is not None:
            np.save(save_dir + "/" + file_id + ".npy", lpc_feat)
            np.save(save_dir + "/" + file_id + ".wgn_s" + ".npy", lpc_feat_wgn_s)
            np.save(save_dir + "/" + file_id + ".env_n" + ".npy", feat_env_noise)


def facegood_preprocess(wav_dir, label_dir, save_dir, env_noise):
    for file in os.listdir(wav_dir):
        file_id = file.split(".")[0]
        abs_path = os.path.join(wav_dir, file)
        print("Processing {:<15s}, path: {}...".format(file, abs_path))

        # label数量 == 视频总帧数
        label_file = os.path.join(label_dir, "bs_value_{}.npy".format(os.path.splitext(file)[0]))
        num_frames = np.load(label_file).shape[0]

        feat, feat_wgn_small, feat_wgn_large, feat_env_noise = preprocess(
            abs_path, num_frames=num_frames, fps=30, sample_rate=SAMPLE_RATE, add_thick_noise=False,
            env_noise=env_noise,
            win_length=WIN_LENGTH, hop_length=HOP_LENGTH, half_chunks_length=HALF_CHUNKS_LENGTH, feat_func=FEAT_FUNC
        )
        if feat is not None:
            np.save(os.path.join(save_dir, file_id + ".npy"), feat)
            np.save(os.path.join(save_dir, file_id + ".wgn_s" + ".npy"), feat_wgn_small)
            # np.save(os.path.join(save_dir, file_id + ".wgn_l" + ".npy"), feat_wgn_large)
            np.save(os.path.join(save_dir, file_id + ".env_n" + ".npy"), feat_env_noise)


def aiwin_preprocess(aiwin_dir, save_dir, env_noise, is_eval=False):
    # 文件映射字典
    data_files = {file.split(".")[0].lower(): os.path.join(aiwin_dir, file)
                  for file in os.listdir(aiwin_dir) if file.split(".")[-1] == "wav"}
    gt_files = {file.split(".")[0].lower().replace("_anim", ""): os.path.join(aiwin_dir, file)
                for file in os.listdir(aiwin_dir) if file.split(".")[-1] == "csv"}

    for file_id, path in data_files.items():
        print("Processing {:<15s}, path: {}...".format(file_id, path))

        if file_id not in gt_files:
            continue

        # label数量 == 视频总帧数
        label_file = gt_files[file_id]
        num_frames = pd.read_csv(label_file).shape[0]

        feat, feat_wgn_small, feat_wgn_large, feat_env_noise = preprocess(
            path, num_frames=num_frames, fps=25, sample_rate=SAMPLE_RATE, add_thick_noise=False, env_noise=env_noise,
            win_length=WIN_LENGTH, hop_length=HOP_LENGTH, half_chunks_length=HALF_CHUNKS_LENGTH, feat_func=FEAT_FUNC
        )

        if feat is not None:
            if not is_eval:
                np.save(os.path.join(save_dir, "aiwin_" + file_id + ".npy"), feat)
                np.save(os.path.join(save_dir, "aiwin_" + file_id + ".wgn_s" + ".npy"), feat_wgn_small)
                # np.save(os.path.join(save_dir, "aiwin_" + file_id + ".wgn_l" + ".npy"), feat_wgn_large)
                np.save(os.path.join(save_dir, "aiwin_" + file_id + ".env_n" + ".npy"), feat_env_noise)
            else:
                np.save(os.path.join(save_dir, "aiwin_eval_" + file_id + ".npy"), feat)
                np.save(os.path.join(save_dir, "aiwin_eval_" + file_id + ".wgn_s" + ".npy"), feat_wgn_small)
                # np.save(os.path.join(save_dir, "aiwin_eval_" + file_id + ".wgn_l" + ".npy"), feat_wgn_large)
                np.save(os.path.join(save_dir, "aiwin_eval_" + file_id + ".env_n" + ".npy"), feat_env_noise)


def public_dataset_preprocess(wav_dir, profile_dir, save_dir, env_noise):
    profiles_dict = {os.path.splitext(file)[0]: os.path.join(profile_dir, file) for file in os.listdir(profile_dir)}

    # 文件映射字典
    data_files = {file.split(".")[0]: os.path.join(wav_dir, file)
                  for file in os.listdir(wav_dir) if file.split(".")[-1] == "wav"}

    for file_id, path in data_files.items():
        if os.path.exists(os.path.join(save_dir, file_id + ".npy")):
            print("Exist file {}".format(os.path.join(save_dir, file_id + ".npy")))
            continue

        if file_id in profiles_dict:
            profile = util.load_json_file(profiles_dict[file_id])
        else:
            continue

        print("Processing {:<15s}, path: {}...".format(file_id, path))
        feat, feat_wgn_small, _, feat_env_noise = preprocess(
            path, ind_path=None, num_frames=int(profile["num_frames"]), fps=profile["fps"],
            sample_rate=SAMPLE_RATE, add_thick_noise=False, env_noise=env_noise,
            win_length=WIN_LENGTH, hop_length=HOP_LENGTH, half_chunks_length=HALF_CHUNKS_LENGTH, feat_func=FEAT_FUNC
        )

        if feat is not None:
            np.save(save_dir + "/" + file_id + ".npy", feat)
            np.save(save_dir + "/" + file_id + ".wgn_s" + ".npy", feat_wgn_small)
            np.save(save_dir + "/" + file_id + ".env_n" + ".npy", feat_env_noise)


if __name__ == '__main__':
    # 参数配置
    use_self_made_arkit = False
    use_self_made_mocap = False

    use_facegood = False
    use_aiwin = False
    use_public = True

    SAMPLE_RATE = 16000
    WIN_LENGTH = 256
    HOP_LENGTH = 128
    N_FEAT = 32
    HALF_CHUNKS_LENGTH = 48
    FEAT_FUNC = lambda x: features.fbank(
        x, sample_rate=SAMPLE_RATE, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_FEAT)

    # 数据路径
    save_dir = "E:/datasets/audio2face/processed_datasets"
    # 这里还加上了噪声数据来模仿环境噪声
    env_noise_file = "E:/3D_face_reconstruct/audio2face/data/train/environmental_noise_2.wav"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 对之前从视频中分离出来的音频进解析
    # sample rate要用之前生成wav文件的时候使用的sr
    env_noise, _ = librosa.load(env_noise_file, sr=16000)

    # 自制数据，ARKIT录制
    if use_self_made_arkit:
        # clean_data_dir = "E:/数据集/人脸视频口型数据/clean"
        # selfmade_dataset_preprocess(clean_data_dir, clean_data_dir, save_dir)

        clean_data_dir = "E:/数据集/人脸视频口型数据/raw_3"
        profile_dir = "E:/数据集/人脸视频口型数据/profile_3"
        gt_dir = "E:/数据集/chinese_video_process/clean_gt_arkit"
        selfmade_arkit_preprocess(clean_data_dir, profile_dir, gt_dir, save_dir, env_noise=env_noise)

    # 自制数据，MocapFace识别
    if use_self_made_mocap:
        pass

    # FACEGOOD样例数据
    if use_facegood:
        facegood_wav_dir_path = "D:/projects/AI/research/pose/audio2face/data/train/raw_wav"
        facegood_label_dir_path = "D:/projects/AI/research/pose/audio2face/data/train/bs_value"
        facegood_preprocess(facegood_wav_dir_path, facegood_label_dir_path, save_dir, env_noise=env_noise)

    # AIWIN训练数据
    elif use_aiwin:
        aiwin_dir_path = "E:/数据集/chinese_video_process/audio2face_data_for_train"
        aiwin_eval_dir_path = "E:/数据集/chinese_video_process/audio2face_data_for_evaluation"
        # aiwin_preprocess(aiwin_dir_path, save_dir, env_noise=env_noise, is_eval=False)
        aiwin_preprocess(aiwin_eval_dir_path, save_dir, env_noise=env_noise, is_eval=True)

    # 公开数据集
    elif use_public:
        public_dataset_clean_wav_dir = "E:/datasets/audio2face/wav_base"
        public_dataset_profile_path = "E:/datasets/audio2face/profile_base"
        public_dataset_preprocess(public_dataset_clean_wav_dir, public_dataset_profile_path, save_dir,
                                  env_noise=env_noise)
