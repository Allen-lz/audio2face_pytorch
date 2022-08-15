# -*-coding:utf-8 -*-

"""
File    : run_demo.py
Time    : 2022/8/11 16:17
Author  : luzeng
"""
import os
import copy
import json
import time
import socket
import warnings
import datetime
import threading
import wave
import pyaudio
import cv2
import librosa
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
sys.path.append(".")
from src.demo.recorder import Recorder
from src.demo.wav_2_lpc import get_audio_frames
from src.demo.animation_convertor import AnimationConvertor
from src.demo.lpc_2_weight import inference_batch, SoundAnimation

warnings.filterwarnings("ignore")


def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('172.19.208.1', 12345))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    return s


valid_arkit_bs_name = ["JawForward", "JawLeft", "JawRight", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker",
                       "MouthLeft", "MouthRight", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft",
                       "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
                       "MouthStretchRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower",
                       "MouthShrugUpper", "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft",
                       "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight"]

# rename, 将第一个字母从大写改成小写
lower_camel_case_valid_arkit_bs_name = [name[0].lower() + name[1:] for name in valid_arkit_bs_name]

json_bs_name_order = [
    "None",
    "browInnerUP",
    "browOutterUpLeft",
    "browOutterUpRight",
    "browDownLeft",
    "browDownRight",
    "eyeWideLeft",
    "eyeWideRight",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPucker",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthFunnel",
    "mouthPress",
    "jawOpen",
    "mouthRollLower",
    "mouthRollUpper",
    "jawForward",
    "jawLeft",
    "jawRight",
    "cheekPuff",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "mouthPressLeft",
    "mouthPressRight",
    "headUp",
    "headDown",
    "headLeft",
    "headRight",
    "headRollLeft",
    "headRollRight",
    "212",
    "213",
    "214",
    "215",
    "216",
]

valid_arkit_bs_name_2 = [
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
]

valid_arkit_bs_name_3 = [
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

valid_arkit_bs_name_4 = [
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

express_list_template = {
    "ExpressList": [
        {"k": 1, "v": 0.0}, {"k": 2, "v": 0.0}, {"k": 3, "v": 0.0}, {"k": 4, "v": 0.0}, {"k": 5, "v": 0.0},
        {"k": 6, "v": 0.0}, {"k": 7, "v": 0.0}, {"k": 8, "v": 0.0}, {"k": 9, "v": 0.0}, {"k": 10, "v": 0.0},
        {"k": 11, "v": 0.0}, {"k": 12, "v": 0.0}, {"k": 13, "v": 0.0}, {"k": 14, "v": 0.0}, {"k": 15, "v": 0.0},
        {"k": 16, "v": 0.0}, {"k": 17, "v": 0.0}, {"k": 18, "v": 0.0}, {"k": 19, "v": 0.0}, {"k": 20, "v": 0.0},
        {"k": 21, "v": 0.0}, {"k": 22, "v": 0.0}, {"k": 23, "v": 0.0}, {"k": 24, "v": 0.0}, {"k": 25, "v": 0.0},
        {"k": 26, "v": 0.0}, {"k": 27, "v": 0.0}, {"k": 28, "v": 0.0}, {"k": 29, "v": 0.0}, {"k": 30, "v": 0.0},
        {"k": 31, "v": 0.0}, {"k": 32, "v": 0.0}, {"k": 33, "v": 0.0}, {"k": 34, "v": 0.0}, {"k": 35, "v": 0.0},
        {"k": 36, "v": 0.0}, {"k": 37, "v": 0.0}, {"k": 38, "v": 0.0}, {"k": 39, "v": 0.0}, {"k": 40, "v": 0.0},
        {"k": 41, "v": 0.0}, {"k": 42, "v": 0.0}, {"k": 43, "v": 0.0}, {"k": 44, "v": 0.0}, {"k": 45, "v": 0.0},
        {"k": 46, "v": 0.0}, {"k": 47, "v": 0.0}, {"k": 48, "v": 0.0}, {"k": 49, "v": 0.0}, {"k": 50, "v": 0.0},
        {"k": 51, "v": 0.0}, {"k": 52, "v": 0.0}, {"k": 53, "v": 0.0}, {"k": 54, "v": 0.0}, {"k": 55, "v": 0.0},
        {"k": 56, "v": 0.0}, {"k": 57, "v": 0.0}, {"k": 58, "v": 0.0}, {"k": 212, "v": 0}, {"k": 213, "v": 0},
        {"k": 214, "v": 0}, {"k": 215, "v": 0}, {"k": 216, "v": 0},
    ]
}

body1 = """{"frame":81,"timestamp":1653020274303}#"""
body2 = """{"cmdList":[{"k":0,"v":{"x":-0.16915,"y":0.44524,"z":-0.14412},"visibility":0.99242},{"k":1,"v":{"x":-0.23624,"y":0.28103,"z":-0.13774},"visibility":0.80215},{"k":2,"v":{"x":-0.25922,"y":0.1164,"z":-0.16851},"visibility":0.3386},{"k":3,"v":{"x":-0.24703,"y":0.10263,"z":-0.18967},"visibility":0.35001},{"k":4,"v":{"x":-0.25588,"y":0.06091,"z":-0.18511},"visibility":0.27888},{"k":5,"v":{"x":0.11057,"y":0.48848,"z":-0.03623},"visibility":0.99176},{"k":6,"v":{"x":0.3029,"y":0.40632,"z":-0.03385},"visibility":0.66099},{"k":7,"v":{"x":0.43823,"y":0.3376,"z":-0.17424},"visibility":0.56821},{"k":8,"v":{"x":0.42734,"y":0.3283,"z":-0.19942},"visibility":0.57001},{"k":9,"v":{"x":0.47437,"y":0.31352,"z":-0.20123},"visibility":0.47294},{"k":10,"v":{"x":0.04314,"y":0.64238,"z":-0.10405},"visibility":0.99465},{"k":11,"v":{"x":0.02436,"y":0.66783,"z":-0.2133},"visibility":0.99679},{"k":12,"v":{"x":-0.08512,"y":0.63474,"z":-0.14779},"visibility":0.9982},{"k":13,"v":{"x":-0.00396,"y":0.66814,"z":-0.22631},"visibility":0.99743},{"k":14,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":15,"v":{"x":-0.10621,"y":0.00281,"z":0.00741},"visibility":1.0},{"k":16,"v":{"x":-0.09716,"y":-0.37997,"z":0.00613},"visibility":1.0},{"k":17,"v":{"x":-0.08832,"y":-0.73587,"z":0.19865},"visibility":1.0},{"k":18,"v":{"x":-0.13203,"y":-0.85234,"z":0.09052},"visibility":1.0},{"k":19,"v":{"x":0.10582,"y":-0.00233,"z":-0.00702},"visibility":1.0},{"k":20,"v":{"x":0.12449,"y":-0.38901,"z":0.0043},"visibility":1.0},{"k":21,"v":{"x":0.15222,"y":-0.72213,"z":0.18485},"visibility":1.0},{"k":22,"v":{"x":0.18626,"y":-0.83152,"z":0.06556},"visibility":1.0},{"k":23,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":24,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":25,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":26,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":27,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781}],"status":0,"valid":1}#"""

label_name_list = pd.read_csv("doc/bsname.txt", encoding="utf-8").values.transpose()[0].tolist()

def convert_to_json_str_3(output):
    indices = [json_bs_name_order.index(name) if name in json_bs_name_order else -1
               for name in valid_arkit_bs_name_3]
    json_list = []
    for i, frame in enumerate(output):
        express_list = copy.deepcopy(express_list_template)
        for ind, val in zip(indices, frame):
            if ind == -1:
                continue
            express_list["ExpressList"][ind - 1]["v"] = min(max(val * 100, 0.), 100.)
        json_str = body1 + body2 + json.dumps(express_list)
        json_list.append(json_str)
    return json_list


def play_wav(wav_path, chunk=1024):
    wf = wave.open(wav_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()

dataset_id = "dataset44"  # 这个dataset_id也只是为了获得mean, std
data_dir = os.path.join("data/train", dataset_id)
label_mean = np.load(os.path.join(data_dir, "label_aug_mean.npy"))
label_std = np.load(os.path.join(data_dir, "label_aug_std.npy"))
demo_config = "src/demo/config/predict_demo_44.json"  # 存放着wav音频文件
json_bs_path = "data/test_demo"
head_animations_file = "src/demo/actions.txt"
add_head_pose = False

def generate_bs_by_wav(video_fps=30, chunks_length=48):
    """
    通过音频文件wav得到bs系数
    """

    # 建立一个输出格式转换器ARKit --> bs
    convertor = AnimationConvertor(head_animations_file, add_head_pose=add_head_pose)

    with open(demo_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    wav_path_list = config["wav_path"]

    for _, wav_path in wav_path_list:
        voice, sr = librosa.load(wav_path, sr=16000)
        # inference
        # 1. 直接从数据集制作的那拆下来的 wav处理模块, 和train时候的数据处理一样, 将其转化成网络接收的输入
        audio_frames = get_audio_frames(voice, frames_per_second=video_fps, chunks_length=chunks_length)
        output = inference_batch(np.array(audio_frames), without_lpc=True)
        output = np.concatenate(output, axis=0)
        output = np.maximum(np.minimum((output - 3.) * label_std + label_mean, 1.), 0.)
        # output[:, 2] = output[:, 2] * 1.2
        # 平滑
        output = conv_smooth(output, mode="same")

        # 将output的格式进行一个转换
        json_list = convertor.convert_to_json_str_3(output, valid_arkit_bs_name_3)

        file_name = os.path.splitext(os.path.split(wav_path)[-1])[0]
        with open(os.path.join(json_bs_path, file_name), "w", encoding="utf-8") as f:
            _ = [print(json_str, file=f) for json_str in json_list]


def sent_bs_json_data():
    fps = 30
    speed_play = float(1.0 / fps)
    _client = socket_client()

    with open(demo_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    wav_path_list = config["wav_path"]

    for i, (need_play, wav_path) in enumerate(wav_path_list):
        if not need_play:
            continue

        print("start sending demo {}".format(wav_path))

        file_name = os.path.splitext(os.path.split(wav_path)[-1])[0]
        json_bs_file = os.path.join(json_bs_path, file_name)

        with open(json_bs_file, "r", encoding="utf-8") as f:
            f_num = 0
            threading._start_new_thread(play_wav, (wav_path,))
            time.sleep(0.15)
            f_btime = time.time()
            for line in f.readlines():
                f_num += 1
                _client.send(line.strip().encode())
                sleep_time = max(speed_play * f_num - (time.time() - f_btime), 0.)
                if sleep_time <= 0:
                    print("blendshape delay, {}".format(f_num))
                time.sleep(sleep_time)
        time.sleep(1)


def conv_smooth(data, mode="same"):
    win_size = 5
    if mode == "same":
        new_data = np.zeros_like(data)
    else:
        new_data = np.zeros((data.shape[0] - win_size + 1, data.shape[1]), dtype=data.dtype)
    conv_win = np.hamming(win_size) / np.sum(np.hamming(win_size))
    for i in range(new_data.shape[1]):
        new_data[:, i] = np.convolve(data[:, i], conv_win, mode=mode)
    return new_data


def video_shoot():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("打开视频失败！")
    cv2.moveWindow("camera", 1200, 300)
    while True:
        _, frame = capture.read()
        if frame is None:
            break
        cv2.imshow("camera", cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()


def consume_blend_shape_output(sound_animation, label_std, label_mean):
    """
    这个是将output_d中的bs系数进行格式转换并通过socket发送出去的函数
    """

    # 与客户端建立连接
    _client = socket_client()
    # 为什么batch是5呢, 因为可能hanning窗口的长度就是5
    result_mat = np.zeros((5, 16), dtype=np.float32)
    # 这个转换器其实不需要具体知道里面是干嘛的, 知道它与格式相关就行
    convertor = AnimationConvertor(head_animations_file, add_head_pose=add_head_pose)
    while True:
        try:
            for weight in sound_animation.yield_output_data():
                weight = np.maximum(np.minimum((weight.numpy() - 3.) * label_std + label_mean, 1.), 0.)
                result_mat = np.concatenate([result_mat[1:], weight], axis=0)
                # result_mat: (5, 16)

                # 平滑
                _weight = conv_smooth(result_mat, mode="valid")
                # _weight[0, [3, 4]] = 0

                # 转json格式
                json_list = convertor.convert_to_json_str_3(_weight, valid_arkit_bs_name_3)

                # socket驱动
                # 就是将数据发送大unity端
                _client.send(json_list[0].encode())

                print(datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        except Exception as err:
            print("Sound animation type error: ", err)


def run_real_time():
    CHUNK = 512
    model_input_length = 1536  # 这个是audio特征的长度
    CHANNELS = 1
    RATE = 16000

    # blendshape表情预测
    sound_animation = SoundAnimation()
    sound_animation.start_multiprocessing()

    # 摄像头线程
    video_thread = threading.Thread(target=video_shoot, args=())
    video_thread.setDaemon(True)
    video_thread.start()

    # 驱动数字人的线程
    voice_thread = threading.Thread(target=consume_blend_shape_output, args=(sound_animation, label_mean, label_std))
    voice_thread.setDaemon(True)
    voice_thread.start()

    # 录音设备线程
    # 其实录音这里的去噪等我不用管, 毕竟我做的是直接输入文字转音频, 是没有噪声的
    rec = Recorder(sound_animation, input_length=model_input_length, chunk=CHUNK, channels=CHANNELS, rate=RATE)
    rec.record()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_time", default=False, action="store_true")
    parser.add_argument("--only_bs_data", default=True, action="store_true")
    args, _ = parser.parse_known_args()

    if args.real_time:
        run_real_time()
    else:
        generate_bs_by_wav()
        sent_bs_json_data()