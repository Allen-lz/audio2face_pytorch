"""
File    : build_dataset
Time    : 2022/8/2 11:09
Author  : Lu Zeng


这个脚本用来对齐音频和视频(图像)
如果检测到了人脸就将主体人脸crop下来，并使用mocapface进行标注(只要40个bs的系数, 不需要头部转向的参数)
并保存该帧对应的音频
"""
import numpy as np
import cv2
from moviepy.editor import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import ffmpeg
from ctypes import *
import mediapipe as mp
import sys
sys.path.append("..")
sys.path.append(".")
from third_part.moCapFace import MoCapFace

dll = cdll.LoadLibrary(os.path.join('third_part', 'LPC.dll'))


def get_source_info_ffmpeg(source_name):
    return_value = 0
    try:
        info = ffmpeg.probe(source_name)
        format_name = info['format']['format_name']

        video_info = next(c for c in info['streams'] if c['codec_type'] == 'video')
        audio_info = next(c for c in info['streams'] if c['codec_type'] == 'audio')
        codec_name = audio_info['codec_name']
        duration_ts = float(audio_info['duration_ts'])
        fps = audio_info['r_frame_rate']

        print("format_name:{} \ncodec_name:{} \nduration_ts:{} \nfps:{}".format(format_name, codec_name, duration_ts, fps))

        codec_name = video_info['codec_name']
        duration_ts = float(video_info['duration_ts'])
        fps = video_info['r_frame_rate']
        width = video_info['width']
        height = video_info['height']
        num_frames = video_info['nb_frames']
        print("format_name:{} \ncodec_name:{} \nduration_ts:{} \nwidth:{} \nheight:{} \nfps:{} \nnum_frames:{}".format(format_name,
                                                                                                       codec_name,
                                                                                                       duration_ts,
                                                                                                       width, height,
                                                                                                       fps, num_frames))
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("init_source:{} error. {}\n".format(source_name, str(e)))

        return return_value, 0, 0
    return return_value, fps, num_frames


def vis_audio(rate, signal):
    print(signal.shape)
    print(f"number of channels = {signal.shape[1]}")
    length = signal.shape[0] / rate
    print(f"length = {length}s")
    time = np.linspace(0., length, signal.shape[0])
    plt.plot(time, signal[:, 0], label="Left channel")
    plt.plot(time, signal[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


# 画出人脸框和关键点
def draw_face(img, bbox, expand_ratio=0.5):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    corpbbox = [max(0, int(bbox[0] - expand_ratio * w)),
                max(0, int(bbox[1] - expand_ratio * h)),
                min(img.shape[1] - 1, int(bbox[2] + expand_ratio * w)),
                min(img.shape[0] - 1, int(bbox[3] + 0.1 * expand_ratio * h))
                ]
    crop = img[corpbbox[1]: corpbbox[3], corpbbox[0]:corpbbox[2], :]
    return crop


def get_square_image(face, type="center"):
    """
    face不是bbox, 而是基于bbox在img上crop出来的人脸图像
    基于短边, 缩短长边
    """
    face_h, face_w = face.shape[:2]
    if type == "center":
        if face_h > face_w:
            pad = (face_h - face_w) // 2
            if pad != 0:
                face = face[pad:-pad, :, :]
        elif face_h < face_w:
            pad = (face_w - face_h) // 2
            if pad != 0:
                face = face[:, pad:-pad, :]

    elif type == "upper":
        if face_h > face_w:
            pad = (face_h - face_w) // 2
            if pad != 0:
                face = face[:-2 * pad, :, :]
        elif face_h < face_w:
            # 在水平方向的crop方式照常
            pad = (face_w - face_h) // 2
            if pad != 0:
                face = face[:, pad:-pad, :]
    return face


def read_frame_as_jpeg(ffmpeg_video, frame_num):
  """
  ffmpeg_video: 是已经加载完成的视频数据
  指定帧数读取任意帧
  """
  out, err = (
    ffmpeg_video.filter('select', 'gte(n,{})'.format(frame_num)).output('pipe:', vframes=1, format='image2', vcodec='mjpeg').run(capture_stdout=True)
  )
  # 将bytes转成nunpy的格式
  try:
    image_np = bytes_to_numpy(out)
    return image_np
  except:
      return int(-1)


def bytes_to_numpy(image_bytes):
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np

if __name__ == "__main__":
    bs2id = {
        21: 'jawopen',
        23: 'mouthpucker',
        19: 'mouthfunnel',
        12: 'mouthsmileleft',
        29: 'mouthsmileright',
        14: 'mouthfrownleft',
        27: 'mouthfrownright',
        20: 'mouthrolllower',
        24: 'mouthrollupper',
        11: 'mouthupperupleft',
        30: 'mouthupperupright',
    }
    need_ids = [21, 23, 19, 12, 29, 14, 27, 20, 24, 11, 30]



    # video_path = "assets/baijiajiangtan.mp4"

    video_path = r"E:/datasets/audio2face/cctv_short_video_bilibili/Av629249051-P1.mp4"

    flag_name = os.path.basename(video_path)

    # image_save_root = "crop_face_images"
    # bs_targets_root = "bs_targets"

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<进行视频的处理<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    mocapfacenet = MoCapFace()

    flag, fps, num_frames = get_source_info_ffmpeg(video_path)
    fps = int(fps.split("/")[0])  # 视频的fps
    num_frames = int(num_frames)  # 视频的总帧数

    frames = []
    bs_targets = []
    ffmpeg_video = ffmpeg.input(video_path)
    # bs_target_txt = open(os.path.join(bs_targets_root, os.path.basename(video_path).split(".")[0] + ".txt"), "w")
    for i in range(num_frames):
        frame = read_frame_as_jpeg(ffmpeg_video, i)

        if isinstance(frame, int):
            # 当前视频帧损坏的情况
            # frames.append(frame)
            bs_targets.append(frame)
            # bs_target_txt.write(str(frame) + "\n")
        else:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                # 只有当有检测到东西的时候才进行下面的操作
                h, w = frame.shape[:2]
                area = 0
                x1y1x2y2 = [0, 0, 0, 0]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = bbox.xmin
                    y1 = bbox.ymin
                    width = bbox.width
                    height = bbox.height
                    this_area = width * height
                    if this_area > area:
                        area = this_area
                        x1y1x2y2 = [int(x1 * w), int(y1 * h), int((x1 + width) * w), int((y1 + height) * h)]

                crop_face = draw_face(frame, x1y1x2y2)
                crop_face = get_square_image(crop_face)
                bs_target = mocapfacenet.forword(crop_face)
                bs_targets.append(bs_target)
                bs_target = list(map(str, bs_target))
                # bs_target_txt.write(" ".join(bs_target) + "\n")

            # 没有检测出人脸的情况
            # frames.append(frame)
            bs_targets.append(int(0))
            # bs_target_txt.write(str(0) + "\n")

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<进行音频的处理<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("assets/tmp.wav")
    rate, signal = wavfile.read("assets/tmp.wav")  # rate：采样率
    # rate: 是采样率
    # signal: 是音频信号

    # vis_audio(rate, signal)
    if signal.shape[-1] == 2:
        signal = np.mean(signal, axis=-1)

    frames_per_second = fps  # 视频fps(自己的数据集要设置好视频的fps才能和音频一一对应)
    chunks_length = 48  # 260音频分割，520ms 前260ms 后260ms 这个是可以自己设置的

    # 每signal个采样一个信号, 这个信号会对应着30帧视频
    audio_frameNum = int(len(signal) / rate * frames_per_second)  # 计算音频对应的视频帧数(一般这个就等于视频帧数)

    # 前后各添加260ms音频
    a = np.zeros(chunks_length * rate // 1000, dtype=np.int16)

    signal = np.hstack((a, signal, a))

    # signal = signal / (2.**15)
    frames_step = 1000.0 / frames_per_second  # 视频每帧的时长间隔33.3333ms
    rate_kHz = int(rate / 1000)  # 采样率：48kHz

    # 开始进行音频的分割
    # audio_frames = [signal[int(i * frames_step * rate_kHz): int((i * frames_step + chunks_length * 2) * rate_kHz)] for i
    #                 in range(audio_frameNum)]
    audio_frames = [signal[int(i * frames_step * rate / 1000): int((i * frames_step + chunks_length * 2) * rate / 1000)] for i
                    in range(audio_frameNum)]

    inputData_array = np.zeros(shape=(1, 32, 64))  # 创建一个空3D数组，该数组(1*32*64)最后需要删除

    for i in range(len(audio_frames)):
        audio_frame = audio_frames[i]  # 每段音频，8320个采样点

        overlap_frames_apart = 0.008
        overlap = int(rate * overlap_frames_apart)  # 128 samples
        frameSize = int(rate * overlap_frames_apart * 2)  # 256 samples
        numberOfFrames = 64

        frames = np.ndarray(
            (numberOfFrames, frameSize))  # initiate a 2D array with numberOfFrames rows and frame size columns
        for k in range(0, numberOfFrames):
            for i in range(0, frameSize):
                if ((k * overlap + i) < len(audio_frame)):
                    frames[k][i] = audio_frame[k * overlap + i]
                else:
                    frames[k][i] = 0

        frames *= np.hanning(frameSize)
        frames_lpc_features = []
        b = (c_double * 32)()

        for k in range(0, numberOfFrames):
            a = (c_double * frameSize)(*frames[k])
            dll.LPC(pointer(a), frameSize, 32, pointer(b))
            frames_lpc_features.append(list(b))

        image_temp1 = np.array(frames_lpc_features)  # list2array
        image_temp2 = image_temp1.transpose()  # array转置
        image_temp3 = np.expand_dims(image_temp2, axis=0)  # 升维
        inputData_array = np.concatenate((inputData_array, image_temp3), axis=0)  # array拼接

    # 删除第一行
    inputData_array = inputData_array[1:]

    # #扩展为4维:(-1, 32, 64, 1)
    inputData_array = np.expand_dims(inputData_array, axis=3)
    # print(inputData_array.shape)
    # 视频的长度是13831, 基本一致
    # (13832, 32, 64, 1)

    # 这里是为了使得视频帧和处理之后的音频长度对齐
    max_l = min(inputData_array.shape[0], num_frames)
    selected_audio = []
    selected_bs_targets = []
    for index, this_audio in enumerate(inputData_array[:max_l]):
        if not isinstance(bs_targets[index], int):
            selected_audio.append(this_audio[np.newaxis, :, :, :])
            selected_bs_targets.append(np.array(bs_targets[index])[np.newaxis, :])
    selected_audio = np.concatenate(selected_audio, axis=0)
    selected_bs_targets = np.concatenate(selected_bs_targets, axis=0)

    selected_bs_targets = selected_bs_targets[:, need_ids]

    print(selected_audio.shape)
    print(selected_bs_targets.shape)

    # 去除共有的前min_len个元素
    min_len = min(selected_audio.shape[0], selected_bs_targets.shape[0])

    np.save("{}_audio.npy".format(flag_name), selected_audio[:min_len])
    np.save("{}_bs_targets.npy".format(flag_name), selected_bs_targets[:min_len])