# -*-coding:utf-8 -*-
"""
# File   : recorder.py
# Time   : 2022/8/11 18:30
# Author : luzeng
"""

import time
import wave
import threading
import pyaudio
import keyboard
import numpy as np

class Recorder:
    def __init__(self, animation, input_length=2048, chunk=1024, channels=1, rate=16000):
        self.animation = animation
        self.CHUNK = chunk
        self.input_length = input_length
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = False
        self._input_signal_stream = [0] * self.input_length  # 512 * 3
        self._frames = []
        self.p = pyaudio.PyAudio()

    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        self._frames = []
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)
        output_stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, output=True)

        while self._running:
            # 音频输入
            data = stream.read(self.CHUNK)
            # byte --> numpy
            voice = np.frombuffer(data, "<i2").astype(np.float32)

            # 数据流模式
            # 就是将当前帧的音频放到中间, 并且self._input_signal_stream是不断更新的
            # print("self._input_signal_stream的长度", len(self._input_signal_stream))
            # print("start index", voice.shape[0])

            self._input_signal_stream = self._input_signal_stream[voice.shape[0]:] + voice.tolist()

            # 归一化, 小值过滤
            _input_signal = np.array(self._input_signal_stream, dtype=np.float32)

            # 增加一个高斯噪声
            _input_signal = _input_signal * np.where(np.abs(_input_signal) > 100, 1., 0.) / 32767.

            # 进队列，供AI模型消费
            # 将读取到的音频放进去进行, 有对应的已经开启了的线程会去读取它并送入到inference中去
            self.animation.input_frames_data(_input_signal)

            # 声音播放
            output_stream.write(data)

        stream.stop_stream()
        stream.close()

    def stop(self):
        self._running = False

    def save(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved to {}".format(filename))

    def record(self):
        while True:
            print('请按下回车键开始录音：')
            keyboard.wait("enter")
            print("Start recording........")
            begin = time.time()
            self.start()
            print('按下回车键暂停')
            keyboard.wait("enter")
            print("Stop recording")
            self.stop()
            fina = time.time()
            t = fina - begin
            print('录音时间为%ds' % t)
