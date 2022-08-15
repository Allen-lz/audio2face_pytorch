# audio2face
## 一、依赖

- requirements.txt

```
tqdm
glob2
keyboard
numpy
pandas
scipy
matplotlib
librosa==0.9.1
mediapipe==0.8.9.1
moviepy==1.0.3
paddleocr==2.5.0.3
PyAudio==0.2.11
sk_video==1.1.10
SoundFile==0.10.3.post1
tensorflow==2.6.0
webrtcvad==2.0.10
```

- 安装依赖

```shell
pip install -r requirements.txt
```

**若无法安装 `webrtcvad` ，可以尝试使用 `pip install webrtcvad-wheels` 命令单独安装 `webrtcvad` 库**



## 二、运行

### 1. 演示demo运行

- 运行录音demo

```shell
cd src/demo
python run_demo.py --only_bs_data
```

- 运行实时demo

```shell
cd src/demo
python run_demo.py --real_time
```
