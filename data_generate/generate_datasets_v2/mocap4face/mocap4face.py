
import json
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import time

addr = '192.168.38.6'

blendershapes_index_map = [
    "None",
    "browInnerUp",
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
    "headDown",
    "headLeft",
    "headRight",
    "headRollLeft",
    "headRollRight",
    "headUp",
    "tongueOut",
]
body1 = """{"frame":81,"timestamp":1653020274303}"""
body2 = """#{"cmdList":[{"k":0,"v":{"x":-0.16915,"y":0.44524,"z":-0.14412},"visibility":0.99242},{"k":1,"v":{"x":-0.23624,"y":0.28103,"z":-0.13774},"visibility":0.80215},{"k":2,"v":{"x":-0.25922,"y":0.1164,"z":-0.16851},"visibility":0.3386},{"k":3,"v":{"x":-0.24703,"y":0.10263,"z":-0.18967},"visibility":0.35001},{"k":4,"v":{"x":-0.25588,"y":0.06091,"z":-0.18511},"visibility":0.27888},{"k":5,"v":{"x":0.11057,"y":0.48848,"z":-0.03623},"visibility":0.99176},{"k":6,"v":{"x":0.3029,"y":0.40632,"z":-0.03385},"visibility":0.66099},{"k":7,"v":{"x":0.43823,"y":0.3376,"z":-0.17424},"visibility":0.56821},{"k":8,"v":{"x":0.42734,"y":0.3283,"z":-0.19942},"visibility":0.57001},{"k":9,"v":{"x":0.47437,"y":0.31352,"z":-0.20123},"visibility":0.47294},{"k":10,"v":{"x":0.04314,"y":0.64238,"z":-0.10405},"visibility":0.99465},{"k":11,"v":{"x":0.02436,"y":0.66783,"z":-0.2133},"visibility":0.99679},{"k":12,"v":{"x":-0.08512,"y":0.63474,"z":-0.14779},"visibility":0.9982},{"k":13,"v":{"x":-0.00396,"y":0.66814,"z":-0.22631},"visibility":0.99743},{"k":14,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":15,"v":{"x":-0.10621,"y":0.00281,"z":0.00741},"visibility":1.0},{"k":16,"v":{"x":-0.09716,"y":-0.37997,"z":0.00613},"visibility":1.0},{"k":17,"v":{"x":-0.08832,"y":-0.73587,"z":0.19865},"visibility":1.0},{"k":18,"v":{"x":-0.13203,"y":-0.85234,"z":0.09052},"visibility":1.0},{"k":19,"v":{"x":0.10582,"y":-0.00233,"z":-0.00702},"visibility":1.0},{"k":20,"v":{"x":0.12449,"y":-0.38901,"z":0.0043},"visibility":1.0},{"k":21,"v":{"x":0.15222,"y":-0.72213,"z":0.18485},"visibility":1.0},{"k":22,"v":{"x":0.18626,"y":-0.83152,"z":0.06556},"visibility":1.0},{"k":23,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":24,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":25,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":26,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":27,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781}],"status":0,"valid":1}"""


def resize_img_keep_ratio(img, target_size=(800, 800)):
    old_size = img.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255))
    return img_new


class MediapipeFaceDetection:
    def __init__(self, tflite_path="./2001161359.tflite", json_path="./2001161359.json"):
        self.face_det = self.MediapipeInit()
        self.tfliteInit(tflite_path)
        self.getMocapDict(json_path)

    def tfliteInit(self, tflite_file):
        # Initialize the interpreter
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.blendershapes_output_details = self.interpreter.get_output_details()[0]
        self.transforms_output_details = self.interpreter.get_output_details()[1]

    def MediapipeInit(self):
        face_det = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=0
        )
        return face_det

    def MediapipeRun(self, image, return_face=False):
        # Convert the BGR image to RGB before processing.
        results = self.face_det.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections is None:
            return None, None
        h, w, _ = image.shape
        face = results.detections[0]
        cx = int(face.location_data.relative_bounding_box.xmin * w)
        cy = int(face.location_data.relative_bounding_box.ymin * h)
        height = int(face.location_data.relative_bounding_box.height * h)
        width = int(face.location_data.relative_bounding_box.width * w)

        side_length = max(height, width) + 60

        y_start = int(max((cy + cy + height) / 2 - side_length / 2, 0.))
        y_end = int(min(y_start + side_length, h))
        x_start = int(max((cx + cx + width) / 2 - side_length / 2, 0.))
        x_end = int(min(x_start + side_length, w))

        # face_image = image[cy:cy+height, cx:cx+width]
        face_image = image[y_start:y_end, x_start:x_end]

        # test_image = face_image
        test_image = cv2.resize(face_image, (256, 256))
        # test_image = resize_img_keep_ratio(face_image, (256, 256))
        s1 = time.time()
        test_image = np.expand_dims(test_image, axis=0).astype(self.input_details["dtype"]) / 255.
        self.interpreter.set_tensor(self.input_details["index"], test_image)
        self.interpreter.invoke()
        blendershapes = self.interpreter.get_tensor(self.blendershapes_output_details["index"])[0] * 100
        transforms = self.interpreter.get_tensor(self.transforms_output_details["index"])
        # print(time.time() - s1)
        if return_face:
            return blendershapes, transforms, test_image
        return blendershapes, transforms

    def MediapipeRunWithoutFaceDetect(self, image):
        # Convert the BGR image to RGB before processing.
        face_image = image
        test_image = cv2.resize(face_image, (256, 256))
        # test_image = resize_img_keep_ratio(face_image, (256, 256))
        s1 = time.time()
        test_image = np.expand_dims(test_image, axis=0).astype(self.input_details["dtype"]) / 255.
        self.interpreter.set_tensor(self.input_details["index"], test_image)
        self.interpreter.invoke()
        blendershapes = self.interpreter.get_tensor(self.blendershapes_output_details["index"])[0] * 100
        transforms = self.interpreter.get_tensor(self.transforms_output_details["index"])
        # print(time.time() - s1)
        return blendershapes, transforms

    def jsonFormat(self, prediction):
        json_kv = {}
        for idx, emoji_val in enumerate(prediction):
            emoji_name = self.mocap[idx]
            json_kv[emoji_name] = emoji_val
        return json_kv

    def getMocapDict(self, path):
        with open(path, 'r') as f:
            j = json.load(f)
        self.mocap = j['model_metadata']['outputs']['blendshapes']['names']
        self.blendershapes_map = {}
        for index, bs in enumerate(blendershapes_index_map):
            self.blendershapes_map[bs] = index


if __name__ == "__main__":
    fmp = MediapipeFaceDetection()
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("打开视频失败！")

    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps:", fps)
    print("size:", size)
    print("fNUMS:", fNUMS)

    f_cnt = 0
    time_cnt = 0
    while True:
        _, frame = capture.read()
        if frame is None:
            break
        f_cnt += 1
        res, _ = fmp.MediapipeRun(frame)
        face_json = fmp.jsonFormat(res)

    capture.release()
