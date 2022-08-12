"""
这是mocapface中抠出的tflite模型
"""
"""
头部姿态估计模型
"""
import numpy as np
import cv2
import json
import tensorflow as tf
class MoCapFace(object):
    def __init__(self):
        """
        使用.tflite来初始化tflite的模型
        """
        self.tfliteInit('model_weights/2001161359.tflite')

    def tfliteInit(self, tflite_file):
        # Initialize the interpreter
        # 初始化解释器
        self.interpreter = tf.lite.Interpreter(model_path=tflite_file)
        # 为tensor分配显存
        self.interpreter.allocate_tensors()

        # 得到输入的place_hoder
        self.input_details = self.interpreter.get_input_details()[0]
        # 获得输出的hooker
        self.blendershapes_output_details = self.interpreter.get_output_details()[0]   # bs系数
        # self.transforms_output_details = self.interpreter.get_output_details()[1]  # 头部朝向

    def forword(self, img):
        """
        img: numpy bgr
        :param img:
        :return:
        """
        test_image = cv2.resize(img, [256, 256])
        test_image = np.expand_dims(test_image, axis=0).astype(self.input_details["dtype"]) / 255.
        self.interpreter.set_tensor(self.input_details["index"], test_image)
        self.interpreter.invoke()
        res = self.interpreter.get_tensor(self.blendershapes_output_details["index"])[0].tolist()
        return res

if __name__ == "__main__":
    mocapface = MoCapFace()
    with open("all_image_path.json", 'r') as im_f:
        images_f = json.load(im_f)

    for img_path in images_f.values():
        img = cv2.imread(img_path)
        res = mocapface.forword(img)
        print(len(res))















