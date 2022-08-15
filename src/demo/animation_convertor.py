# -*-coding:utf-8 -*-

"""
# File   : animation_convertor.py
# Time   : 2022/8/13 19:51
# Author : luzeng
# version: python 3.6
"""

import json
import copy

ARGUMENTS_ORDER = [
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

EXPRESS_LIST_TEMPLATE = {
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

BODY1 = """{"frame":81,"timestamp":1653020274303}#"""
BODY2 = """{"cmdList":[{"k":0,"v":{"x":-0.16915,"y":0.44524,"z":-0.14412},"visibility":0.99242},{"k":1,"v":{"x":-0.23624,"y":0.28103,"z":-0.13774},"visibility":0.80215},{"k":2,"v":{"x":-0.25922,"y":0.1164,"z":-0.16851},"visibility":0.3386},{"k":3,"v":{"x":-0.24703,"y":0.10263,"z":-0.18967},"visibility":0.35001},{"k":4,"v":{"x":-0.25588,"y":0.06091,"z":-0.18511},"visibility":0.27888},{"k":5,"v":{"x":0.11057,"y":0.48848,"z":-0.03623},"visibility":0.99176},{"k":6,"v":{"x":0.3029,"y":0.40632,"z":-0.03385},"visibility":0.66099},{"k":7,"v":{"x":0.43823,"y":0.3376,"z":-0.17424},"visibility":0.56821},{"k":8,"v":{"x":0.42734,"y":0.3283,"z":-0.19942},"visibility":0.57001},{"k":9,"v":{"x":0.47437,"y":0.31352,"z":-0.20123},"visibility":0.47294},{"k":10,"v":{"x":0.04314,"y":0.64238,"z":-0.10405},"visibility":0.99465},{"k":11,"v":{"x":0.02436,"y":0.66783,"z":-0.2133},"visibility":0.99679},{"k":12,"v":{"x":-0.08512,"y":0.63474,"z":-0.14779},"visibility":0.9982},{"k":13,"v":{"x":-0.00396,"y":0.66814,"z":-0.22631},"visibility":0.99743},{"k":14,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":15,"v":{"x":-0.10621,"y":0.00281,"z":0.00741},"visibility":1.0},{"k":16,"v":{"x":-0.09716,"y":-0.37997,"z":0.00613},"visibility":1.0},{"k":17,"v":{"x":-0.08832,"y":-0.73587,"z":0.19865},"visibility":1.0},{"k":18,"v":{"x":-0.13203,"y":-0.85234,"z":0.09052},"visibility":1.0},{"k":19,"v":{"x":0.10582,"y":-0.00233,"z":-0.00702},"visibility":1.0},{"k":20,"v":{"x":0.12449,"y":-0.38901,"z":0.0043},"visibility":1.0},{"k":21,"v":{"x":0.15222,"y":-0.72213,"z":0.18485},"visibility":1.0},{"k":22,"v":{"x":0.18626,"y":-0.83152,"z":0.06556},"visibility":1.0},{"k":23,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":24,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":25,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":26,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781},{"k":27,"v":{"x":0.01971,"y":0.63476,"z":-0.22465},"visibility":0.99781}],"status":0,"valid":1}#"""


class AnimationConvertor:

    def __init__(self, path, add_head_pose=True):
        self.current_step = 0
        self.path = path
        self.add_head_pose = add_head_pose
        self.__load_head_animations()

    def convert_to_json_str_3(self, animation_list, bs_name):
        indices = [ARGUMENTS_ORDER.index(name) if name in ARGUMENTS_ORDER else -1 for name in bs_name]
        json_list = []
        for i, frame in enumerate(animation_list):
            express_list = copy.deepcopy(EXPRESS_LIST_TEMPLATE)
            for ind, val in zip(indices, frame):
                if ind == -1:
                    continue
                express_list["ExpressList"][ind - 1]["v"] = min(max(val * 100, 0.), 100.)
            if self.add_head_pose:
                express_list = self.add_head_animations(express_list)
            json_str = BODY1 + BODY2 + json.dumps(express_list)
            json_list.append(json_str)
        return json_list

    def add_head_animations(self, expression):
        if self.current_step >= len(self.head_animations):
            self.current_step = 0
        current_animation = self.head_animations[self.current_step]

        # 头部
        expression["ExpressList"][ARGUMENTS_ORDER.index("headDown") - 1]["v"] = current_animation["headdown"]
        expression["ExpressList"][ARGUMENTS_ORDER.index("headLeft") - 1]["v"] = current_animation["headleft"]
        expression["ExpressList"][ARGUMENTS_ORDER.index("headRight") - 1]["v"] = current_animation["headright"]
        expression["ExpressList"][ARGUMENTS_ORDER.index("headRollLeft") - 1]["v"] = current_animation["headrollleft"]
        expression["ExpressList"][ARGUMENTS_ORDER.index("headRollRight") - 1]["v"] = current_animation["headrollright"]
        expression["ExpressList"][ARGUMENTS_ORDER.index("headUp") - 1]["v"] = current_animation["headup"]

        # 眨眼
        eye_blink = max(current_animation["eyeblinkleft"], current_animation["eyeblinkright"])
        expression["ExpressList"][ARGUMENTS_ORDER.index("eyeBlinkLeft") - 1]["v"] = eye_blink
        expression["ExpressList"][ARGUMENTS_ORDER.index("eyeBlinkRight") - 1]["v"] = eye_blink

        self.current_step += 1
        return expression

    def __load_head_animations(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.head_animations = [json.loads(line.strip()) for line in f.readlines()]
