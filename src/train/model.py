# -*-coding:utf-8 -*-

"""
File    : model.py
Time    : 2022/05/30 10:57
Author  : Xu Jiajian
"""

import tensorflow as tf


class HyperParameter:
    ZC_MICRO = "zc_micro"  # 中文-剑桥词典
    MICRO = "micro"  # 剑桥词典
    LARGE = "large"

    # GPU编号
    gpu = "0"

    # 训练超参
    epoch = 150
    batch_size = 256
    dropout_rate = 0.3
    starter_learning_rate = 3e-4

    # 模型超参
    conv_type = ZC_MICRO
    grl_gamma = 15
    l2_reg_weight = 5e-4

    # 数据情况记录
    dataset_id = "dataset44"
    input_dim1 = 32
    input_dim2 = 13
    input_dim3 = 1

    # 输出维度
    bs_label_dim = 16
    classifier_dim = 106  # 106个个体, 这个应该为了应对不同人说话的问题(但是使用文字转语音应该没有这种问题)

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "dropout_rate": self.dropout_rate,
            "starter_learning_rate": self.starter_learning_rate,
            "conv_type": self.conv_type,
            "grl_gamma": self.grl_gamma,  # 这个是和梯度相关的一个系数
            "l2_reg_weight": self.l2_reg_weight,  # l2正则化
            "dataset_id": self.dataset_id,
            "input_dim1": self.input_dim1,
            "input_dim2": self.input_dim2,
            "input_dim3": self.input_dim3,
            "bs_label_dim": self.bs_label_dim,
            "classifier_dim": self.classifier_dim,
        }


class WavBlendShapeModel(tf.keras.Model):
    """
    声音驱动神经网络模型
    """

    def __init__(self,
                 output_size,
                 dropout_rate,
                 l2_reg_weight=1e-4,
                 speakers_class_num=10,
                 name="WavBlendShapeModel",
                 conv_encoder_type="zc_micro",
                 build_adverse_model: bool = True,
                 total_epoch=100,
                 grl_gamma=10.,
                 **kwargs,
                 ):
        super(WavBlendShapeModel, self).__init__(name=name, **kwargs)

        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.l2_reg_weight = l2_reg_weight
        self.speakers_class_num = speakers_class_num  # 说话者的类别
        self.conv_encoder_type = conv_encoder_type  # 卷积编码器的类型
        self.build_adverse_model = build_adverse_model  # 这个应该是一个对抗模型
        self.total_epoch = total_epoch
        self.grl_gamma = grl_gamma

        # 卷积编码器
        if self.conv_encoder_type == "zc_micro":
            self.conv_encoder_function = self.__build_zc_micro_model()
        elif self.conv_encoder_type == "micro":
            self.conv_encoder_function = self.__build_micro_model()
        else:
            self.conv_encoder_function = self.__build_large_model()

        # 回归预测层
        self.regression_layer_function = self.__regression_layers()

        # 说话人识别对抗层
        if self.build_adverse_model:
            self.spk_encoder = self.__build_micro_model(name="speaker_verification_encoder")
            self.grl_layer = GradientReverseLayer(total_epoch=self.total_epoch, gamma=self.grl_gamma)
            self.spk_classifier_function = self.__classifier_layers()

    def __build_zc_micro_model(self):
        self.conv_encoder1 = tf.keras.Sequential([
            CustomConvLayer(
                filters=32, kernel_size=[3, 1], strides=[2, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock1"
            ),
            CustomConvLayer(
                filters=64, kernel_size=[3, 1], strides=[4, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock2"
            ),
            CustomConvLayer(
                filters=96, kernel_size=[3, 1], strides=[4, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock3"
            ),
        ], name="conv_encoder1")
        self.conv_encoder2 = tf.keras.Sequential([
            CustomConvLayer(
                filters=128, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock6"
            ),
            CustomConvLayer(
                filters=128, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock7"
            ),
            CustomConvLayer(
                filters=128, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock8"
            ),
        ], name="conv_encoder2")

        def _forward_func(inputs, input_zc=None, training=True, **kwargs):
            """
            inputs: (16, 32, 13, 1)
            input_zc: (16, 13)
            """
            hidden_1 = self.conv_encoder1(inputs, training=training)
            input_zc = tf.expand_dims(tf.expand_dims(input_zc, 1), 3)
            # 这里只是单纯的concat(但是只有中文的网络有这么做)

            # print(hidden_1.shape, input_zc.shape)
            hidden_2 = tf.concat([hidden_1, input_zc], axis=-1)

            # 这里又是将zc cat到后面了
            hidden_3 = self.conv_encoder2(hidden_2, training=training)

            return hidden_3

        return _forward_func

    def __build_large_model(self, name="conv_encoder"):
        """
        大编码器-large
        """
        self.conv_encoder = tf.keras.Sequential([
            CustomConvLayer(
                filters=64, kernel_size=[3, 1], strides=[2, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock1"
            ),
            CustomConvLayer(
                filters=128, kernel_size=[3, 1], strides=[2, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock2"
            ),
            CustomConvLayer(
                filters=192, kernel_size=[3, 1], strides=[2, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock3"
            ),
            CustomConvLayer(
                filters=256, kernel_size=[3, 1], strides=[4, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock4"
            ),
            CustomConvLayer(
                filters=256, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock6"
            ),
            CustomConvLayer(
                filters=256, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock7"
            ),
            CustomConvLayer(
                filters=256, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock8"
            ),
        ], name=name)

        def _forward_func(inputs, training=True, **kwargs):
            return self.conv_encoder(inputs, training=training)

        return _forward_func

    def __build_micro_model(self, name="conv_encoder"):
        """
        小编码器-micro
        """
        self.conv_encoder = tf.keras.Sequential([
            CustomConvLayer(
                filters=32, kernel_size=[3, 1], strides=[2, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock1"
            ),
            CustomConvLayer(
                filters=64, kernel_size=[3, 1], strides=[4, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock2"
            ),
            CustomConvLayer(
                filters=96, kernel_size=[3, 1], strides=[4, 1], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock3"
            ),
            CustomConvLayer(
                filters=128, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock6"
            ),
            CustomConvLayer(
                filters=128, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock7"
            ),
            CustomConvLayer(
                filters=128, kernel_size=[1, 3], strides=[1, 3], padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="convblock8"
            ),
        ], name=name)

        def _forward_func(inputs, training=True, **kwargs):
            return self.conv_encoder(inputs, training=training)

        return _forward_func

    def __regression_layers(self):
        """
        回归预测层
        """
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer_1 = tf.keras.layers.Dense(
            64, kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="dense1")
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.ln_layer = tf.keras.layers.LayerNormalization(name="dense1_ln")
        self.dense_layer_2 = tf.keras.layers.Dense(
            self.output_size, kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight), name="output")

        def _forward_func(inputs, training=True, **kwargs):
            flat = self.flatten_layer(inputs)
            dense1 = self.dense_layer_1(flat, training=training)
            ln = self.ln_layer(dense1, training=training)
            act1 = tf.keras.activations.swish(ln)
            dropout = self.dropout_layer(act1, training=training)
            output = self.dense_layer_2(dropout, training=training)
            return output

        return _forward_func

    def __classifier_layers(self):
        """
        说话人分类层
        """
        self.cls_flatten_layer = tf.keras.layers.Flatten()
        self.cls_dense_layer_1 = tf.keras.layers.Dense(
            128,
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight),
            name="cls_dense1")
        self.cls_ln_layer = tf.keras.layers.LayerNormalization(name="cls_dense1_ln")
        self.cls_dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.cls_dense_layer_2 = tf.keras.layers.Dense(
            self.speakers_class_num,
            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_reg_weight),
            name="cls_output")
        self.cls_output_activation = tf.keras.layers.Softmax()

        def _forward_func(inputs, training=True, **kwargs):
            flat = self.cls_flatten_layer(inputs)
            dense1 = self.cls_dense_layer_1(flat, training=training)
            ln = self.cls_ln_layer(dense1, training=training)
            act1 = tf.keras.activations.swish(ln)
            dropout = self.cls_dropout_layer(act1, training=training)
            logits = self.cls_dense_layer_2(dropout, training=training)
            output = self.cls_output_activation(logits)
            return output

        return _forward_func

    def call(self, inputs, training=True, adverse_training=True, current_epoch=0, **kwargs):
        conv_hidden = self.conv_encoder_function(inputs, training=training, **kwargs)
        output = self.regression_layer_function(conv_hidden, training=training, **kwargs)

        if self.build_adverse_model is True and adverse_training is True:
            # 只有在训练的时候才使用
            # 将经过了conv2d编码之后的vector送入到 由人生来预测说话的人的身份的回归层中去
            # 隐层编码
            spk_hidden = self.spk_encoder(inputs, training=training, **kwargs)
            # 人物身份分类头
            spk_class = self.spk_classifier_function(
                self.grl_layer(conv_hidden, current_epoch / self.total_epoch) + spk_hidden,
                training=training,
                **kwargs
            )
            return output, spk_class

        return output


class CustomConvLayer(tf.keras.layers.Layer):
    """
    卷积模块
    conv2d-->bn-->swish
    """

    def __init__(self, filters, kernel_size, strides=(1, 1), with_ln=True,
                 activation=tf.keras.activations.swish, name="convblock", **kwargs):
        super(CustomConvLayer, self).__init__(name=name)
        self.with_ln = with_ln
        self.activation = activation

        # 这里使用的是一个2d卷积, 不是想象当中的1d卷积
        self.conv_layer = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, activation=None, strides=strides, name=name + "_conv", **kwargs
        )
        if self.with_ln:
            self.norm = tf.keras.layers.LayerNormalization(name=name + "_ln")
        if self.activation is not None:
            self.act = activation

    def call(self, inputs, training=True):
        hidden = self.conv_layer(inputs, training=training)
        if self.with_ln:
            hidden = self.norm(hidden, training=training)
        if self.activation is not None:
            hidden = self.act(hidden)
        return hidden


class GradientReverseLayer(tf.keras.layers.Layer):
    """
    梯度反转层
    前向传播时不做操作，反向传播时将梯度乘以 -λ
    λ = 2 / (1 + np.exp(-γ * p)) - 1
    p = cur_ep / total_epoch
    γ为超参，默认设置为10
    """

    def __init__(self, total_epoch=100, gamma=10.):
        """
        gamma: 15.0
        """
        super(GradientReverseLayer, self).__init__(name="GRL")
        self.total_epoch = total_epoch
        self.gamma = gamma

    def call(self, inputs, current_epoch):
        """
        p: 会在训练的过程中从0-->1
        """
        # 返回输入的原值和一个函数式子
        return GradientReverseLayer._grad_reverse_op(inputs, p=current_epoch / self.total_epoch, gamma=self.gamma)

    @classmethod
    @tf.custom_gradient
    def _grad_reverse_op(cls, x, p=0., gamma=10.):
        # Return a tensor with the same shape and contents as input.
        # https://blog.csdn.net/TeFuirnever/article/details/88908870
        # 但是这个有必要吗, 新建一个一模一样的tensor
        y = tf.identity(x)
        # 随着epoch的增大, p从0到1逐渐增大, -gamma * p逐渐减小, _lambda逐渐增大
        _lambda = 2 / (1 + tf.exp(-gamma * p)) - 1

        def reverse_grad(dy):
            """
            这个函数应该会传播到这里的梯度乘上一个系数_lambda
            # y是前向传播的时候的, reverse_grad()是反向传播的时候才会用到
            # 看这个博客, 看前半部分就可以理解了
            """
            return -dy * _lambda

        return y, reverse_grad


class ModifiedMarginLoss(tf.keras.losses.Loss):
    """
    自定义区间损失

        | max(y - y_hat, (y_hat - y) / 4),   y > mean
    L = |
        | max(y_hat - y, (y - y_hat) / 4),   y <= mean

    或

        | max(y - y_hat, (y_hat - y) / 4),   y > mean
    L = |
        | abs(y - y_hat)                 ,   y <= mean

    """

    def __init__(self, mean_value=None, augmented_lower_bound=True, name="ModifiedMarginLoss"):
        super(ModifiedMarginLoss, self).__init__(name=name)
        self.mean_value = mean_value
        self.augmented_lower_bound = augmented_lower_bound

    def __call__(self, y_true, y_pred, weight=None):
        if self.mean_value is not None:
            if self.augmented_lower_bound:
                losses = tf.where(
                    tf.greater(y_true, self.mean_value),
                    tf.math.maximum(y_true - y_pred, (y_pred - y_true) / 4),
                    tf.math.maximum(y_pred - y_true, (y_true - y_pred) / 4)
                )
            else:
                losses = tf.where(
                    tf.greater(y_true, self.mean_value),
                    tf.math.maximum(y_true - y_pred, (y_pred - y_true) / 4),
                    tf.abs(y_true - y_pred)
                )
        else:
            losses = tf.abs(y_true - y_pred)

        if weight is None:
            weight = tf.ones_like(y_true)

        losses = tf.reduce_mean(losses * weight)
        return losses


class WeightedHuberLoss(tf.keras.losses.Loss):
    """
    加权 Huber Loss
    主要针对不同列进行加权

    weight 矩阵 [B, Y]，B=batch_size，Y=output_size
    """

    def __init__(self, delta=1., name="WeightedHuberLoss"):
        super(WeightedHuberLoss, self).__init__(name=name)
        # self.delta = tf.cast(delta, dtype=tf.float32)
        self.delta = delta

    def __call__(self, y_true, y_pred, weight=None):
        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
        losses = tf.where(
            abs_error <= self.delta,
            half * tf.square(error),
            self.delta * abs_error - half * tf.square(self.delta))

        if weight is None:
            weight = tf.ones_like(y_true)

        losses = tf.reduce_mean(losses * weight)
        return losses
