# -*-coding:utf-8-*-

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASS, KERNEL_SIZE


def build_model():
    """
    构建论文神经网络
    以下命名尽量按论文命名策略
    使用function API搭建网络
    :return:
    """
    # 输入层
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))  # (H, W, C)

    # conv1层
    x = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv1_1",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(input_tensor)
    x = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv1_2",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3), strides=(2, 2))(x)  # 使用卷积代替池化
    x = BatchNormalization(name="conv1_bn")(x)  # output (128, 128, 64)

    # conv2层
    x = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv2_1",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv2_2",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3), strides=(2, 2))(x)
    x = BatchNormalization(name="conv2_bn")(x)  # output (64, 64, 128)

    # conv3层
    x = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv3_1",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv3_2",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv3_3",
               kernel_initializer="he_normal", strides=(2, 2))(x)
    x = BatchNormalization(name="conv3_bn")(x)  # output (32, 32, 256)

    # conv4层
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv4_1",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv4_2",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv4_3",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization(name="conv4")(x)  # output (32, 32, 512)

    # conv5层
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", dilation_rate=2,  # 膨胀率论文结构标注
               name="conv5_1", kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", dilation_rate=2,
               name="conv5_2", kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", dilation_rate=2,
               name="conv5_3", kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization(name="conv5_bn")(x)  # output (32, 32, 512)

    # conv6层
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", dilation_rate=2,  # 膨胀率
               name="conv6_1", kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", dilation_rate=2,
               name="conv6_2", kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", dilation_rate=2,
               name="conv6_3", kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization(name="conv6_bn")(x)  # output (32, 32, 512)

    # conv7层
    # 原论文这里是512
    x = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv7_1",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv7_2",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv7_3",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization(name="conv7_bn")(x)  # output (32, 32, 512)

    # conv8层
    # 原论文这里是256，考虑到数据集较小，无需过多特征
    x = UpSampling2D(size=(2, 2))(x)  # keras里deconv为upsampling（上采样）
    x = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv8_1",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv8_2",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation="relu", padding="same", name="conv8_3",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)  # output (64, 64, 256)

    outputs = Conv2D(NUM_CLASS, (1, 1), activation="softmax", padding="same", name="pred")(x)  # output (64, 64, 313)

    model_build = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")

    return model_build


if __name__ == "__main__":
    """
    以下代码用于测试是否创建成功已经模型可视化
    """
    # CPU模型
    with tf.device("/cpu:0"):
        model = build_model()
    print(model.summary())

    # GPU模型(必须大于等于2个GPU)
    # model_gpus = multi_gpu_model(model, gpus=4)
    # print(model_gpus.summary())
