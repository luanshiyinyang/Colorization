# -*-coding:utf-8-*-
"""
本模块放置自定义损失函数
"""
import numpy as np
import keras.backend as K


def categorical_crossentropy_color(y_true, y_pred):
    """
    优化的交叉熵
    :param y_true:
    :param y_pred:
    :return:
    """
    prior_factor = np.load("../data/params/prior_factor.npy").astype(np.float32)
    q = 313
    y_true = K.reshape(y_true, (-1, q))
    y_pred = K.reshape(y_pred, (-1, q))

    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(prior_factor, idx_max)
    weights = K.reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_entropy = K.categorical_crossentropy(y_pred, y_true)
    cross_entropy = K.mean(cross_entropy, axis=-1)

    return cross_entropy
