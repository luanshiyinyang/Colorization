# -*-coding:utf-8-*-
"""
本模块演示预训练模型效果
"""
import glob
import cv2
import numpy as np
from argparse import ArgumentParser
import os
from config import IMG_HEIGHT, IMG_WIDTH, NEIGHBOUR_NUM, T, EPSILON
from model import build_model


def parse_argument():
    """
    解析命令行参数
    :return:
    """
    ap = ArgumentParser()
    ap.add_argument("-o", "--output", default='../results/', help="colorization result output folder")
    ap.add_argument("-i", "--input", default='../data/images/test/', help="input test images folder")
    args = vars(ap.parse_args())
    return args


def get_images(path):
    """
    读取本地图片
    :param path:
    :return:
    """
    filenames = glob.glob(path + "*")
    images_bgr = []
    images_gray = []
    for i in range(len(filenames)):
        print("processing image{}".format(i+1))

        filename = filenames[i]
        bgr = cv2.imread(filename)  # 读入彩色图
        bgr = cv2.resize(bgr, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_CUBIC)

        gray = cv2.imread(filename, 0)  # 读入灰度图
        gray = cv2.resize(gray, (IMG_HEIGHT, IMG_HEIGHT), cv2.INTER_CUBIC)
        images_bgr.append(bgr)
        images_gray.append(gray)
    return np.array(images_bgr), np.array(images_gray)


def test(args):
    """
    预测
    :param args
    :return:
    """
    if not os.path.exists('../models/training_best_weights_128end.h5'):
        print("no model in root/models")
    else:
        model = build_model()
        model.load_weights('../models/training_best_weights_128end.h5')
        h, w = IMG_HEIGHT // 4, IMG_WIDTH // 4
        q_ab = np.load("../data/params/pts_in_hull.npy")
        number_q = q_ab.shape[0]

        images_bgr, images_gray = get_images(args['input'])
        raws = []
        grays = []
        preds = []
        for i in range(images_bgr.shape[0]):
            x_test = np.empty((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
            x_test[0, :, :, 0] = images_gray[i] / 255.
            x_colorized = model.predict(x_test).reshape(h*w, number_q)

            # 调整概率
            x_colorized = np.exp(np.log(x_colorized+EPSILON) / T)
            x_colorized = x_colorized / np.sum(x_colorized, 1)[:, np.newaxis]
            # 调整权重
            q_a = q_ab[:, 0].reshape((1, 313))
            q_b = q_ab[:, 1].reshape((1, 313))
            x_a = np.sum(x_colorized*q_a, 1).reshape((h, w))
            x_b = np.sum(x_colorized*q_b, 1).reshape((h, w))
            x_a = cv2.resize(x_a, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_CUBIC)
            x_b = cv2.resize(x_b, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_CUBIC)
            x_a = x_a + 128
            x_b = x_b + 128
            out_lab = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.int32)
            out_lab[:, :, 0] = cv2.cvtColor(cv2.cvtColor(images_gray[i], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)[:, :, 0]
            out_lab[:, :, 1] = x_a
            out_lab[:, :, 2] = x_b
            out_lab = out_lab.astype(np.uint8)
            out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
            out_bgr = out_bgr.astype(np.uint8)

            if not os.path.exists(args['output']):
                os.mkdir(args['output'])

            cv2.imwrite(args_['output'] + 'img{}_raw.png'.format(i), images_bgr[i])
            raws.append(images_bgr[i])
            cv2.imwrite(args_['output'] + 'img{}_gray.png'.format(i), images_gray[i])
            grays.append(images_gray[i])
            cv2.imwrite(args_['output'] + 'img{}_pred.png'.format(i), out_bgr)
            preds.append(out_bgr)

        import matplotlib.pyplot as plt
        for i in range(3):
            plt.subplot(3, 3, i+1)
            plt.imshow(cv2.cvtColor(raws[i], cv2.COLOR_BGR2RGB))
            plt.subplot(3, 3, i+3+1)
            plt.imshow(grays[i], cmap='gray')
            plt.subplot(3, 3, i+6+1)
            plt.imshow(cv2.cvtColor(preds[i], cv2.COLOR_BGR2RGB))
        plt.savefig('zc.png')


if __name__ == '__main__':
    args_ = parse_argument()
    test(args_)
