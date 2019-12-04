# -*-coding:utf-8-*-
"""
本模块放置数据生成器
"""
import numpy as np
import cv2
import glob
from sklearn.neighbors import NearestNeighbors
from keras.utils import Sequence
import os
from config import IMG_HEIGHT, IMG_WIDTH, NEIGHBOUR_NUM, TRAIN_SAMPLES, VALID_SAMPLES


class DataGenSequence(Sequence):

    def __init__(self, dataset_type, batch_size):
        """
        初始化
        :param dataset_type:
        """
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        # 判断数据类型
        self.data_folder = '../data/images/train/'
        if self.dataset_type == 'train':
            self.filenames = glob.glob(self.data_folder + '*')[:TRAIN_SAMPLES]  # 取前80%
        else:
            self.filenames = glob.glob(self.data_folder + '*')[TRAIN_SAMPLES:]  # 取后20%

        # 加载量化的ab值
        q_ab = np.load("../data/params/pts_in_hull.npy")
        self.number_q = q_ab.shape[0]
        # 训练近邻算法，论文使用5个邻居
        self.neighbour_model = NearestNeighbors(n_neighbors=NEIGHBOUR_NUM, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        """
        生成器长度
        :return:
        """
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))  # 向上取整

    def __getitem__(self, idx):
        """
        取出一批数据
        :param idx: 第几批
        :return:
        """
        i = idx * self.batch_size
        out_img_height, out_img_width = IMG_HEIGHT // 4, IMG_WIDTH // 4
        batch_size = min(self.batch_size, (len(self.filenames) - i))  # 不足需要取出剩下的
        batch_x = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        batch_y = np.empty((batch_size, out_img_height, out_img_width, self.number_q), dtype=np.float32)

        for i_batch in range(batch_size):
            filename = self.filenames[i]
            try:
                bgr_img = cv2.resize(cv2.imread(filename), (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_CUBIC)  # bgr (0-255]
                gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
                lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

                x = gray_img / 255.  # 输入标准化
                out_lab = cv2.resize(lab_img, (out_img_height, out_img_width), cv2.INTER_CUBIC)
                # 居中图像值 a in [42, 226], b in [20, 223] ---> a in[-86, 98], b in [-108, 95]
                out_ab = out_lab[:, :, 1:].astype(np.int32) - 128  # 丢弃L通道
                y = image_ab_to_y(out_ab, self.neighbour_model, self.number_q)

                if np.random.random_sample() > 0.5:
                    # 以一定概率左右翻转矩阵
                    x = np.fliplr(x)
                    y = np.fliplr(y)

                batch_x[i_batch, :, :, 0] = x
                batch_y[i_batch] = y
                i += 1
            except:
                print(filename)
        return batch_x, batch_y

    def on_epoch_end(self):
        """
        一轮结束则打乱数据集
        :return:
        """
        np.random.shuffle(self.filenames)


def image_ab_to_y(img_ab, neighbour_model, number_q):
    """
    编码目标格式，也就是将a,b映射到ab值上
    作者在这里映射时选取最近的5个邻居并且高斯分布计算分布
    :param img_ab:
    :param neighbour_model:
    :param number_q:
    :return:
    """
    h, w = img_ab.shape[:2]  # 取到长宽
    a = np.ravel(img_ab[:, :, 0])  # a通道展平为向量
    b = np.ravel(img_ab[:, :, 1])  # b通道展平为向量
    ab = np.vstack((a, b)).T  # 叠加
    # 最近5邻居的距离及其下标
    neighbour_distance, neighbour_index = neighbour_model.kneighbors(ab)
    # 使用高斯分布计算分布
    sigma_neighbour = 5
    wts = np.exp(-neighbour_distance ** 2 / (2 * sigma_neighbour ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # 构造目标格式
    y = np.zeros((ab.shape[0], number_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, neighbour_index] = wts
    y = y.reshape((h, w, number_q))
    return y


def get_data_all():
    """
    不使用生成器，一次全部读入内存
    :return:
    """
    def generate():
        """
        没有读取过，则读取并生成npy文件
        :return:
        """
        # 加载量化的ab值
        q_ab = np.load("../data/params/pts_in_hull.npy")
        number_q = q_ab.shape[0]
        # 训练近邻算法，论文使用5个邻居
        neighbour_model = NearestNeighbors(n_neighbors=NEIGHBOUR_NUM, algorithm='ball_tree').fit(q_ab)

        data_folder = '../data/images/train/'
        filenames = glob.glob(data_folder + '*')
        out_img_height, out_img_width = IMG_HEIGHT // 4, IMG_WIDTH // 4
        batch_size = len(filenames)
        batch_x = []
        batch_y = []

        for i_batch in range(batch_size):
            filename = filenames[i_batch]
            bgr_img = cv2.resize(cv2.imread(filename), (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_CUBIC)  # bgr (0-255]
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

            x = gray_img / 255.  # 输入标准化
            out_lab = cv2.resize(lab_img, (out_img_height, out_img_width), cv2.INTER_CUBIC)
            # 居中图像值 a in [42, 226], b in [20, 223] ---> a in[-86, 98], b in [-108, 95]
            out_ab = out_lab[:, :, 1:].astype(np.int32) - 128  # 丢弃L通道
            y = image_ab_to_y(out_ab, neighbour_model, number_q)

            if np.random.random_sample() > 0.5:
                # 以一定概率左右翻转矩阵
                x = np.fliplr(x)
                y = np.fliplr(y)

            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.array(batch_x).astype('float32')
        batch_y = np.array(batch_y).astype('float32')
        np.save('../data/images/x_all.npy', batch_x)
        np.save('../data/images/y_all.npy', batch_y)
        return batch_x, batch_y

    if os.path.exists('../data/images/x_all.npy') and os.path.exists('../data/images/y_all.npy'):
        x_all, y_all = np.load('../data/images/x_all.npy'), np.load('../data/images/y_all.npy')
    else:
        x_all, y_all = generate()
    return x_all, y_all
