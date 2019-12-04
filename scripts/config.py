# -*-coding:utf-8-*-
"""
本模块存放配置参数
"""
import glob

IMG_HEIGHT = 256  # 图片高度
IMG_WIDTH = 256  # 图片宽度
CHANNEL = 3  # 图片通道数

EPOCH = 200  # 训练轮数
PATIENCE = 20  # EarlyStopping耐心
TRAIN_SAMPLES = int(len(glob.glob('../data/images/train/*')) * 0.8)  # 训练样本数目
VALID_SAMPLES = int(len(glob.glob('../data/images/train/*')) * 0.2)  # 验证样本数目
NUM_CLASS = 313  # 作者使用313
KERNEL_SIZE = 3  # 卷积核大小
WEIGHT_DECAY = 1e-3  # 衰减
EPSILON = 1e-8  # ε值
NEIGHBOUR_NUM = 5  # 邻居数目，参照论文
T = 0.38  # 温度
