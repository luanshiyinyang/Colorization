# -*-coding:utf-8-*-
"""
本模块放置工具包
"""
from tensorflow.python.client import device_lib
import multiprocessing


def get_available_gpus():
    """
    返回可用的GPU数目
    :return:
    """
    return len([gpu.name for gpu in device_lib.list_local_devices() if gpu.device_type == 'gpu'])


def get_available_cpus():
    """
    返回可用的CPU数目
    :return:
    """
    return multiprocessing.cpu_count()


if __name__ == '__main__':
    print(get_available_gpus())
    print(get_available_cpus())

