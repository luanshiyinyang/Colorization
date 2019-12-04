# -*-coding:utf-8-*-

import cv2 as cv
import numpy as np
import tqdm


def rgb2lab():
    """
    显示RGB转Lab映射范围
    :return:
    """
    # 记录所有L，a，b组合
    L = [0] * (256 ** 3)
    a = [0] * (256 ** 3)
    b = [0] * (256 ** 3)

    index = 0
    for R in tqdm.tqdm(range(256)):
        for G in range(256):
            for B in range(256):
                im = np.array((B, G, R), np.uint8).reshape((1, 1, 3))
                cv.cvtColor(im, cv.COLOR_BGR2LAB, im)
                L[index] = im[0, 0, 0]
                a[index] = im[0, 0, 1]
                b[index] = im[0, 0, 2]
                index += 1

    print("L in [{},{}]".format(min(L), max(L)))
    print("a in [{},{}]".format(min(a), max(a)))
    print("b in [{},{}]".format(min(b), max(b)))


if __name__ == '__main__':
    rgb2lab()
