# -*-coding:utf-8-*-
"""颜色再平衡，为了阻碍背景色的影响，创造颜色鲜明的色彩
"""

import cv2 as cv
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import sklearn.neighbors as nn
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
import glob


def load_sample_data():
    """
    载入数据转为Lab并去除ab通道
    内存要大
    :return:
    """
    img_size = 64
    filenames = glob.glob('../data/images/train/*')
    np.random.shuffle(filenames)
    num_samples = 1300000  # 按照论文，采样1.3M
    X_ab = np.empty((num_samples, img_size, img_size, 2))
    for i in range(num_samples):
        filename = filenames[i]
        bgr = cv.resize(cv.imread(filename), (img_size, img_size), cv.INTER_CUBIC)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        lab = lab.astype(np.int32)
        X_ab[i] = lab[:, :, 1:] - 128
    return X_ab


def compute_color_prior(X_ab, show_grid=False):
    """
    计算先验色
    :param X_ab:
    :param show_grid:
    :return:
    """
    q_ab = np.load("../data/params/pts_in_hull.npy")

    if show_grid:
        plt.figure(figsize=(15, 15))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        for i in range(q_ab.shape[0]):
            ax.scatter(q_ab[:, 0], q_ab[:, 1])
            ax.annotate(str(i), (q_ab[i, 0], q_ab[i, 1]), fontsize=6)
            ax.set_xlim([-110, 110])
            ax.set_ylim([-110, 110])

    X_a = np.ravel(X_ab[:, :, :, 0])
    X_b = np.ravel(X_ab[:, :, :, 1])
    X_ab = np.vstack((X_a, X_b)).T

    if show_grid:
        plt.hist2d(X_ab[:, 0], X_ab[:, 1], bins=100, normed=True, norm=LogNorm())
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        plt.colorbar()
        plt.show()
        plt.clf()
        plt.close()

    nearest = nn.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(q_ab)
    distances, index = nearest.kneighbors(X_ab)

    index = np.ravel(index)
    counts = np.bincount(index)
    idxs = np.nonzero(counts)[0]
    prior_prob = np.zeros((q_ab.shape[0]))
    for i in range(q_ab.shape[0]):
        prior_prob[idxs] = counts[idxs]

    # 返回先验概率
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    # Save
    np.save("../data/params/prior_prob.npy", prior_prob)

    if show_grid:
        plt.hist(prior_prob, bins=100)
        plt.yscale("log")
        plt.show()


def smooth_color_prior(sigma=5, do_plot=False):
    prior_prob = np.load("../data/params/prior_prob.npy")
    prior_prob += 1e-8 * np.min(prior_prob)
    prior_prob = prior_prob / (1.0 * np.sum(prior_prob))

    f = interp1d(np.arange(prior_prob.shape[0]), prior_prob)
    xx = np.linspace(0, prior_prob.shape[0] - 1, 1000)
    yy = f(xx)
    window = gaussian(2000, sigma)  # 2000 pts in the window, sigma=5
    smoothed = convolve(yy, window / window.sum(), mode='same')
    fout = interp1d(xx, smoothed)
    prior_prob_smoothed = np.array([fout(i) for i in range(prior_prob.shape[0])])
    prior_prob_smoothed = prior_prob_smoothed / np.sum(prior_prob_smoothed)

    np.save("../data/params/prior_prob_smoothed.npy", prior_prob_smoothed)

    if do_plot:
        plt.plot(prior_prob)
        plt.plot(prior_prob_smoothed, "g--")
        plt.plot(xx, smoothed, "r-")
        plt.yscale("log")
        plt.show()


def compute_prior_factor(gamma=0.5, alpha=1, do_plot=False):
    prior_prob_smoothed = np.load("../data/params/prior_prob_smoothed.npy")
    u = np.ones_like(prior_prob_smoothed)
    u = u / np.sum(1.0 * u)
    prior_factor = (1 - gamma) * prior_prob_smoothed + gamma * u
    prior_factor = np.power(prior_factor, -alpha)
    prior_factor = prior_factor / (np.sum(prior_factor * prior_prob_smoothed))
    np.save("../data/prior_factor.npy", prior_factor)

    if do_plot:
        plt.plot(prior_factor)
        plt.yscale("log")
        plt.show()


if __name__ == '__main__':
    X_ab = load_sample_data()
    compute_color_prior(X_ab, show_grid=True)
    smooth_color_prior(do_plot=True)
    compute_prior_factor(do_plot=True)
