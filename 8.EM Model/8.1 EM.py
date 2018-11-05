import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    style = 'sklearn'  # or 'others'

    np.random.seed(0)

    # 第一类数据
    # 均值取原点(0,0,0)
    mu1_fact = (0, 0, 0)
    # 方差取3*3单位阵
    cov_fact = np.identity(3)
    print(cov_fact)
    # 给定（均值，方差，样本数量）会返回400*3矩阵数据
    data1 = np.random.multivariate_normal(mu1_fact, cov_fact, 400)

    # 第二类数据
    # 均值取原点(2,2,1)
    mu2_fact = (2, 2, 1)
    # 方差取3*3单位阵
    cov_fact = np.identity(3)
    # 给定（均值，方差，样本数量）会返回100*3矩阵数据
    data2 = np.random.multivariate_normal(mu2_fact, cov_fact, 100)  # 给定（均值，方差，样本数量）会返回100*3矩阵数据

    # 垂直堆叠两类数据，形成500*3矩阵
    data = np.vstack((data1, data2))
    # 500个数据标记，无监督不用，用来计算正确率
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':
        # n_components：类别数量 || covariance_type：方差类型（full、tied、diag、spherical） || tol：容差 || max_iter：最大迭代次数
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        g.fit(data)
        print('类别概率:\t', g.weights_[0])
        print('均值:\n', g.means_, '\n')  # 实际均值(0,0,0)和(2,2,1)
        print('方差:\n', g.covariances_, '\n')  # 实际方差是3*3单位阵
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        n, d = data.shape
        # 随机指定
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print(i, ":\t", mu1, mu2)
        print('类别概率:\t', pi)
        print('均值:\t', mu1, mu2)
        print('方差:\n', sigma1, '\n\n', sigma2, '\n')

    # 预测分类
    # 将求出来的2个高斯模型的4个参数代入高斯模型
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    # 概率密度函数值
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'原始数据', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    # 确保均值的对应顺序
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    print(u'准确率：%.2f%%' % (100 * acc))
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)
    plt.suptitle(u'EM算法的实现', fontsize=20)
    plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.show()
