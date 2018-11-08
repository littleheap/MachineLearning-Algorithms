import warnings
import numpy as np
import matplotlib as mpl
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin


def expand(a, b):
    d = (b - a) * 0.05
    return a - d, b + d


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # hmmlearn(0.2.0) < sklearn(0.18)
    np.random.seed(0)

    n = 5  # 隐状态数目
    n_samples = 1000
    pi = np.random.rand(n)
    pi /= pi.sum()
    print('初始概率：', pi)  # 长度为5的初始概率

    # 生成n*n的转换概率
    A = np.random.rand(n, n)
    mask = np.zeros((n, n), dtype=np.bool)
    # 特殊位置清0
    mask[0][1] = mask[0][4] = True
    mask[1][0] = mask[1][2] = True
    mask[2][1] = mask[2][3] = True
    mask[3][2] = mask[3][4] = True
    mask[4][0] = mask[4][3] = True
    A[mask] = 0
    for i in range(n):
        A[i] /= A[i].sum()
    print('转移概率：\n', A)

    # 生成5个均值
    means = np.array(((30, 30), (0, 50), (-25, 30), (-15, 0), (15, 0)))
    print('均值：\n', means)

    # 生成5个方差
    covars = np.empty((n, 2, 2))
    for i in range(n):
        # covars[i] = np.diag(np.random.randint(1, 5, size=2))
        covars[i] = np.diag(np.random.rand(2) + 0.001) * 10  # np.random.rand ∈[0,1)
    print('方差：\n', covars)

    # 建立模型
    model = hmm.GaussianHMM(n_components=n, covariance_type='full')
    model.startprob_ = pi
    model.transmat_ = A
    model.means_ = means
    model.covars_ = covars
    sample, labels = model.sample(n_samples=n_samples, random_state=0)

    # 估计参数
    model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=10)
    model = model.fit(sample)
    y = model.predict(sample)
    np.set_printoptions(suppress=True)
    print('##估计初始概率：\n', model.startprob_)
    print('##估计转移概率：\n', model.transmat_)
    print('##估计均值：\n', model.means_)
    print('##估计方差：\n', model.covars_)

    # 类别
    order = pairwise_distances_argmin(means, model.means_, metric='euclidean')
    print(order)
    pi_hat = model.startprob_[order]
    A_hat = model.transmat_[order]
    A_hat = A_hat[:, order]
    means_hat = model.means_[order]
    covars_hat = model.covars_[order]
    change = np.empty((n, n_samples), dtype=np.bool)
    for i in range(n):
        change[i] = y == order[i]
    for i in range(n):
        y[change[i]] = i
    print('估计初始概率：', pi_hat)
    print('估计转移概率：\n', A_hat)
    print('估计均值：\n', means_hat)
    print('估计方差：\n', covars_hat)
    print(labels)
    print(y)
    acc = np.mean(labels == y) * 100
    print('准确率：%.2f%%' % acc)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.scatter(sample[:, 0], sample[:, 1], s=50, c=labels, cmap=plt.cm.Spectral, marker='o',
                label=u'观测值', linewidths=0.5, zorder=20)
    plt.plot(sample[:, 0], sample[:, 1], 'r-', zorder=10)
    plt.scatter(means[:, 0], means[:, 1], s=100, c=np.random.rand(n), marker='D', label=u'中心', alpha=0.8, zorder=30)
    x1_min, x1_max = sample[:, 0].min(), sample[:, 0].max()
    x2_min, x2_max = sample[:, 1].min(), sample[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
