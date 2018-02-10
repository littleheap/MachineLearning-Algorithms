# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    N = 300
    x = np.random.rand(N) * 8 - 4  # [-4,4)
    x.sort()
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3 * x) + np.random.randn(N) * 0.01
    # y1 = np.sin(x) + np.random.randn(N) * 0.05
    # y2 = np.cos(x) + np.random.randn(N) * 0.1
    y = np.vstack((y1, y2))
    y = np.vstack((y1, y2)).T
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    deep = 3
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)

    x_test = np.linspace(-4, 4, num=1000).reshape(-1, 1)
    print(x_test)
    y_hat = dt.predict(x_test)
    print(y_hat)
    plt.scatter(y[:, 0], y[:, 1], c='r', s=40, label='Actual')
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='s', s=100, label='Depth=%d' % deep, alpha=1)
    plt.legend(loc='upper left')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.grid()
    plt.show()
