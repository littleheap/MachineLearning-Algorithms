# -*- encoding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import csv


def f(x):
    return 0.5 * np.exp(-(x + 3) ** 2) + np.exp(-x ** 2) + + 0.5 * np.exp(-(x - 3) ** 2)


if __name__ == "__main__":
    # 设定随机种子，保证每次运行结果相同
    np.random.seed(0)
    # 200个红色样本点
    N = 200
    # x属于[-5,5)
    x = np.random.rand(N) * 10 - 5
    # 排序x定义域
    x = np.sort(x)
    # y函数+随机噪声
    y = f(x) + 0.05 * np.random.randn(N)
    # 转换成一列
    x.shape = -1, 1  # 或200, 1

    # CV：cross validation
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False)
    ridged = Pipeline([('poly', PolynomialFeatures(degree=10)), ('Ridge', ridge)])
    # bagging操作，100次，每次取30%样本
    bagging_ridged = BaggingRegressor(ridged, n_estimators=100, max_samples=0.3)
    # 决策树回归
    dtr = DecisionTreeRegressor(max_depth=5)
    # 可以整合四种回归策略
    regs = [
        ('DecisionTree Regressor', dtr),
        ('Ridge Regressor(6 Degree)', ridged),
        ('Bagging Ridge(6 Degree)', bagging_ridged),
        ('Bagging DecisionTree Regressor', BaggingRegressor(dtr, n_estimators=100, max_samples=0.3))]

    x_test = np.linspace(1.1 * x.min(), 1.1 * x.max(), 1000)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # figsize：宽度 高度 || facecolor：背景白色
    plt.figure(figsize=(12, 8), facecolor='w')
    # 训练数据：原始离散样本点，红圈，
    plt.plot(x, y, 'ro', label=u'训练数据')
    # 测试数据：黑色线，粗3.5
    plt.plot(x_test, f(x_test), color='k', lw=3.5, label=u'真实值')
    # 设定四种个颜色
    clrs = 'bmyg'
    # 提取四种评估模型
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        y_test = reg.predict(x_test.reshape(-1, 1))
        plt.plot(x_test, y_test.ravel(), color=clrs[i], lw=i + 1, label=name, zorder=6 - i)
    plt.legend(loc='upper left')
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title(u'回归曲线拟合', fontsize=21)
    plt.ylim((-0.2, 1.2))
    plt.tight_layout(2)
    plt.grid(True)
    plt.show()
