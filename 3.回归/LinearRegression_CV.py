# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('Advertising.csv')  # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.75)
    # print(x_train, y_train)
    # model = Lasso()  # L1正则
    model = Ridge()  # 岭回归

    # 0.001 ~ 100
    alpha_can = np.logspace(-3, 2, 10)
    # 5折交叉验证，cv是cross verify交叉验证，给定alpha超参数
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    # 喂数据训练
    lasso_model.fit(x, y)
    print('超参数：\n', lasso_model.best_params_)

    # 获取预测值
    y_hat = lasso_model.predict(np.array(x_test))
    # 计算误差平方均值
    mse = np.average((y_hat - np.array(y_test)) ** 2)
    # 开方衡量误差程度
    rmse = np.sqrt(mse)
    print(mse, rmse)

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
