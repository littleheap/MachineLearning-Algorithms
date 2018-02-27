# -*- coding:utf-8 -*-

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
    线性回归：目标给每个特征分配合理权重，并额外评估出一个偏移量
    数据样例：3种不同渠道投入对应的广告收入，TV || Radio || Newspaper => Sales，一共200条数据
'''

if __name__ == "__main__":
    # 数据路径
    path = 'Advertising.csv'

    '''
    # 手写读取数据 - 请自行分析，在Iris代码中给出类似的例子
    f = open(path)
    x = []
    y = []
    for i, d in enumerate(f):
        # 第0行数据类别不要
        if i == 0:
            continue
        # 去空格等不标准输入
        d = d.strip()
        # 如果没有数据
        if not d:
            continue
        # 分割数据
        d = list(map(float, d.split(',')))
        # 排除第一列索引，从第二列读到倒数第二列
        x.append(d[1:-1])
        # 最后一列为sale
        y.append(d[-1])
    print(x)
    print(y)
    x = np.array(x)
    y = np.array(y)
    print('------------------------------')
    '''

    '''    
    # python自带库
    with open(path, "rt", encoding="utf-8") as vsvfile:
        reader = csv.reader(vsvfile)
        rows = [row for row in reader]
        print(rows)
        print('------------------------------')
    '''

    '''
    # numpy读入
    p = np.loadtxt(path, delimiter=',', skiprows=1)  # 省略第1行
    print(p)
    print('------------------------------')
    '''

    # pandas读入
    data = pd.read_csv(path)  # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)

    # # 绘制1
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')  # 图例显示位置
    plt.grid()
    plt.show()

    # 绘制2
    plt.figure(figsize=(9, 12))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

    print('-----------------------------------')
    # 分离训练测试数据，random_state是随机种子，因为Python中随机种子变化，所以此处固定
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print(x_train, y_train)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print(linreg.coef_)  # 系数
    print(linreg.intercept_)  # 截距

    y_hat = linreg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # 均方误差：平方和取均值
    rmse = np.sqrt(mse)  # 求平方根
    print(mse, rmse)

    # 绘制测试值和预测值
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
