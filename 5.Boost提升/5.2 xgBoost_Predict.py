# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split  # cross_validation


def iris_type(s):
    it = {b'Iris-setosa': 0,
          b'Iris-versicolor': 1,
          b'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = u'..\\3.回归\\iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    # 测试数据设置50，训练数据则为100
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    # 训练数据和标记组装
    data_train = xgb.DMatrix(x_train, label=y_train)
    # 测试数据和标记组装
    data_test = xgb.DMatrix(x_test, label=y_test)
    # 测试数据和训练数据整合
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    # objective：多分类问题，用softmax
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    # 训练函数
    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    # 预测标记
    y_hat = bst.predict(data_test)
    # 计算结果
    result = y_test.reshape(1, -1) == y_hat
    print('正确率:\t', float(np.sum(result)) / len(y_hat))
