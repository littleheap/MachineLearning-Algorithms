# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split  # cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

'''
    数据集：分类三种酒，对应的13个特征，第1列为标记数据，后面13列为13种特征
'''


# 正确率计算函数
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)


if __name__ == "__main__":
    data = np.loadtxt('wine.data', dtype=float, delimiter=',')
    # 第1列是分割点，前面是标记数据y，后面的是特征向量x
    y, x = np.split(data, (1,), axis=1)
    # x正则化，保证每一列均值是0，方差为1
    # x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    # Logistic回归
    lr = LogisticRegression(penalty='l2')  # L2正则
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic回归 ')

    # XGBoost
    # 因为当前版本要求标记从0开始，所以把为3的标记设置为0，形成0 1 2三类标记
    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost ')
