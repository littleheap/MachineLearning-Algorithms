# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

if __name__ == "__main__":

    np.random.seed(0)  # 保证随机种子一定
    M = 20  # 20个样本
    N = 5  # 每个数据是5维的

    x = np.random.randint(2, size=(M, N))  # [low, high) 给定2情况下，只能随机int即0和1，生成20*5的矩阵
    print('x = \n', x)
    print('x.shape = ', x.shape)

    x = np.array(list(set([tuple(t) for t in x])))  # 去重数据，去掉同样特征对应不同类别的数据，set元祖的元素，再list就会去重
    print('new x = \n', x)
    print('new x.shape = ', x.shape)

    M = len(x)
    y = np.arange(M)  # 制造类别数据，此处是0~16

    mnb = MultinomialNB(alpha=1)  # 可尝试切换成GaussianNB()
    # mnb = GaussianNB()  # 可以达到100%，在去重的情况下
    mnb.fit(x, y)
    y_hat = mnb.predict(x)
    print('预测类别：', y_hat)
    print('准确率：%.2f%%' % (100 * np.mean(y_hat == y)))
    print('系统得分：', mnb.score(x, y))
    # from sklearn import metrics
    # print metrics.accuracy_score(y, y_hat)  # 和上面一样
    err = y_hat != y
    print('错误情况：\n', err)
    for i, e in enumerate(err):
        if e:
            print(y[i], '：\t', x[i], '被认为与', x[y_hat[i]], '一个类别')
