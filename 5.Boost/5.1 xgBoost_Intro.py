import numpy as np
import xgboost as xgb

'''
    数据内容：蘑菇125特征对应是否有毒，测试集训练集简化存储模式，只将125特征列中，存在为是的特征标为1并集合。
'''


# 自定义损失函数的梯度和二阶导
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0 - p)
    return g, h


# 错误率定义函数
def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    # 读取训练数据和测试数据
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')

    # 设置参数：
    '''
        max_depth-树深度
        eta-衰减因子：防止过拟合，1为原始模型
        silent-是否输出树的生成情况：1表示不输出
        objective-输出情况：binary是二分类，softmax是多分类。Logistic分类界限为0.5，logitraw的输出值为实数域，分类界限为0。
    '''
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    # eval：evaluate估计数据 || train：训练数据
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 3  # 决策树个数
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    # 自定义损失函数的梯度和二阶导
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)  # [0.15353635 0.84625006 0.15353635 ... 0.95912963 0.02411181 0.95912963]
    print(y)  # [0. 1. 0. ... 1. 0. 1.]
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数：\t', len(y_hat))  # 样本总数：	 1611
    print('错误数目：\t%4d' % error)  # 错误数目：	  10
    print('错误率：\t%.5f%%' % (100 * error_rate))  # 错误率：	0.62073%
