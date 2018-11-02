import numpy as np
import scipy.sparse
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# 读数据函数
def read_data(path):
    y = []
    row = []
    col = []
    values = []
    r = 0  # 首行
    for d in open(path):
        d = d.strip().split()  # 以空格分开
        y.append(int(d[0]))
        d = d[1:]
        for c in d:
            key, value = c.split(':')
            row.append(r)  # 添加行
            col.append(int(key))  # 添加列
            values.append(float(value))  # 添加Value
        r += 1
    # 稀疏矩阵，只存1的地方就行 || 稠密矩阵，0 1在矩阵中全部显示
    x = scipy.sparse.csr_matrix((values, (row, col))).toarray()
    y = np.array(y)
    return x, y


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)


if __name__ == '__main__':
    x, y = read_data('agaricus_train.txt')
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    # Logistic回归
    lr = LogisticRegression(penalty='l2')  # L2正则
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic回归 ')

    # XGBoost
    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost ')
    # XGBoost 正确率：	 0.9992325402916347
