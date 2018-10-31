import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

'''
    LR：就是Logistic回归而不是Linner回归
    莺尾花四个特征：
    花萼长度 || 花萼宽度 || 花瓣长度 || 花瓣宽度 + 类别（3种各50条数据）
'''


def iris_type(s):
    it = {b'Iris-setosa': 0,
          b'Iris-versicolor': 1,
          b'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = u'iris.data'  # 数据文件路径

    '''
    # 手写读取数据
    f = open(path)
    x = []
    y = []
    for d in f:
        # 去空格
        d = d.strip()
        # 如果有数据
        if d:
            # 用逗号分割
            d = d.split(',')
            # 最后一列类别给y
            y.append(d[-1])
            # 前面四列数据给x
            x.append(map(float, d[:-1]))
    print('原始数据X：\n', x)
    print('原始数据Y：\n', y)
    x = np.array(x)
    y = np.array(y)
    print('Numpy格式X：\n', x)
    print('Numpy格式Y-1:\n', y)
    # 用数值替换类别
    y[y == 'Iris-setosa'] = 0
    y[y == 'Iris-versicolor'] = 1
    y[y == 'Iris-virginica'] = 2
    print('Numpy格式Y-2:\n', y)
    y = y.astype(dtype=np.int)
    print('Numpy格式Y-3:\n', y)
    '''

    '''
    # 使用sklearn的数据预处理
    df = pd.read_csv(path, header=0)
    # 所有行都要，列截取到倒数第1列前
    x = df.values[:, :-1]
    # 所有行都要，列只要最后1列
    y = df.values[:, -1]
    print('x = \n', x)
    print('y = \n', y)
    # 用preprocessing预处理类型数据
    le = preprocessing.LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    print(le.classes_)
    y = le.transform(y)
    print('Last Version, y = \n', y)
    '''

    # 路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    print(data)

    # 将数据的0到3列组成x，第4列得到y，4是分割位，anis为1是指：按列分割，水平方向
    x, y = np.split(data, (4,), axis=1)

    # 为了可视化，仅使用前两列特征，行全要，列只要前两列：花萼长度，花萼宽度
    x = x[:, :2]

    print(x)
    print(y)

    # x = StandardScaler().fit_transform(x)
    # lr = LogisticRegression()   # Logistic回归模型
    #     lr.fit(x, y.ravel())        # 根据数据[x,y]，计算回归参数

    # 管道处理：先标准化处理，再喂给Logist回归模型
    lr = Pipeline([('sc', StandardScaler()),
                   ('clf', LogisticRegression())])
    lr.fit(x, y.ravel())  # ravel()将列向量转置为行向量，由于fit函数的要求

    # 画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    # 凑另外两个维度
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(x[:, 0].shape), edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.savefig('Logistic.png')  # 存储图片
    plt.show()

    # 训练集上的预测结果
    y_hat = lr.predict(x)
    y = y.reshape(-1)
    result = y_hat == y
    print(y_hat)
    print(result)
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))
