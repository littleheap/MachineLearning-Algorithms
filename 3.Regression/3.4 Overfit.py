import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

if __name__ == "__main__":

    N = 9

    # 定义域
    x = np.linspace(0, 6, N) + np.random.randn(N)
    x = np.sort(x)

    # 值域
    y = x ** 2 - 4 * x - 3 + np.random.randn(N)

    x.shape = -1, 1
    y.shape = -1, 1

    # 模型管道处理：取若干阶，再线性回归
    model_1 = Pipeline([
        ('poly', PolynomialFeatures()),  # 具体超参数后面运行时候设置
        ('linear', LinearRegression(fit_intercept=False))])
    model_2 = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-3, 2, 100), fit_intercept=False))])
    model_3 = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-3, 2, 100), fit_intercept=False))])

    models = model_1, model_2, model_3

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(suppress=True)

    plt.figure(figsize=(7, 11), facecolor='w')
    # 阶数池，从1~8
    d_pool = np.arange(1, N, 1)  # 阶：最大为8
    m = d_pool.size

    clrs = []  # 颜色
    for c in np.linspace(16711680, 255, m):
        clrs.append('#%06x' % int(c))

    line_width = np.linspace(5, 2, m)

    titles = u'线性回归', u'Ridge回归', u'Lasso回归'

    for t in range(3):
        # 获取当前模型
        model = models[t]
        # 绘制三行一列图片
        plt.subplot(3, 1, t + 1)
        plt.plot(x, y, 'ro', ms=10, zorder=N)
        for i, d in enumerate(d_pool):
            model.set_params(poly__degree=d)  # 设置阶数这个超参量，poly+__+超参数名称
            model.fit(x, y)
            lin = model.get_params('linear')['linear']
            if t == 0:
                # 线性回归没有alpha
                print(u'线性回归：%d阶，系数为：' % d, lin.coef_.ravel())
            else:
                print(u'岭回归/Lasso：%d阶，alpha=%.6f，系数为：' % (d, lin.alpha_), lin.coef_.ravel())
            x_hat = np.linspace(x.min(), x.max(), num=100)
            x_hat.shape = -1, 1
            y_hat = model.predict(x_hat)
            s = model.score(x, y)
            print(s, '\n')
            zorder = N - 1 if (d == 2) else 0
            plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], label=(u'%d阶，score=%.3f' % (d, s)), zorder=zorder)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.title(titles[t], fontsize=16)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.suptitle(u'多项式曲线拟合', fontsize=18)
    plt.savefig('Overfit.png')  # 存储图片
    plt.show()
