import operator
from functools import reduce

'''
    二分类多次迭代后准确率可以拉升
'''


def c(n, k):
    return reduce(operator.mul, range(n - k + 1, n + 1)) / reduce(operator.mul, range(1, k + 1))


def bagging(n, p):
    s = 0
    for i in range(int(n / 2 + 1), n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s


if __name__ == "__main__":
    for t in range(10, 101, 10):
        print(t, '次采样正确率：', bagging(t, 0.6))

'''
    10 次采样正确率： 0.6331032576
    20 次采样正确率： 0.7553372033163932
    30 次采样正确率： 0.8246309464931707
    40 次采样正确率： 0.8702342941780972
    50 次采样正确率： 0.9021926358467504
    60 次采样正确率： 0.9253763056485725
    70 次采样正确率： 0.9425655385148007
    80 次采样正确率： 0.9555029441181861
    90 次采样正确率： 0.9653473393248491
    100 次采样正确率： 0.972900802242991
'''