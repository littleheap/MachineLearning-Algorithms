# -*- coding:utf-8 -*-

'''
    二分类多次迭代后准确率可以拉升
'''

import operator
from functools import reduce


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
