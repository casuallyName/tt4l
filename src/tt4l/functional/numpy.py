# @Time     : 2024/10/9 17:25
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import numpy as np


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
