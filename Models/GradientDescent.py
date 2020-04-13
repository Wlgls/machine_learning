# -*- encoding: utf-8 -*-
'''
@File    :   GradientAscent.py
@Time    :   2020/04/12 12:18:58
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

import numpy as np

def sigmoid(inx):
    return 1 / (1+np.exp(-inx))

def BatchGradientDescent(X, y, alpha=0.01, maxCycles=1000):
    # 逻辑回归的批量梯度下降
    X = np.array(X)
    y = np.array(y)
    X = np.insert(X, 0, 1, axis=1)

    w = np.ones(X.shape[1])
    n = X.shape[0]
    for i in range(maxCycles):
        h = sigmoid(-y*np.dot(X, w.T))
        w = w - alpha*np.sum((h*(-y*X.T)).T, axis=0)
    return w

def StochasticGradDes(X, y, alpha=0.01, maxCycles=200):
    X = np.array(X)
    y = np.array(y)
    X = np.insert(X, 0, 1, axis=1)

    w = np.ones(X.shape[1])
    n = X.shape[0]

    for i in range(maxCycles):
        index = np.random.randint(n)
        h = sigmoid(-y[index]*np.dot(X[index], w))
        w = w - alpha * h *(-y[index]*X[index])
        # 批量的时候，是同时更新，但是在这里，是w仅根据一个样本更改
    return w

if __name__ == '__main__':
    data = np.loadtxt('Data/testSet.txt', delimiter='\t')
    data[data[:, -1]==0][:, -1] = -1
    StochasticGradDes(data[:, :-1], data[:, -1])