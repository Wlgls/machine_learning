# -*- encoding: utf-8 -*-
'''
@File    :   LogisticRegression.py
@Time    :   2020/04/12 12:05:18
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

# y属于 {-1, 1}

import numpy as np

class Logistic_Reg:
    def __init__(self, Algorithm='batch'):
        self.w_ = None
        self.Algotithm = Algorithm

    def sigmoid(self, inX):
        a = 1 / (1+np.exp(-inX))
        return a  
    
    def BatchGradientDescent(self, X, y, alpha=0.01, maxCycles=1000):
        # 逻辑回归的批量梯度下降
        w = np.ones((X.shape[1]))
        for i in range(maxCycles):
            h = self.sigmoid(-y*np.dot(X, w.T))
            w = w - alpha*np.sum((h*(-y*X.T)).T, axis=0)
        return w

    def StochasticGradDes(self, X, y, alpha=0.01, maxCycles=200):
        # 逻辑回归随机梯度下降
        w = np.ones((X.shape[1]))
        n = X.shape[0]
        for i in range(maxCycles):
            index = np.random.randint(n)
            h = self.sigmoid(-y[index]*np.dot(X[index], w))
            w = w - alpha * h *(-y[index]*X[index])
            # 批量的时候，是同时更新，但是在这里，是w仅根据一个样本更改
        return w

    def fit(self,X, y, alpha=0.01, maxCycle=1000):
        # 批量梯度下降
        X = np.array(X)
        y = np.array(y)
        X = np.insert(X, 0, 1, axis=1)

        if self.Algotithm == 'batch':
            w = self.BatchGradientDescent(X,y, alpha, maxCycle)
        if self.Algotithm == 'Stochastic':
            w = self.StochasticGradDes(X, y, alpha, maxCycle)
        self.w_ = w

    def predict(self, X):
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        
        return self.sigmoid(np.dot(X, self.w_))