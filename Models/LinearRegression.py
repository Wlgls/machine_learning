# -*- encoding: utf-8 -*-
'''
@File    :   LinearRegression.py
@Time    :   2020/04/12 11:05:30
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''
import numpy as np


class LinearRegression:
    def __init__(self):
        self._w = None
        
    def fit(self, X, y):
        """使用最小二乘法求解
        X: 特征， N*d 不包括x0
        y: 标签
        """
        if len(X.shape) == 1:
            X = np.array(X).reshape((len(X), 1))
        else:
            X = np.array(X)
        y = np.array(y)
        X = np.insert(X, 0, 1, axis=1)
        
        # 最小二乘法就直接最终结果了
        self._w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    
    def predict(self, X):
        """预测数据
        X: 特征
        """
        if len(X.shape) == 1:
            X = np.array(X).reshape((len(X), 1))
        else:
            X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self._w)
