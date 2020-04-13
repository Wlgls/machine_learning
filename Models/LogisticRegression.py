# -*- encoding: utf-8 -*-
'''
@File    :   LogisticRegression.py
@Time    :   2020/04/12 12:05:18
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''


import numpy as np
from GradientDescent import *

class Logistic_Reg:
    def __init__(self, Algorithm='batch'):
        self.w_ = None
        self.Algotithm = Algorithm
    def sigmoid(self, inX):
        a = 1 / (1+np.exp(-inX))
        return a  
    
    def fit(self,X, y, alpha=0.01, maxCycle=1000):
        # 批量梯度下降
        if self.Algotithm == 'batch':
            w = BatchGradientDescent(X,y, alpha, maxCycle)
        if self.Algotithm == 'Stochastic':
            w = StochasticGradDes(X, y, alpha, maxCycle)
        self.w_ = w
    def predict(self, X):
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        
        return self.sigmoid(np.dot(X, self.w_))