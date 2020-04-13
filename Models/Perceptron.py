# -*- encoding: utf-8 -*-
'''
@File    :   perceptron.py
@Time    :   2020/04/11 23:17:44
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

import numpy as np

class PLA:
    def __init__(self):
        self.w_ = None
        self.count_ = 0
        
    def fit(self,X, y, alpha=1, maxCycles=10000):
        self.count = 0
        
        y = np.array(y)
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        
        m, n = X.shape
        w = np.zeros(n)
        j=0
        
        for i in range(maxCycles):
            if y[j] * np.sum(w * X[j]) <= 0:  
                w =  w+ alpha * y[j] * X[j]
                self.count += 1# 更新次数
                j = 0 # 每次更新从头开始
            else:
                j += 1
            if j >= m:
                break
        self.w_ = w
        return w
    
    def predict(self, X):
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        
        y_pre = np.dot(X, self.w_)
        y_pre[y_pre<0] = -1
        y_pre[y_pre>=0] = 1
        return y_pre
        

if __name__ == '__main__':
    pass