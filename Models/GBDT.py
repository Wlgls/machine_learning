# -*- encoding: utf-8 -*-
'''
@File    :   GBDT.py
@Time    :   2020/04/25 09:20:06
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

from DecisionTree import DecisionTree_CART as DT
import numpy as np
import matplotlib.pyplot as plt

class GBDT(object):
    def __init__(self):
        self.ModelArr = []

    def fit(self, X, y, maxIter=3):
        y = y.reshape((-1, 1))

        ffit = np.zeros(y.shape)
        
        for i in range(maxIter):
            r = y - ffit# 残差

            # 将X和残差拼接，送到决策树算法中
            alg = DT()
            alg.BuildTree(np.concatenate((X, r), axis=1))
            # 在做DecisionTree的时候，有一个阈值，所以我们的算法甚至可能不运行
            if alg.T is None:
                break
            self.ModelArr.append(alg)
            # 更新ffit
            ffit = ffit + alg.predict(X).reshape((-1, 1))

        return self.ModelArr
    
    def predict(self, X):  
        yfit = np.zeros(len(X))
        for model in self.ModelArr:
            yfit += model.predict(X)
        return yfit

    def costErr(self, y, yfit):
        y = y.reshape((-1, 1))
        yfit = yfit.reshape((-1, 1))
        return np.sum((yfit-y)**2)/len(yfit)


def test():
    # y = 2x + 1
    X = np.array([1, 2, 4, 5, 6, 8, 9]).reshape((-1, 1))
    y = 2*X + 1 + np.random.normal(0, 1, 7).reshape((-1, 1))
    model = GBDT()
    model.fit(X, y)
    yfit = model.predict(X)
    print(model.costErr(y, yfit))

    plt.scatter(X, y)
    plt.plot(X, yfit)
    plt.show()


if __name__ == '__main__':
    test()