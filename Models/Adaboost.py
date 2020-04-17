# -*- encoding: utf-8 -*-
'''
@File    :   Adaboost.py
@Time    :   2020/04/14 09:39:17
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

import numpy as np

class DecisionStump:

    def __init__(self):
        pass

    def fit(self, X, y, u):
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))

        m, n = X.shape
        numstep = 10 # 因为是数值型数据，所以需要设置分割线，我们可以设置有是个分割线，然后寻找最好的分割线
        bestStump = {}
        bestClass = np.ones((m, 1))# 此次DS的分类结果，用于在Ada中计算全部error
        minError = np.inf

        for i in range(n):
            Stepmin, Stepmax = np.min(X[:, i]), np.max(X[:, i])
            StepSize = (Stepmax-Stepmin)/numstep
            for j in range(1, numstep+1):
                for inequal in ['lt', 'gt']:
                    thresh = Stepmin + j * StepSize

                    yfit = self.predict(X, i, thresh, inequal)
                    err = self.costError(yfit, y, u)

                    if err < minError:
                        minError = err
                        bestClass = yfit.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = thresh
                        bestStump['ineq'] = inequal
 
        return bestStump, bestClass, minError

    def predict(self, X, dimen, thresh, inequal):
        yfit = np.ones((X.shape[0], 1))
        if inequal == 'lt':
            # 小于分割为负
            yfit[X[:, dimen] <= thresh] = -1
        else:
            # 大于分割为正
            yfit[X[:, dimen] > thresh] = -1
        return yfit
    
    def costError(self, yfit, y, u):
        # 将shape统一，用于计算
        yfit = yfit.reshape((len(yfit), 1))
        y = y.reshape((len(y), 1))
        u = np.array(u).reshape((len(u), 1))

        return np.sum((yfit!=y)*u)  # 权重误差 

class Adaboost:
    def __init__(self, Models):
        self.model = Models
        self.WeakClassArr = []

    def sign(self, y):
        y[y<0] = -1
        y[y>=0] = 1
        return y

    def fit(self, X, y, MaxIter=40, eta=0):
        m = X.shape[0]
        y = y.reshape((m, 1))
        u = np.ones((m, 1))/m
        aggclass = np.zeros((m, 1))# 最终合成的分类，用于停止分类器
        for i in range(MaxIter):
            bestStump, bestClass, epsilon = self.model.fit(X, y, u)

            alpha = 0.5 * np.log((1.0-epsilon)/max(epsilon, 1e-16))# 防止为0
            bestStump['alpha'] = alpha
   
            self.WeakClassArr.append(bestStump)
            expon = -1*alpha*y*bestClass# 分类正确为-alpha, 分类错误为+alpha
            u = u*np.exp(expon) / np.sum(u)

            aggclass += alpha*bestClass # 判断最终合并的分类结果

            aggErr = np.sum(self.sign(aggclass)!=y)/m

            if aggErr <= eta:
                break
                
    def predict(self, X):
        m = X.shape[0]
        aggclass = np.zeros((m, 1))

        for weakmodel in self.WeakClassArr:
            classEst = self.model.predict(X, weakmodel['dim'], \
                weakmodel['thresh'], weakmodel['ineq'])
            aggclass += weakmodel['alpha']*classEst

        return self.sign(aggclass)

if __name__ == '__main__':
    data = np.array([[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]])
    y = np.array([1, 1, -1, -1, 1])
    u = np.ones((5, 1))/5
    model = DecisionStump()
    a, b, c = model.fit(data, y, u)
    
    model = Adaboost(model)
    model.fit(data, y, 9)
    print(model.WeakClassArr)