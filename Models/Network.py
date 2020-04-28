# -*- encoding: utf-8 -*-
'''
@File    :   Network.py
@Time    :   2020/04/27 19:53:00
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

"""
在实现时 y \in {0, 1}
激活函数是 sigmoid
只是一个简单的3*3*1的网络，所以，不会想着过分的复杂化
而是尽量简单，帮助理解
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.datasets import make_classification

class Network_331(object):
    # 先写一个3×3×1的吧，之后再扩展
    def __init__(self, L=3, n_inputs=3, n_outputs=1, n_hiddens=3 ):        
        self.w1 = 2*np.random.normal(size=(n_inputs+1, n_hiddens))-1    # 4*3
        self.w2 = 2*np.random.normal(size=(n_hiddens+1, n_outputs))-1 # 4*1
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def d_sigmoid(self, x):
        tmp = self.sigmoid(x)
        return tmp*(1-tmp)

    def forward(self, data):
        # 隐藏层
        X = np.insert(data, 0, 1, axis=1)   # 输入 加上偏置
        s1 = np.dot(X, self.w1)  # 加权求和 m*3
        z1 = self.sigmoid(s1) # 激活 m*3

        # 输出层
        xout = np.insert(z1, 0, 1, axis=1)  # 输入 加上偏置
        sout = np.dot(xout, self.w2)  # 加权求和 m*1
        zout = self.sigmoid(sout)   # 激活 m*1
        return X, s1, xout, sout, zout

    def backpropbyvec(self, data, label):
        x1, s1, xout, sout, zout = self.forward(data)
        # s1不包含偏置，且没有激活，xout是对s1进行激活后取偏置的响亮
        s1 = np.insert(s1, 0, 1, axis=1)# 求个偏置    # m*4
       
        # 反向传播
        # 代价函数为: ln(-yi ln(\theta(s))-(1-yi)ln(1-\theta(s)))
        # partj =  yi-\theta(s) = yi - zout# m*1
        delta2 = label - zout   # 输出层的偏差 # m*1
        #deltah1 = delta2 * self.w2[1][0] * d_sigmoid(s1[:, 1])  # m*1
        #deltah2 = delta2 * self.w2[2][0] * d_sigmoid(s1[:, 2])  # m*1
        #deltah3 = delta2 * self.w2[3][0] * d_sigmoid(s1[:, 3])  # m*1

        # self.w2 -> 4*1  =>> m*1 × 1*4 => m*4 => 第一列 没用，因为偏置没有delta 
        deltah = np.dot(delta2, self.w2.T)*self.d_sigmoid(s1)# m*4

        # 4*m * 1*m => 4*1
        w2_grad = np.dot(xout.T, delta2)/len(x1) # 4*1

        # 隐藏层  4*m × m*3 ==> 4*3
        w1_grad = np.dot(x1.T, deltah[:, 1:])/len(data)   # 4*3

        return w2_grad.reshape((-1, 1)), w1_grad   #4*1   4*3 
    
    def backprop(self, data, label):
        # 随机梯度下降，每次仅使用一个
        x, s1, xout, sout, zout = self.forward(data)
        w2_grad = np.zeros(self.w2.shape) # 4*1
        w1_grad = np.zeros(self.w1.shape) # 4*3
        for i in range(len(data)):
            x1i = x[i, :].reshape((1, -1)) # 1×4
            s1i = s1[i, :].reshape((1, -1)) # 1*3
            xouti = xout[i, :].reshape((1, -1))# 1*4

            zout1 = zout[i, :].reshape((1, -1))# 1*1
            yi = label[i, :].reshape((1, -1))# 1*1
            souti = sout[i, :].reshape((1, -1))
            s1i = np.insert(s1i, 0, 1, axis=1) # 1*4
            delta2 = self.sigmoid(-yi*souti)*(-yi).reshape((-1, 1)) # 1*1

            delta1 = np.dot(self.w2, delta2)*self.d_sigmoid(s1i).T#
            # 4*1

            w2_grad += np.dot(xouti.T, delta2)    # 1*4
            w1_grad += np.dot(x1i.T, delta1[1:].T)

        w2_grad = w2_grad/len(data)
        w1_grad = w1_grad/len(data)
        return w2_grad, w1_grad

    def fit(self, data, label, eta=0.02, maxIter=1000):
        
        for i in range(maxIter):
            w2_grad, w1_grad = self.backpropbyvec(data, label)

            self.w2 -= eta * w2_grad
            self.w1 -= eta * w1_grad

    def predict(self, data):
        _, _, _,_, yfit = self.forward(data)
        return yfit

def make_data():
    data = np.array([0, 1])
    x = []
    label = []
    for i in itertools.product(data, data, data):
        x.append(i)
        if(sum(i)%2 == 0):
            label.append(1)
        else:
            label.append(0)
    return np.array(x), np.array(label).reshape((-1, 1))

def test():
    data, label = make_data()
    print(label)
    model = Network_331()
    model.fit(data, label)
    yfit = model.predict(data)
    print(yfit)
    # print(yfit)
    yfit[yfit>=0.5] = 1
    yfit[yfit<0.5] = 0
    print(np.sum(label!=yfit))
    print(yfit)

def main():
    data,target= make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0,n_repeated=0, n_classes=2, n_clusters_per_class=1)
    print(data.shape, target.shape)
    target = target.reshape((-1, 1))
    #target[target==0] = -1
    model = Network_331()
    model.fit(data, target, maxIter=10000)
    yfit = model.predict(data)
    # print(yfit)
    yfit[yfit>=0.5] = 1
    yfit[yfit<0.5] = 0
    # print(np.concatenate((target, yfit), axis=1))
    print(np.sum(target!=yfit))
if __name__ == '__main__':
    # test()
    main()