# -*- encoding: utf-8 -*-
'''
@File    :   PCA.py
@Time    :   2020/04/17 15:52:56
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

import numpy as np

class PCA:

    def __init__(self, percentage='int'):
        self.percentage = percentage
        self.cov = None # 协方差矩阵
        self.lamb = None    # 特征值矩阵
        self.C = None   # 特征向量矩阵

    def MeanStand(self, data):
        self.mean = np.mean(data, axis=0)
        data = data - self.mean
        return data

    def fit(self, data):
        m, n = data.shape# m*n

        data = self.MeanStand(data)
        # self.cov = 1 / m * np.dot(data.T, data)
        self.cov = np.cov(data, rowvar=0)

        self.lamb, self.C = np.linalg.eig(self.cov)# 特征值和特征向量
        # 需要注意的是每一列代表一个特征向量
        

        
    def findK(self, percentage):
        temp = np.sort(self.lamb)[:-1]# 降序
        arraysum = np.sum(temp)
        tempsum = 0
        count = 0
        for i in temp:
            tempsum += i
            count+= 1
            if(tempsum >= arraysum*percentage):
                return count

    def DimensionReduct(self, data, K):
        index = np.argsort(self.lamb)[::-1]# 倒叙
        if(self.percentage == 'percentage'):
            K = self.findK(K)
        P = self.C.T[index[:K]]

        return np.dot(self.MeanStand(data), P.T)



if __name__ == '__main__':

    pass