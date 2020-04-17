# -*- encoding: utf-8 -*-
'''
@File    :   Kmeans.py
@Time    :   2020/04/17 16:40:13
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''
import numpy as np

class Kmeans(object):
    def __init__(self):
        self.ClusterCenter = None
    def fit(self, data, K, maxIter=50):
        m, n = data.shape
        labels = np.zeros(m)# 用于存储标签
        
        minerr = np.inf
        minCluCen = None
        for iter in range(maxIter):
            self.ClusterCenter = data[np.random.choice(len(data), K, replace=False)]
            # 初始化聚簇中心点

            tmp = np.zeros(self.ClusterCenter.shape)# 更新
            changed = True
            while changed:
                for i in range(m):
                    dist = np.sum((self.ClusterCenter-data[i])**2, axis=1)
                    labels[i] = np.argmin(dist)
                for i in range(len(self.ClusterCenter)):
                    tmp[i] = np.mean(data[labels==i], axis=0)
                if (tmp == self.ClusterCenter).all():
                    changed = False
                else:
                    self.ClusterCenter = tmp
            err = self.Cost(data)
            if err < minerr:
                err = minerr
                minCluCen = self.ClusterCenter
        self.ClusterCenter = minCluCen

    def predict(self, data):
        yfit = np.zeros(len(data),dtype=np.int)
        for i, x in enumerate(data):
            dist = np.sum((self.ClusterCenter-data[i])**2, axis=1)
            yfit[i] = np.argmin(dist)
        return yfit
    
    def Cost(self, data):
        yfit = self.predict(data)
        cost = np.sum((data-self.ClusterCenter[yfit])**2)/len(data)
        return cost