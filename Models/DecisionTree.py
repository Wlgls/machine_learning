# -*- encoding: utf-8 -*-
'''
@File    :   DecisionTree.py
@Time    :   2020/04/16 14:53:29
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''

""" 
在构建过程中，我们的父节点保存了分为两个子节点的特征
而两个子节点各自保存了他们分类的属性值
"""

from collections import Counter
import numpy as np

class TreeNode(object):
    def __init__(self, feature=None, value=None, label=None, childs=None,Seg=None, inequal=None):
        self.feature = feature# 该节点选择的特征
        self.value = value# 该节点的分类值， 离散性属性
        self.label = label  # 该节点所代表的标签
        self.childs = childs# 该节点的子节点 # 列表
        
        self.Seg = Seg  # 分割点
        self.inequal = inequal # 回归属性 ['le', 'gt']
    def __str__(self):
        return '{}'.format(self.__dict__)

class DecisionTree_byID3_OR_C45(object):
    def __init__(self, function='ID3'):
        self.T = None
        self.function = function
    
    def ComputeEnt(self, Data):
        # 计算经验熵
        # Data是标签数据
        y = np.unique(Data)
        Ent = 0
        for i in y:
            ratio = np.sum(Data==i) / len(Data)
            Ent += ratio * np.log2(ratio)
        return -Ent

    def SplitData(self, feature, Data):
        # 根据feature分割数据
        retData = []
        unife = np.unique(Data[:, feature])
        for i in unife:
            retData.append(Data[Data[:, feature]==i])
        return unife, retData


    def FindBestFeature(self, Data, Features):
        EEnt = self.ComputeEnt(Data[:, -1])# 经验熵
        
        m, n = Data.shape
        bestfeature = 0
        minEnt = np.inf # 最小的信息增益或者信息增益比

        for i in Features:# 遍历每一个特征
            _, retData = self.SplitData(i, Data)

            PEnt = 0# 计算条件熵
            for item in retData:
                PEnt += len(item)/m * self.ComputeEnt(item[:, -1])
            
            # 计算信息增益
            Gent = EEnt - PEnt

            # 如果是C4.5。那么再求一个信息增益比
            if self.function == 'C4.5':
                Gent = Gent /  self.ComputeEnt(Data[:, i])

            # 比较，求最小的信息增益或者最小的信息增益比
            if minEnt != min(minEnt, Gent):
                minEnt = min(minEnt, Gent)
                bestfeature = i
        return i, minEnt

    def SetLabel(self, Data):
        # 设定数量最多的为标签
        value, count = np.unique(Data[:, -1], return_counts=True)
        return value[np.argmax(count)]
    
    def BuildTree(self, Data, Features, value=None, eta=0):
        
        # 如果所有的y一样或者features为空，则设置为叶节点，不继续分割

        if len(np.unique(Data[:, -1]))==1 or Features == []:
            node = TreeNode(value=value, label=self.SetLabel(Data))
            return node
        
        # 寻找最佳特证， 并将其移除
        bestFeature, minEnt = self.FindBestFeature(Data, Features)

        #  设置eta， 如果信息增益或者信息增益比过小，则直接停止
        if minEnt < eta:
            print(minEnt)
            node = TreeNode(value=value, label=self.SetLabel(Data))
            return node
        
        Features.remove(bestFeature)
        bestLabel = self.SetLabel(Data)
        
        # 用来分类的特征，所代表的值，预测标签。
        Node = TreeNode(feature=bestFeature, value=value, label=bestLabel, childs=[])
        
        if self.T is None:# 如果根节点为确定，则确定根节点
            self.T = Node

        unife, Datai = self.SplitData(bestFeature, Data)# 分开数据，获取数据值和对应的数据集
        for f, d in zip(unife, Datai):
            childnode = self.BuildTree(d, Features, f)# 建立子树
            Node.childs.append(childnode)
        return Node

    def predict(self, X):
        T = self.T
        while True:
            if T.childs is None:
                return T.label
            for child in T.childs:
                if X[T.feature] == child.value:
                    T = child
                    break 

""" 我想不到方法解决多个离散值如何分为两组的方法，而逐一尝试实在过于复杂了
所以CART只用连续了 """

class DecisionTree_CART(object):
    def __init__(self, function='regression'):
        self.T = None
        self.function=function

    """ def Iscontinua(self, data, feature):
        # 判断是否为连续性特征
        # 判断标准为有多少个不同值
        tmp = np.unique(data[:, feature])
        if len(tmp) <= 3:# 如果小于4个就认为是离散的, 太多了需要自行转换吧，否则太麻烦了
            return False
        return True """
        
    def SplitData(self, data, feature, value):
        # 需要注意的是，如果是连续的，我们的value是一个值
        # 但如果是离散的，我们应该是两个列表
        
        ltdata = data[data[:, feature] <= value]
        gtdata = data[data[:, feature] > value]
        return ltdata, gtdata

    def setlabel(self, data):
        if self.function == 'regression':
            return np.mean(data[:, -1])
        else:
            value, count = np.unique(Data[:, -1], return_counts=True)
            return value[np.argmax(count)]

    def LinearErr(self, Data):
        # 平均值方差
        # print(len(Data))
        if len(Data)==0:
            return 0
        else:
            return np.sum((Data[:, -1]-np.mean(Data[:, -1]))**2) / len(Data)

    def Gini(self, Data):
        # 基尼指数
        unif = np.unique(Data[:, -1])
        impurity = 0
        for i in unif:
            impurity += np.sum(Data[:, -1]==i) / len(Data)
        return (1-impurity)/len(Data)

    def FindBestFeature_Point(self, Data):
        # 就像adaboost一样，假设有10个分割点，遍历他们，找到最佳
        numIter = 10
        m, n = Data.shape
        minerr = np.inf
        bestFeature = None
        bestSeg = None
        for i in range(n-1):
            minstep, maxstep = np.min(Data[:,i]), np.max(Data[:, i])
            numsetp = (maxstep-minstep) / numIter
            for j in range(1, numIter):
                value = minstep + numsetp*j
                # 分割数据
                ltdata, gtdata = self.SplitData(Data, i, value)
                # 计算分数
                if self.function == 'regression':
                    err = self.LinearErr(ltdata)+self.LinearErr(gtdata)
                else:
                    err = self.Gini(ltdata)+self.Gini(gtdata)
                #寻找最小分数，作为特征和分割点 
                if err < minerr:
                    minerr = err
                    bestFeature = i
                    bestSeg = value

        return bestFeature, bestSeg, minerr

    
    def BuildTree(self, Data, inequal=None, eta=0.008):
        bestFeature, bestSeg, minerr = self.FindBestFeature_Point(Data)
        if minerr < eta or len(Data)==1:# 设置终止条件
            return TreeNode(inequal=inequal, label=self.setlabel(Data))
        Node = TreeNode(feature=bestFeature, Seg=bestSeg, label=self.setlabel(Data), childs=[])
        if self.T is None:
            self.T = Node
        ltdata, gtdata = self.SplitData(Data, bestFeature, bestSeg)
        Node.childs.append(self.BuildTree(ltdata, inequal='lt'))
        Node.childs.append(self.BuildTree(gtdata, inequal='gt'))
        return Node
    
    def predictx(self, x):
        x = np.array(x)
        T = self.T
        while True:
            if T.childs is None:
                return T.label
            else:
                if x[T.feature] <= T.Seg:
                    T = T.childs[0]
                else:
                    T = T.childs[1]
    def predict(self, Data):
        yfit = np.zeros(len(Data))
        for i, d in enumerate(Data):
            yfit[i] = self.predictx(d)
        return yfit


def test_ID3():
    dataset = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, -1, -1],
                        [-1, 1, -1],
                        [-1, 1, -1]])
    model = DecisionTree_byID3_OR_C45(function='C4.5')
    ent = model.ComputeEnt(dataset[:, -1])
    model.BuildTree(dataset, [0, 1])
    print(model.T)
    y = model.predict([1, -1])
    print(y)

def test_CART():
    Data = np.loadtxt('Data/cart.txt', delimiter='\t')
    print(Data.shape)
    model = DecisionTree_CART()
    model.BuildTree(Data)  
    yfit = model.predict(Data[:, :-1])
    print('平均误差:', np.sum((yfit-Data[:, -1])**2)/len(Data))
if __name__ == '__main__':
    test_CART()
    #test_ID3()