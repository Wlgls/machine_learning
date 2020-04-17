# -*- encoding: utf-8 -*-
'''
@File    :   SplitData.py
@Time    :   2020/04/12 11:15:27
@Author  :   Wlgls 
@Version :   1.0
@Contact :   smithguazi@gmail.com
'''
import numpy as np

def splitdata(data, ratio_test):

    index = np.random.permutation(len(data))
    test_index = int(np.floor(len(data)*ratio_test))

    data_test = data[index[:test_index]]
    data_train = data[index[test_index:]]
    return data_train, data_test
