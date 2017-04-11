# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import numpy as np
class HchannelGeneration:
    def __init__(self,BS_num):
        self.BS_num = BS_num
    def channelGeneration(self):
        H=[[0 for column in xrange(3)] for row in xrange(3)]
        for i in xrange(self.BS_num):
            for j in xrange(self.BS_num):
                x = np.random.randn(3, 3)
                H[i][j] = np.matrix(x)
        return H
