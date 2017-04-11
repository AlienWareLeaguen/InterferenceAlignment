# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import numpy as np

class SmallestDistance:
    def __init__(self,H12,H13,H21,H23,H31,H32,redEigVects,lowDataMat):
        self.H12 = H12
        self.H13 = H13
        self.H21 = H21
        self.H23 = H23
        self.H31 = H31
        self.H32 = H32
        self.redEigVects=redEigVects
        self.lowDataMat=lowDataMat
    def smallestDistance(self):
        H12 = self.H12.reshape((1, 9)) * self.redEigVects
        H13 = self.H13.reshape((1, 9)) * self.redEigVects
        H21 = self.H21.reshape((1, 9)) * self.redEigVects
        H23 = self.H23.reshape((1, 9)) * self.redEigVects
        H31 = self.H31.reshape((1, 9)) * self.redEigVects
        H32 = self.H32.reshape((1, 9)) * self.redEigVects
        Distance=[[0 for column in xrange(100000)]for row in xrange(6)]
        for item in xrange(100000):
            c1 = self.lowDataMat[item, :] - H12
            c2 = self.lowDataMat[item, :] - H13
            c3 = self.lowDataMat[item, :] - H21
            c4 = self.lowDataMat[item, :] - H23
            c5 = self.lowDataMat[item, :] - H31
            c6 = self.lowDataMat[item, :] - H32
            d1 = np.linalg.norm(c1)
            d2 = np.linalg.norm(c2)
            d3 = np.linalg.norm(c3)
            d4 = np.linalg.norm(c4)
            d5 = np.linalg.norm(c5)
            d6 = np.linalg.norm(c6)
            Distance[0][item] = d1
            Distance[1][item] = d2
            Distance[2][item] = d3
            Distance[3][item] = d4
            Distance[4][item] = d5
            Distance[5][item] = d6
        distance1 = []
        distance2 = []
        distance3 = []
        distance4 = []
        distance5 = []
        distance6 = []
        for item in xrange(100000):
            distance1.append(Distance[0][item])
            distance2.append(Distance[1][item])
            distance3.append(Distance[2][item])
            distance4.append(Distance[3][item])
            distance5.append(Distance[4][item])
            distance6.append(Distance[5][item])
        del Distance
        min1 = distance1.index(min(distance1))
        min2 = distance2.index(min(distance2))
        min3 = distance3.index(min(distance3))
        min4 = distance4.index(min(distance4))
        min5 = distance5.index(min(distance5))
        min6 = distance6.index(min(distance6))
        del distance1
        del distance2
        del distance3
        del distance4
        del distance5
        del distance6
        return min1,min2,min3,min4,min5,min6






