# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import numpy as np

class MyFun:
    def myfun(x):
        x = np.mat(x.reshape((3, 3)))
        H_channel = np.load('H.npy')
        HTemp12 = np.mat(H_channel[1][0])
        HTemp13 = np.mat(H_channel[2][0])
        HTemp21 = np.mat(H_channel[0][1])
        HTemp23 = np.mat(H_channel[2][1])
        HTemp31 = np.mat(H_channel[0][2])
        HTemp32 = np.mat(H_channel[1][2])
        P23 = (HTemp23 * np.mat(x[:, 2]))*(((HTemp23 * np.mat(x[:, 2])).T*(HTemp23 * np.mat(x[:, 2]))).I*(HTemp23 * np.mat(x[:, 2])).T)
        P32 = (HTemp32 * np.mat(x[:, 1]))*(((HTemp32 * np.mat(x[:, 1])).T*(HTemp32 * np.mat(x[:, 1]))).I*(HTemp32 * np.mat(x[:, 1])).T)
        P13 = (HTemp13 * np.mat(x[:, 2]))*(((HTemp13 * np.mat(x[:, 2])).T*(HTemp13 * np.mat(x[:, 2]))).I*(HTemp13 * np.mat(x[:, 2])).T)
        P31 = (HTemp31 * np.mat(x[:, 0]))*(((HTemp31 * np.mat(x[:, 0])).T*(HTemp31 * np.mat(x[:, 0]))).I*(HTemp31 * np.mat(x[:, 0])).T)
        P12 = (HTemp12 * np.mat(x[:, 1]))*(((HTemp12 * np.mat(x[:, 1])).T*(HTemp12 * np.mat(x[:, 1]))).I*(HTemp12 * np.mat(x[:, 1])).T)
        P21 = (HTemp21 * np.mat(x[:, 0]))*(((HTemp21 * np.mat(x[:, 0])).T*(HTemp21 * np.mat(x[:, 0]))).I*(HTemp21 * np.mat(x[:, 0])).T)
        return 6-np.trace(P12*P13)-np.trace(P21*P23)-np.trace(P31*P32)
