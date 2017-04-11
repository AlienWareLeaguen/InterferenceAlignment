# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
from ChannelGeneration import HchannelGeneration
import numpy as np
import math
import MySQLdb
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fmin_bfgs
from PcaAchieve import PCA
from SmallestDistance import SmallestDistance

def myfun(x):
    x = np.mat(x.reshape((3, 3)))
    H_channel1 = np.load('reconMat1.npy')
    H_channel2 = np.load('reconMat2.npy')
    H_channel3 = np.load('reconMat3.npy')
    H_channel4 = np.load('reconMat4.npy')
    H_channel5 = np.load('reconMat5.npy')
    H_channel6 = np.load('reconMat6.npy')
    HTemp12 = np.mat(H_channel1.reshape((3, 3)))
    HTemp13 = np.mat(H_channel2.reshape((3, 3)))
    HTemp21 = np.mat(H_channel3.reshape((3, 3)))
    HTemp23 = np.mat(H_channel4.reshape((3, 3)))
    HTemp31 = np.mat(H_channel5.reshape((3, 3)))
    HTemp32 = np.mat(H_channel6.reshape((3, 3)))
    P23 = (HTemp23 * np.mat(x[:, 2])) * (
    ((HTemp23 * np.mat(x[:, 2])).T * (HTemp23 * np.mat(x[:, 2]))).I * (HTemp23 * np.mat(x[:, 2])).T)
    P32 = (HTemp32 * np.mat(x[:, 1])) * (
    ((HTemp32 * np.mat(x[:, 1])).T * (HTemp32 * np.mat(x[:, 1]))).I * (HTemp32 * np.mat(x[:, 1])).T)
    P13 = (HTemp13 * np.mat(x[:, 2])) * (
    ((HTemp13 * np.mat(x[:, 2])).T * (HTemp13 * np.mat(x[:, 2]))).I * (HTemp13 * np.mat(x[:, 2])).T)
    P31 = (HTemp31 * np.mat(x[:, 0])) * (
    ((HTemp31 * np.mat(x[:, 0])).T * (HTemp31 * np.mat(x[:, 0]))).I * (HTemp31 * np.mat(x[:, 0])).T)
    P12 = (HTemp12 * np.mat(x[:, 1])) * (
    ((HTemp12 * np.mat(x[:, 1])).T * (HTemp12 * np.mat(x[:, 1]))).I * (HTemp12 * np.mat(x[:, 1])).T)
    P21 = (HTemp21 * np.mat(x[:, 0])) * (
    ((HTemp21 * np.mat(x[:, 0])).T * (HTemp21 * np.mat(x[:, 0]))).I * (HTemp21 * np.mat(x[:, 0])).T)
    return 6 - np.trace(P12 * P13) - np.trace(P21 * P23) - np.trace(P31 * P32)


def fmax(x, a, b):
    return a*(x**b)

def AlgorithmInit(Bs_num):
    #maxSINR 最大化信干燥比，b1与b2分别为初始化的干扰协方差矩阵
    B1 = [[0 for column in xrange(1)]for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        B1[i][0] = np.zeros((3,3))
    B2 = [[0 for column in xrange(1)]for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        B2[i][0] = np.zeros((3,3))
    v1 = [[0 for column in xrange(1)]for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        v1[i][0] = np.zeros((3,1))
    u1 = [[0 for column in xrange(1)]for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        u1[i][0] = np.zeros((3,1))
    #minWLI 最小泄露比
    Q1 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        Q1[i][0] = np.zeros((3, 3))
    Q2 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        Q2[i][0] = np.zeros((3, 3))
    v2 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        v2[i][0] = np.zeros((3, 1))
    u2 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        u2[i][0] = np.zeros((3, 1))
    #MMSE 最小均方误差估计
    S1 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        S1[i][0] = np.zeros((3, 3))
    S2 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        S2[i][0] = np.zeros((3, 3))
    v3 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        v3[i][0] = np.zeros((3, 1))
    u3 = [[0 for column in xrange(1)] for row in xrange(Bs_num)]
    for i in xrange(Bs_num):
        u3[i][0] = np.zeros((3, 1))
    return B1, B2, Q1, Q2, S1, S2, v1, v2, v3, u1, u2, u3

if __name__ == "__main__":
    conn = MySQLdb.connect(host='localhost', user='root', passwd='oppo', port=3306, charset='utf8')
    conn.select_db('inteferencealignment')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM channel")
    results = cursor.fetchall()
    resultMat = np.matrix(results)
    Pca = PCA(3)
    # 返回低维数据和特征矢量
    lowDataMat, redEigVects, meanValue = Pca.pca(resultMat)
    np.save('lowDataMat.npy', lowDataMat)
    H_Instance = HchannelGeneration(3)
    H = H_Instance.channelGeneration()
    np.save('H.npy', H)
    H12 = H[1][0]
    H13 = H[2][0]
    H21 = H[0][1]
    H23 = H[2][1]
    H31 = H[0][2]
    H32 = H[1][2]
    smallDis=SmallestDistance(H12,H13,H21,H23,H31,H32,redEigVects,lowDataMat)
    min1,min2,min3,min4,min5,min6=smallDis.smallestDistance()
    min1_H = lowDataMat[min1, :]
    min2_H = lowDataMat[min2, :]
    min3_H = lowDataMat[min3, :]
    min4_H = lowDataMat[min4, :]
    min5_H = lowDataMat[min5, :]
    min6_H = lowDataMat[min6, :]
    reconMat1 = (min1_H * redEigVects.T) + meanValue
    np.save('reconMat1.npy',reconMat1)
    reconMat2 = (min2_H * redEigVects.T) + meanValue
    np.save('reconMat2.npy', reconMat2)
    reconMat3 = (min3_H * redEigVects.T) + meanValue
    np.save('reconMat3.npy', reconMat3)
    reconMat4 = (min4_H * redEigVects.T) + meanValue
    np.save('reconMat4.npy', reconMat4)
    reconMat5 = (min5_H * redEigVects.T) + meanValue
    np.save('reconMat5.npy', reconMat5)
    reconMat6 = (min6_H * redEigVects.T) + meanValue
    np.save('reconMat6.npy', reconMat6)
    x0 = np.ones((3, 3))
    fopt = fmin_bfgs(myfun, x0)
    V4=np.mat(fopt.reshape(3,3)).T
    Vm4 = [[0 for i in xrange(3)] for j in xrange(1)]
    Vm4[0][0] = V4[:, 0]
    Vm4[0][1] = V4[:, 1]
    Vm4[0][2] = V4[:, 2]
    U4=[[0 for i in xrange(3)]for j in xrange(1)]
    u1, s1, v1 = np.linalg.svd(reconMat1.reshape((3,3)) * V4[:,1])
    u1 = u1/np.linalg.norm(u1)
    U4[0][0] = np.mat(u1[:,2])
    u2, s2, v2 = np.linalg.svd(reconMat4.reshape((3,3)) * V4[:,2])
    u2 = u2/np.linalg.norm(u2)
    U4[0][1] = np.mat(u2[:,2])
    u3, s3, v3 = np.linalg.svd(reconMat5.reshape((3,3)) * V4[:, 0])
    u3 = u3 / np.linalg.norm(u3)
    U4[0][2] = np.mat(u3[:,2])
    signal_inf1 = 0
    signal_inf2 = 0
    signal_inf3 = 0
    signal_inf4 = 0
    signal_int1 = 0
    signal_int2 = 0
    signal_int3 = 0
    signal_int4 = 0
    SNR_DB=[0, 5, 10, 15, 20, 25, 30, 35, 40]
    B1, B2, Q1, Q2, S1, S2, v1, v2, v3, u1, u2, u3 = AlgorithmInit(3)
    Vm1 = [[0 for column in xrange(3)] for row in xrange(1)]
    Vm2 = [[0 for column in xrange(3)] for row in xrange(1)]
    Vm3 = [[0 for column in xrange(3)] for row in xrange(1)]
    U1 = [[0 for column in xrange(3)] for row in xrange(1)]
    U2 = [[0 for column in xrange(3)] for row in xrange(1)]
    U3 = [[0 for column in xrange(3)] for row in xrange(1)]
    Rmaxsinr = [[0 for column in xrange(3)]for row in xrange(100)]
    Rminwli = [[0 for column in xrange(3)] for row in xrange(100)]
    Rmmse = [[0 for column in xrange(3)] for row in xrange(100)]
    Rbfgs = [[0 for column in xrange(3)] for row in xrange(100)]
    maxSINR_sum = [0 for column in xrange(9)]
    minWLI_sum = [0 for column in xrange(9)]
    mmse_sum = [0 for column in xrange(9)]
    bfgs_sum = [0 for column in xrange(9)]
    Rmaxsinr_sample = [0 for column in xrange(100)]
    Rminwli_sample = [0 for column in xrange(100)]
    Rmmse_sample = [0 for column in xrange(100)]
    Rbfgs_sample = [0 for column in xrange(100)]
    for n in xrange(9):
        powNum = (10**(np.true_divide(SNR_DB[n], 10)))/1000
        for nsample in xrange(100):
            tempmaxSINR = 0
            tempminWLI = 0
            tempMMSE = 0
            tempBFGS = 0
            for i in xrange(3):
                v1[i][0] = np.random.randn(3, 1)
                v1[i][0] = np.matrix(v1[i][0]/np.linalg.norm(v1[i][0]))
                Vm1[0][i] = v1[i][0]
                v2[i][0] = np.random.randn(3, 1)
                v2[i][0] = np.matrix(v2[i][0]/np.linalg.norm(v2[i][0]))
                Vm2[0][i] = v2[i][0]
                v3[i][0] = np.random.randn(3, 1)
                v3[i][0] = np.matrix(v3[i][0]/np.linalg.norm(v3[i][0]))
                Vm3[0][i] = v3[i][0]
            for iteration in xrange(100):
                # 下行迭代计算接收矩阵
                for i in xrange(3):
                    for j in xrange(3):
                        B1[i][0] = B1[i][0]+powNum*H[j][i]*v1[j][0]*v1[j][0].T*H[j][i].T
                        Q1[i][0] = Q1[i][0]+powNum*H[j][i]*v2[j][0]*v2[j][0].T*H[j][i].T
                        S1[i][0] = S1[i][0]+powNum*H[j][i]*v3[j][0]*v1[j][0].T*H[j][i].T
                    B1[i][0] = B1[i][0]-powNum*H[i][i]*v1[i][0]*v1[i][0].T*H[i][i].T+np.eye(3)
                    u1[i][0] = B1[i][0].I*(H[i][i]*v1[i][0])*np.linalg.norm(B1[i][0].I*H[i][i]*v1[i][0])
                    u1[i][0] = u1[i][0]/np.linalg.norm(np.mat(u1[i][0]))
                    U1[0][i] = u1[i][0]
                    Q1[i][0] = Q1[i][0]-powNum*H[i][i]*v2[i][0]*v2[i][0].T*H[i][i].T
                    u,s,v = np.linalg.svd(Q1[i][0])
                    u2[i][0] = v[:,2]
                    u2[i][0] = u2[i][0]/np.linalg.norm(np.mat(u2[i][0]))
                    U2[0][i] = u2[i][0]
                    S1[i][0] = S1[i][0]-powNum*H[i][i]*v3[i][0] * v3[i][0].T * H[i][i].T
                    u3[i][0] = (S1[i][0]+np.eye(3)).I*(powNum*H[i][i]*v3[i][0])
                    u3[i][0] = u3[i][0]/np.linalg.norm(np.mat(u3[i][0]))
                    U3[0][i] = u3[i][0]
                # 上行迭代计算发射矩阵
                for i in xrange(3):
                    for j in xrange(3):
                        B2[i][0] = B2[i][0]+powNum*H[i][j]*u1[j][0]*u1[j][0].T*H[i][j].T
                        Q2[i][0] = Q2[i][0]+powNum*H[i][j]*u2[j][0]*u2[j][0].T*H[i][j].T
                        S2[i][0] = S1[i][0]+powNum*H[i][j]*u3[j][0]*u1[j][0].T*H[i][j].T
                    B2[i][0] = B2[i][0]-powNum*H[i][i].T*u1[i][0]*u1[i][0].T*H[i][i]+np.eye(3)
                    v1[i][0] = B2[i][0].I*(H[i][i].T*u1[i][0])/np.linalg.norm(B2[i][0].I*H[i][i].T*u1[i][0])
                    v1[i][0] = v1[i][0]/np.linalg.norm(v1[i][0])
                    Vm1[0][i] = v1[i][0]
                    Q2[i][0] = Q2[i][0]-powNum*H[i][i].T*u2[i][0]*u2[i][0].T*H[i][i]
                    u,s,v = np.linalg.svd(Q2[i][0])
                    v2[i][0] = v[:,2]
                    v2[i][0] = v2[i][0]/np.linalg.norm(v2[i][0])
                    Vm2[0][i] = v2[i][0]
                    S2[i][0] = S2[i][0]-powNum*H[i][i].T*u3[i][0]*u3[i][0].T * H[i][i]
                    v3[i][0] = (S2[i][0]+np.eye(3)).I*(powNum*H[i][i].T*u3[i][0])
                    v3[i][0] = v3[i][0]/np.linalg.norm(v3[i][0])
                    Vm3[0][i] = v3[i][0]
            #速率函数
            tempmaxSINR = 0
            tempminWLI = 0
            tempMMSE = 0
            tempBFGS = 0
            for i in xrange(3):
                signal_int1 = powNum*np.mat(U1[0][i]).T*np.mat(H[i][i])*np.mat(Vm1[0][i])*np.mat(Vm1[0][i]).T*np.mat(H[i][i]).T*np.mat(U1[0][i])
                signal_int2 = powNum*np.mat(U2[0][i]).T*np.mat(H[i][i])*np.mat(Vm2[0][i])*np.mat(Vm2[0][i]).T * np.mat(H[i][i]).T*np.mat(U2[0][i])
                signal_int3 = powNum*np.mat(U3[0][i]).T*np.mat(H[i][i])*np.mat(Vm3[0][i])*np.mat(Vm3[0][i]).T * np.mat(H[i][i]).T*np.mat(U3[0][i])
                signal_int4 = powNum*np.mat(U4[0][i]).T*np.mat(H[i][i])*np.mat(Vm4[0][i])*np.mat(Vm4[0][i]).T * np.mat(H[i][i]).T * np.mat(U4[0][i])
                for j in xrange(3):
                    signal_inf1 = signal_inf1+powNum*np.mat(U1[0][i]).T*np.mat(H[i][j])*np.mat(Vm1[0][j])*np.mat(Vm1[0][j]).T*np.mat(H[i][j]).T*np.mat(U1[0][i])
                    signal_inf2 = signal_inf2+powNum*np.mat(U2[0][i]).T*np.mat(H[i][j])*np.mat(Vm2[0][j])*np.mat(Vm2[0][j]).T*np.mat(H[i][j]).T*np.mat(U2[0][i])
                    signal_inf3 = signal_inf3+powNum*np.mat(U3[0][i]).T*np.mat(H[i][j])*np.mat(Vm3[0][j])*np.mat(Vm3[0][j]).T*np.mat(H[i][j]).T*np.mat(U3[0][i])
                    signal_inf4 = signal_inf4+powNum*np.mat(U4[0][i]).T*np.mat(H[i][j])*np.mat(Vm4[0][j]) * np.mat(Vm4[0][j]).T*np.mat(H[i][j]).T * np.mat(U4[0][i])
                signal_inf1 = signal_inf1-signal_int1
                signal_inf2 = signal_inf2-signal_int2
                signal_inf3 = signal_inf3-signal_int3
                signal_inf4 = signal_inf4-signal_int4
                Rmaxsinr[nsample][i] = math.log(np.abs(np.linalg.det(np.mat(np.eye(3))+signal_int1/np.mat(U1[0][i]).T*np.mat(np.eye(3))*np.mat(U1[0][i])+signal_inf1)))
                Rminwli[nsample][i] = math.log(np.abs(np.linalg.det(np.mat(np.eye(3))+signal_int2/np.mat(U2[0][i]).T*np.mat(np.eye(3))*np.mat(U2[0][i]) + signal_inf2)))
                Rmmse[nsample][i] = math.log(np.abs(np.linalg.det(np.mat(np.eye(3))+signal_int3/np.mat(U3[0][i]).T *np.mat(np.eye(3))*np.mat(U3[0][i]) + signal_inf3)))
                Rbfgs[nsample][i] = math.log(np.abs(np.linalg.det(np.mat(np.eye(3)) + signal_int4/np.mat(U4[0][i]).T*np.mat(np.eye(3)) * np.mat(U4[0][i]) + signal_inf4)))
                tempmaxSINR = tempmaxSINR+Rmaxsinr[nsample][i]
                tempminWLI=tempminWLI+Rminwli[nsample][i]
                tempMMSE=tempMMSE+Rmmse[nsample][i]
                tempBFGS=tempBFGS+Rbfgs[nsample][i]
                signal_inf1 = 0
                signal_int1 = 0
                signal_inf4=0
                signal_int4=0
                signal_int3 = 0
                signal_inf3 = 0
                signal_int2 = 0
                signal_inf2 = 0
            Rmaxsinr_sample[nsample] = tempmaxSINR
            Rminwli_sample[nsample] = tempminWLI
            Rmmse_sample[nsample] = tempMMSE
            Rbfgs_sample[nsample] = tempBFGS
        maxSINR_sum[n] = np.sum(Rmaxsinr_sample)
        maxSINR_sum[n] = maxSINR_sum[n]/nsample
        minWLI_sum[n] = np.sum(Rminwli_sample)
        minWLI_sum[n] = minWLI_sum[n]/nsample
        mmse_sum[n] = np.sum(Rmmse_sample)
        mmse_sum[n] = mmse_sum[n]/nsample
        bfgs_sum[n] = np.sum(Rbfgs_sample)
        bfgs_sum[n] = bfgs_sum[n]/nsample
    x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    x1 = np.arange(0, 40, 1)
    y1 = np.array(maxSINR_sum)
    y2 = np.array(minWLI_sum)
    y3 = np.array(mmse_sum)
    y4 = np.array(bfgs_sum)
    fita1, fitb1=optimize.curve_fit(fmax, x, y1, [1, 1])
    fita2, fitb2 = optimize.curve_fit(fmax, x, y2, [1, 1])
    fita3, fitb3 = optimize.curve_fit(fmax, x, y3, [1, 1])
    fita4, fitb4 = optimize.curve_fit(fmax, x, y4, [1, 1])
    plot1 ,= plt.plot(x1, fmax(x1, fita1[0], fita1[1]), 'r-.', label='MaxSINR Sum', linewidth=1)
    plot2 ,= plt.plot(x1, fmax(x1, fita2[0], fita2[1]), 'bs', label='MinWLI Sum', linewidth=1)
    plot3 ,= plt.plot(x1, fmax(x1, fita3[0], fita3[1]), 'g^', label='MMSE Sum', linewidth=1)
    plot4 ,= plt.plot(x1, fmax(x1, fita4[0], fita4[1]), 'k--', label='BFGS Sum', linewidth=1)
    plt.legend([plot1, plot2, plot3, plot4], ["MaxSINR", "MinWLI", "MMSE", "BFGS"], loc='upper left')
    plt.title('System SumRate')
    plt.xlabel('SNR_DB')
    plt.ylabel('Sum Rate')
    plt.show()




















