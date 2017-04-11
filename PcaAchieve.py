# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import MySQLdb
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
# class PCA:
    # def __init__(self,topN):
    #     self.topN=topN
def pca(dataMat,topN):
    meansValue=np.mean(dataMat,axis=0)
    meanRemove=dataMat-meansValue
    covMat=np.cov(meanRemove,rowvar=0)
    eigVals,eigVectors=np.linalg.eig(np.mat(covMat))
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[:-(topN+1):-1]
    redEigVects=eigVectors[:,eigValInd]
    lowDataMat=meanRemove*redEigVects
    # reconMat=(lowDataMat*redEigVects.T)+meansValue
    return lowDataMat           #, redEigVects, meansValue


if __name__=="__main__":
    try:
        conn=MySQLdb.connect(host='localhost',user='root',passwd='oppo',port=3306,charset='utf8')
        conn.select_db('inteferencealignment')
        cursor=conn.cursor()
        cursor.execute("SELECT * FROM channel")
        results=cursor.fetchall()
        resultMat=np.matrix(results)
        # meansVal = np.mean(resultMat, axis=0)
        # meanRem = resultMat - meansVal
        # covMatTemp = np.cov(meanRem, rowvar=0)
        # eigVal, eigVector = np.linalg.eig(np.mat(covMatTemp))
        # print eigVal
        # eigValIn = sorted(eigVal,reverse=True)
        # eigSum=np.sum(eigValIn)
        # print eigSum
        # x=eigValIn
        # y=np.true_divide(x,eigSum)
        # plt.figure(figsize=(8,6))
        # plt.plot(x,y)
        lowMat = pca(resultMat,3)
        x=list(lowMat[:,0])
        y=list(lowMat[:,1])
        z=list(lowMat[:,2])
        fig=plt.figure(figsize=(8,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(x,y,z)
        # plt.subplot(132)
        # plt.hist(lowMat,30,normed=True)
        # plt.subplot(133)
        # plt.hist(reconMat,30,normed=True)
        plt.show()
    except MySQLdb.Error,e:
        print "Mysql Error %d:%s" % (e.args[0], e.args[1])
    finally:
        cursor.close()
        conn.close()
