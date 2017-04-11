# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
from ChannelGeneration import HchannelGeneration
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def fmax(x, a, b):
    return a*(x**b)

if __name__ == '__main__':
    x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    x1 = np.arange(0, 40, 1)
    y1 = np.logspace(0,9,9,base=2)
    y2 = np.logspace(0,8,9,base=2)
    y3 = np.logspace(0,7,9,base=2)
    y4 = np.logspace(0,6,9,base=2)
    fita1, fitb1 = optimize.curve_fit(fmax, x, y1, [1, 1])
    fita2, fitb2 = optimize.curve_fit(fmax, x, y2, [1, 1])
    fita3, fitb3 = optimize.curve_fit(fmax, x, y3, [1, 1])
    fita4, fitb4 = optimize.curve_fit(fmax, x, y4, [1, 1])
    plot1 ,= plt.plot(x1, fmax(x1, fita1[0], fita1[1]), 'r-.')
    plot2 ,= plt.plot(x1, fmax(x1, fita2[0], fita2[1]), 'bs')
    plot3 ,= plt.plot(x1, fmax(x1, fita3[0], fita3[1]), 'g^')
    plot4 ,= plt.plot(x1, fmax(x1, fita4[0], fita4[1]), 'k--')
    plt.legend([plot1, plot2, plot3, plot4],["Ma","fg","sdf","er"],loc=2,borderaxespad=0.)
    plt.title('System SumRate')
    plt.xlabel('SNR_DB')
    plt.ylabel('Sum Rate')
    plt.show()
