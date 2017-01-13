import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from numpy import *
import numpy.linalg as lg
import matplotlib.pyplot as plt


def dis(data1,data2):
    dis = (data2[1] - data1[1]) *(data2[1] - data1[1]) +\
          (data2[0] - data1[0]) *(data2[0] - data1[0])
    dis = sqrt(dis)
    return dis


def classify_print(data,sortI,delta):
    print sortI[1]
    data1x = range(0, 1500)
    data1y = range(0, 1500)
    data2x = range(0, 1500)
    data2y = range(0, 1500)
    a1 = 0
    a2 = 0
    for i in range(len(data[1, :])):
        r=0
        for l in range(len(data[1, :])):
            x = dis(data[0:2, i], data[0:2, l])
            r=r+sortI[l]*exp(-x*x/(2*delta*delta))
            if((1-r)<0.5):
                data1x[a1] = data[0, i]
                data1y[a1] = data[1, i]
                a1 = a1 + 1
            else:
                data[2, i] = 0
                data2x[a2] = data[0, i]
                data2y[a2] = data[1, i]
        # print r
        data[2,i]=r
    # print data
    plt.plot(data1x, data1y, 'b.')
    plt.plot(data2x, data2y, 'r.')
    plt.show()

def main():
    matfn = 'Data-Ass2.mat'
    data = sio.loadmat(matfn)
    data = data['data']
    delta = 1
    f=np.zeros((len(data[1,:]),len(data[1,:])))
    # data1x = range(0, 1000)
    # data1x[0]=1
    # data1x[1] = 1
    for i in range(len(data[1,:])):
        for l in range(len(data[1, :])):
            x=dis(data[0:2,i],data[0:2,l])
            f[i,l]=exp(-x*x/(2*delta*delta))
    sort = data[2,:]
    sortT=transpose(sort)
    sortI = lg.inv(f) * sortT
    print sortI
    classify_print(data,sortI,delta)



main()