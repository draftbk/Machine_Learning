import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from numpy import *
import numpy.linalg as lg
import matplotlib.pyplot as plt


def deltaZ(a,b):
    return (a-b)*b*(1-b)

def deltaY(y,dZ,w):
    return y*(1-y)*dZ*w

def activation(w,a1,a2):

    b=1+exp(-(w[0]*1+w[1]*a1+w[2]*a2))
    b=1/b
    return b

def activation2(w,a1,a2,a3):

    b=1+exp(-(w[0]*a3+w[1]*a1+w[2]*a2))
    b=1/b
    return b


def printRes(data,w1,w2,w3,w4):
    data1x=range(0, 1500)
    data1y =range(0, 1500)
    data2x =range(0, 1500)
    data2y =range(0, 1500)
    a1=0
    a2=0
    for i in range(len(data[1, :])):
        y1 = activation(w1, data[0, i], data[1, i])
        y2 = activation(w2, data[0, i], data[1, i])
        y3 = activation(w3, data[0, i], data[1, i])
        z = activation2(w4, y1, y2, y3)
        # print abs(z-data[2,i])
        if (abs(z - 0) > 0.5):
            data[2,i]=1
            data1x[a1]=data[0,i]
            data1y[a1]=data[1,i]
            a1=a1+1
        else:
            data[2, i]=0
            data2x[a2] = data[0, i]
            data2y[a2] = data[1, i]
            a2 = a2 + 1

    plt.plot(data1x, data1y, 'b.')
    plt.plot(data2x, data2y, 'r.')
    plt.show()
def main():
    matfn = 'Data-Ass2.mat'
    data = sio.loadmat(matfn)
    data = data['data']
    print data[2,1]
    # data1x = range(0, 1000)
    # data1x[0]=1
    # data1x[1] = 1
    for i in range(len(data[1,:])):
        if(data[2,i]==-1):
            data[2,i]=data[2,i]+1
    n = 0.05
    w1 = [0.3, 0.8, 1]
    w2 = [0.6, 0.9, 1]
    w3 = [1.6, 1.2, 1]
    w4 = [0.5, 1.0, 0.3]
    y1,y2,y3,z = 0,0,0,0
    error = 123
    c=0
    while(error<>0):
        error=0
        c=c+1
        if(c%100==0):
            print c
        # print c
        for i in range(len(data[1,:])):
            y1=activation(w1,data[0,i],data[1,i])
            y2 = activation(w2, data[0, i], data[1, i])
            y3 = activation(w3, data[0, i], data[1, i])
            z = activation2(w4, y1, y2, y3)
            d4 = deltaZ(data[2, i], z)
            d1 = deltaY(y1, d4, w4[1])
            d2 = deltaY(y2, d4, w4[2])
            d3 = deltaY(y3, d4, w4[0])
            t = np.array([1, data[0, i], data[1, i]])
            w1 = w1 + n * d1 * t
            # t=np.matrix([1,data[0, i], data[1, i]])
            # w1 = w1 + n * d1 * [1,data[0, i], data[1, i]]
            w2 = w2 + n * d2 * t
            w3 = w3 + n * d3 * t
            w4 = w4 + n * d4 * np.array([y3, y1, y2])
            # if (i == 0):
            #     print d1
                # print n *[1,data[0, i], data[1, i]]


        # figure out the error number -slf
        for i in range(len(data[1, :])):
            y1 = activation(w1, data[0, i], data[1, i])
            y2 = activation(w2, data[0, i], data[1, i])
            y3 = activation(w3, data[0, i], data[1, i])
            z = activation2(w4, y1, y2, y3)
            # print abs(z-data[2,i])
            if(abs(z-data[2,i])>0.5):
                error = error + 1

    print 'w1:'
    print w1
    print 'w2:'
    print w2
    print 'w3:'
    print w3
    print 'w4:'
    print w4
    print 'number of iterations:'
    print c

    printRes(data,w1,w2,w3,w4)



main()