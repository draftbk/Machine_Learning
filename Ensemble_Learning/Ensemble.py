import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from numpy import *
import numpy.linalg as lg
import matplotlib.pyplot as plt
from svmutil import *
from svm import *
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

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
    matfn = 'Data-Ass2.mat'
    data1 = sio.loadmat(matfn)
    data1 = data1['data']
    data1 = transpose(data1)
    X = data1[:, 0:2]  # training samples
    y = data1[:, 2]  # training target
    clf = svm.LinearSVC()  # class
    clf.fit(X, y)  # training the svc model
    X2 = data1[0:11, 0:2]  # training samples
    y2 = data1[0:11, 2]  # training target
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X2, y2)
    data1x=[]
    data1y =[]
    data2x =[]
    data2y =[]
    a1=0
    a2=0
    err=0
    for i in range(len(data[1, :])):
        y1 = activation(w1, data[0, i], data[1, i])
        y2 = activation(w2, data[0, i], data[1, i])
        y3 = activation(w3, data[0, i], data[1, i])
        z = activation2(w4, y1, y2, y3)
        #init vote
        bia=0
        # bp
        if (abs(z - 0) > 0.5):
          bia=bia+0.901
        else:
          bia=bia-0.901
        # knn
        result1 = neigh.predict(data1[i, 0:2])
        if (result1 == 1):
          bia = bia + 0.826
        else:
          bia = bia - 0.826
        # SVM
        result1 = clf.predict(data1[i, 0:2])
        if (result1 == 1):
          bia = bia + 0.903
        else:
          bia = bia - 0.903
        # total
        if (bia > 0):
            if (data[2, i] == 0):
                err = err + 1
            data[2,i]=1
            data1x.append(data[0,i])
            data1y.append(data[1,i])
            a1=a1+1
        else:
            if (data[2, i] == 1):
                err = err + 1
            data[2, i]=0
            data2x.append(data[0, i])
            data2y.append(data[1, i])
            a2 = a2 + 1
    print 'err_number:'
    print err
    print 'err_rate:'
    print 1.0*err/3000
    print 'right_rate:'
    print 1-1.0 * err / 3000
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
    error = 1230
    c=0
    while(error>1000):
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


    printRes(data,w1,w2,w3,w4)



main()