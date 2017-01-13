import os
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
import math
from numpy import *
import numpy.linalg as lg
from svmutil import *
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def SVM():
    matfn = 'Data-Ass2.mat'
    data = sio.loadmat(matfn)
    data = data['data']
    data = transpose(data)
    X = data[:, 0:2]  # training samples
    y = data[:, 2]  # training target
    clf = svm.LinearSVC()  # class
    clf.fit(X, y)  # training the svc model
    result = clf.predict([2, 2])  # predict the target of testing samples
    print result  # target
    data1x = []
    data1y = []
    data2x = []
    data2y = []
    a1 = 0
    a2 = 0
    err = 0
    for i in range(len(data[:, 0])):
        result1 = clf.predict(data[i, 0:2])
        if (result1 == 1):
            if (data[i, 2] == -1):
                err = err + 1

            data[i, 2] = 1
            data1x.append(data[i, 0])
            data1y.append(data[i, 1])
            a1 = a1 + 1
        else:
            if (data[i, 2] == 1):
                err = err + 1
            data[i, 2] = -1
            data2x.append(data[i, 0])
            data2y.append(data[i, 1])
            a2 = a2 + 1

    print 'err_number:'
    print err
    print 'err_rate:'
    print 1.0 * err / 3000
    print 'right_rate:'
    print 1 - 1.0 * err / 3000
    plt.plot(data1x, data1y, 'b.')
    plt.plot(data2x, data2y, 'r.')
    plt.show()

def KNN():
    matfn = 'Data-Ass2.mat'
    data = sio.loadmat(matfn)
    data = data['data']
    data = transpose(data)
    X = data[0:11, 0:2]  # training samples
    y = data[0:11, 2]  # training target

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, y)
    print(neigh.predict([-10, 10]))
    data1x = []
    data1y = []
    data2x = []
    data2y = []
    a1 = 0
    a2 = 0
    err = 0
    for i in range(len(data[:, 0])):
        result1 = neigh.predict(data[i, 0:2])
        if (result1 == 1):
            if (data[i, 2] == -1):
                err = err + 1

            data[i, 2] = 1
            data1x.append(data[i, 0])
            data1y.append(data[i, 1])
            a1 = a1 + 1
        else:
            if (data[i, 2] == 1):
                err = err + 1
            data[i, 2] = -1
            data2x.append(data[i, 0])
            data2y.append(data[i, 1])
            a2 = a2 + 1

    print 'err_number:'
    print err
    print 'err_rate:'
    print 1.0 * err / 3000
    print 'right_rate:'
    print 1 - 1.0 * err / 3000
    plt.plot(data1x, data1y, 'b.')
    plt.plot(data2x, data2y, 'r.')
    plt.show()

def main():
    SVM()
    # KNN()

main()