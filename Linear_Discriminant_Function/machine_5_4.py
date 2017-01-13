# coding=utf-8
import csv
import random
import math
import numpy as np
from numpy import *
import numpy.linalg as lg
import matplotlib.pyplot as plt


# used to load csv-type data
def loadCsv(filename):
  lines = csv.reader(open(filename, "rb"))
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

# --correct
def Jacobi(a,w1,w3):

    temp=np.zeros((1,20))
    for i in range(20):
        if i<10:
            temp[0,i]=np.dot(transpose(a), transpose(w1[i, :]))
        else:
            temp[0,i] = np.dot(transpose(a),transpose(w3[i-10,:]))
    return temp

def dich(x):
    result=0
    x=np.matrix(x)
    # print len(x[0,:])
    for i in range(3):
        result=result+x[0,i]*x[0,i]
    result=math.sqrt(result)
    return result

# 梯度
def grad(a,w1,w3):
    j=Jacobi(a,w1,w3)
    tempGrad=np.zeros((1,3))
    for i in range(len(j[0,:])):
        if i<10:
            if j[0,i]<=0:
                tempGrad=tempGrad+np.dot(j[0,i]/dich(w1[i,:]),w1[i,:])
        else:
            if j[0, i] <= 0:
                tempGrad = tempGrad + np.dot(j[0, i] / dich(w3[i-10, :]), w3[i-10, :])
    finalGrad=tempGrad
    return finalGrad

def Hesen(a,w1,w3):
    j = Jacobi(a, w1, w3)
    hesen = np.zeros((1, 3))
    for i in range(len(j[0, :])):
        if i < 10:
            if j[0, i] <= 0:
                hesen = hesen + np.dot(transpose(w1[i, :]), w1[i, :])/dich(w1[i, :])
        else:
            if j[0, i] <= 0:
                hesen = hesen + np.dot(transpose(w3[i - 10, :]), w3[i - 10, :]) / dich(w3[i - 10, :])
    finalHesen = hesen
    return finalHesen

# 距离
def distance(a):
    result=0
    for i in range(3):
        result=result+a[0,i]*a[0,i]
    result=math.sqrt(result)
    return result

#
def meth_tidu(a,w1,w3):
    j=Jacobi(a,w1,w3)
    tempResult=0
    errPoint=0
    for i in range(len(j[0,:])):
        if i<10:
            if j[0,i]<0:
                tempResult=(tempResult+j[0,i]**2/dich(w1[i,:]))/2
                errPoint=errPoint+1
        else:
            if j[0, i] < 0:
                tempResult = (tempResult + j[0, i]**2 / dich(w3[i-10, :]))/2
                errPoint = errPoint + 1
    return tempResult,errPoint

def meth_Newton(a,w1,w3):
    j=Jacobi(a,w1,w3)
    tempResult=0
    errPoint=0
    for i in range(len(j[0,:])):
        if i<10:
            if j[0,i]<0:
                tempResult=(tempResult+j[0,i]**2/dich(w1[i,:]))/2
                errPoint=errPoint+1
        else:
            if j[0, i] < 0:
                tempResult = (tempResult + j[0, i]**2 / dich(w3[i-10, :]))/2
                errPoint = errPoint + 1
    return tempResult,errPoint

#     梯度下降法
def main_tidu():
    w1 = loadCsv('data5_w1.csv')
    w1 = np.matrix(w1)
    w3 = loadCsv('data5_w3.csv')
    w3 = np.matrix(w3)
    n=0.001
    limit=0.002
    for i in range(len(w3[:,0])):
        for i1 in range(3):
            w3[i,i1]=-w3[i,i1]
    a=np.matrix([1,1,1])
    a=transpose(a)
    value=grad(a,w1,w3)
    index=0
    tempGraph=np.zeros((3,5000))
    while distance(value*n)>=limit:
        tempGraph[1,index]=index
        tempGraph[0,index],tempGraph[2,index] =meth_tidu(a,w1,w3)
         # = distance(value*n)
        index=index+1
        a=a-transpose(n*value)
        value=grad(a,w1,w3)
    a1, a2 = meth_tidu(a, w1, w3)
    print '参数:'
    print a
    print '错误点个数:'
    print a2
    plt.plot(tempGraph[1,0:index-1],tempGraph[0,0:index-1],'r.')
    plt.show()



# 牛顿法
def main_Newton():
    w1 = loadCsv('data5_w1.csv')
    w1 = np.matrix(w1)
    w3 = loadCsv('data5_w3.csv')
    w3 = np.matrix(w3)
    # 改
    limit = 1e-56
    for i in range(len(w3[:, 0])):
        for i1 in range(3):
            w3[i, i1] = -w3[i, i1]
    a = np.matrix([1, 1, 1])
    a = transpose(a)
    value = grad(a, w1, w3)
    H=Hesen(a,w1,w3)
    index = 0
    tempGraph = np.zeros((3, 50000))
    while distance(transpose(lg.inv(H)*transpose(value))) >= limit:
        print distance(transpose(lg.inv(H) * transpose(value)))
        tempGraph[1, index] = index
        tempGraph[0, index],tempGraph[2, index] = meth_Newton(a, w1, w3)
         # = distance(value * n)
        index = index + 1
        a = a - (lg.inv(H) * transpose(value))
        value = grad(a, w1, w3)
        H = Hesen(a, w1, w3)
    # 给最后加一个负点,方便看前面的点(不然前面的点在线上,太不明显)
    tempGraph[1, index] = index
    tempGraph[0, index], tempGraph[2, index] = -0.001,0
    a1, a2 = meth_Newton(a, w1, w3)
    print '参数:'
    print a
    print '错误点个数:'
    print a2
    plt.plot(tempGraph[1, 0:index+1], tempGraph[0, 0:index+1], 'r.-')
    plt.show()


# 梯度下降法
# main_tidu()
# 牛顿法
main_Newton()