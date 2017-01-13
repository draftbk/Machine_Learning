# coding=utf-8
import csv
import random
import math
import numpy as np
from numpy import *
from scipy.linalg import solve
import matplotlib.pyplot as plt

# used to load csv-type data
def loadCsv(filename):
  lines = csv.reader(open(filename, "rb"))
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

# 题中的三角形概率模型
def triangleModel(x,u,a):
    rate=np.zeros(shape=(len(u[:,0]),len(u[0,:])))
    for i in range(0, len(u[:,0])):
        for i1 in range(0,len(u[0,:])):
            if abs(x-u[i,i1])>a[i,i1]:
                rate[i,i1]=0
            else:
                rate[i,i1]=(a[i,i1]-abs(x-u[i,i1]))/math.pow(a[i,i1],2)
                print rate[i,i1]

    return rate

def gauss(a):
    p = np.zeros(shape=(len(a[:, 0]), len(a[0, :])))
    for i in range(0, len(a[:,0])):
        for i1 in range(0, len(a[0, :])):
            p[i,i1]=(1/math.pow(2*pi,1/2))*exp(-math.pow(a[i,i1],2)/2)
    return p

def jifen(temp):
    result=0
    for i in range(0,len(temp[:,0])):
        line=0
        for i1 in range(0,len(temp[0,:])):
            line=line+temp[i,i1]*0.1
        result= result+line*0.1
    return result

def final(x,temp,u,a):
    result=np.zeros(len(x))
    for i in range(0,len(x)):
        n=0
        for i1 in range(0,len(temp[:,0])):
            line=0
            for i2 in range(0,len(temp[0,:])):
                if a[i1,i2]>abs(x[i]-u[i1,i2]):
                   n1=(a[i1,i2]-abs(x[i]-u[i1,i2]))/math.pow(a[i1,i2],2)
                else:
                   n1=0
                n2=temp[i1,i2]
                line=line+n1*n2*0.1
            n=n+line*0.1
        result[i]=n
    return result

def main():
    filename = 'data3_1_w2.csv'
    dataset = loadCsv(filename)
    dataset=np.matrix(dataset)
    # 假设它符合正态分布,3以外就忽略了
    u = np.arange(-3, 3.1, 0.1)
    a = np.arange(0, 3.1, 0.1)
    u,a=np.meshgrid(u,a)
    # 第一个数据处理
    temp=triangleModel(dataset[0,1],u,abs(a))
    print u[0,0]
    # 都乘起来,就是公式51
    for i in range(0, len(dataset)):
        temp = temp*triangleModel(dataset[i, 1], u, abs(a))
    # 因为设它是正态分布的,按公式里的话它应该要
    print temp[25, 25]
    temp = temp * gauss(u)*gauss(a)*2
    temp=temp/jifen(temp)
    x=np.arange(0,1,0.01)
    result=final(x,temp,u,a)
    plt.plot(x,result,'r+')
    plt.show()
main()