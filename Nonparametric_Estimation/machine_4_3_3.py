# coding=utf-8
import csv
import random
import math
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# used to load csv-type data
def loadCsv(filename):
  lines = csv.reader(open(filename, "rb"))
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

def meth_k(x,x1,k):
  temp=np.zeros((1,k))
  for b in range(k):
    temp[0,b]=-1
  temp=np.matrix(temp)

  for i in range(k):
    tempDistance=10000
    A=0
    for a in range(len(x1[:,0])):
      isIn=0
      for a1 in range(k):
        if temp[0,a1]==-1:
          break
        else:
          if temp[0,a1]==a:
            isIn=1
      if isIn==0:
        y=x-x1[a,:]
        dis=np.linalg.norm(y)
        if dis<tempDistance:
          tempDistance=dis
          A=a
    temp[0,i]=A
    # print temp
    finalDistance=tempDistance
  p=k*4.0/(3*10*finalDistance*finalDistance*finalDistance)
  
  return p

def main():
  w1 = loadCsv('data4_w1.csv')
  w1=np.matrix(w1)
  w2 = loadCsv('data4_w2.csv')
  w2 = np.matrix(w2)
  w3 = loadCsv('data4_w3.csv')
  w3 = np.matrix(w3)
  p=np.zeros((3,3))
  x1=np.matrix([-0.41,0.82,0.88])
  x2=np.matrix([0.14, 0.72, 4.1])
  x3=np.matrix([-0.81, 0.61, -0.38])
  k=5
  p[0, 0] = meth_k(x1, w1, k)
  p[0, 1] = meth_k(x1, w2, k)
  p[0, 2] = meth_k(x1, w3, k)
  p[1, 0] = meth_k(x2, w1, k)
  p[1, 1] = meth_k(x2, w2, k)
  p[1, 2] = meth_k(x2, w3, k)
  p[2, 0] = meth_k(x3, w1, k)
  p[2, 1] = meth_k(x3, w2, k)
  p[2, 2] = meth_k(x3, w3, k)
  print p

main()