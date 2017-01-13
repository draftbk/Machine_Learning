# coding=utf-8
import csv
import random
import math
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


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
        if abs(x-x1[a])<tempDistance:
          tempDistance=abs(x-x1[a])
          A=a
    temp[0,i]=A
    # print temp
    finalDistance=tempDistance
  p=k*1.0/(10*2*finalDistance)
  if p>100:
    p=1

  return p

def main():
  w3 = loadCsv('data4_w3.csv')
  w3 = np.matrix(w3)
  x = np.arange(-1, 4, 0.0001)
  p=np.zeros((3,len(x)))
  for i in range(len(x)):
    p[0, i] = meth_k(x[i], w3[:, 0], 1)
    p[1, i] = meth_k(x[i], w3[:, 0], 3)
    p[2, i] = meth_k(x[i], w3[:, 0], 5)
  plt.plot(x,p[2,:],'r+')
  plt.show()

main()