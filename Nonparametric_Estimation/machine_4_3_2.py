# coding=utf-8
import csv
import random
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
        dis=np.linalg.norm(x-x1[a,:])

        if dis<tempDistance:
          tempDistance=dis
          A=a
    temp[0,i]=A
    # print temp
    finalDistance=tempDistance
  p=k*1.0/(10*finalDistance*finalDistance)
  if p>100:
    p=1

  return p

def main():
  w2 = loadCsv('data4_w2.csv')
  w2 = np.matrix(w2)
  x = np.arange(-1, 3, 0.1)
  y = np.arange(-1, 4, 0.1)
  lenX=len(x)
  lenY=len(y)
  x,y=np.meshgrid(x,y)
  p1=np.zeros((lenY,lenX))
  p2 = np.zeros((lenY, lenX))
  p3 = np.zeros((lenY, lenX))

  for i in range(lenY):
    for i1 in range(lenX):
      p1[i, i1] = meth_k([x[i,i1],y[i,i1]], w2[:, 0:2], 1)
      p2[i, i1] = meth_k([x[i,i1],y[i,i1]], w2[:, 0:2], 3)
      p3[i, i1] = meth_k([x[i,i1],y[i,i1]], w2[:, 0:2], 5)

  fig = plt.figure()
  ax = Axes3D(fig)
  # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
  ax.plot_surface(x, y, p3, rstride=1, cstride=1, cmap='rainbow')

  plt.show()

main()