# coding=utf-8
import csv
import random
import math
import numpy as np
from numpy import *
from scipy.linalg import solve

# used to load csv-type data
def loadCsv(filename):
  lines = csv.reader(open(filename, "rb"))
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

def parzen(h,x,w):
    n=10
    v=h*h*h
    temp=0
    # print x
    # print w[1,:]
    for i in range(len(w)):
        a=(x-w[i,:])
        b=a.transpose()
        temp=temp+math.exp(-np.dot(a,b) / (2 * h * h))/v
    return temp/n


def main():
    w1 = loadCsv('data4_w1.csv')
    w1=np.matrix(w1)
    w2 = loadCsv('data4_w2.csv')
    w2 = np.matrix(w2)
    w3 = loadCsv('data4_w3.csv')
    w3 = np.matrix(w3)
    x1=np.matrix([0.5,1.0,0.0])
    x2=np.matrix([0.31, 1.51, -0.50])
    x3=np.matrix([-0.3, 0.44, -0.1])

    p=np.zeros((3,3))
    h=0.1
    p[0,0]=parzen(h,x1,w1)
    p[0,1] = parzen(h,x1,w2)
    p[0,2] = parzen(h,x1,w3)
    p[1, 0] = parzen(h, x2, w1)
    p[1, 1] = parzen(h, x2, w2)
    p[1, 2] = parzen(h, x2, w3)
    p[2, 0] = parzen(h, x3, w1)
    p[2, 1] = parzen(h, x3, w2)
    p[2, 2] = parzen(h, x3, w3)
    print p


main()
