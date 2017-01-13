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

def main():
    w1 = loadCsv('data5_w1.csv')
    w1 = np.array(w1)
    w3 = loadCsv('data5_w3.csv')
    w3 = np.array(w3)
    print w1[:,1]
    plt.plot(transpose(w1[:,1]),transpose(w1[:,2]),'r.')
    plt.plot(transpose(w3[:,1]),transpose(w3[:, 2]), 'b.')
    X = np.arange(-5, 8, 0.01)
    Y =np.arange(-5, 8, 0.01)
    # 梯度下降时
    # b = 0.79988347
    # a=0.46115957
    # c=0.23393494
    b=5.34790375e-62
    a=1.70160574e-62
    c=2.21816463e-62
    for i in range(len(X)):
        Y[i]=(a*X[i]+b)/c
    plt.plot(X, Y, 'g.')
    plt.show()

main()