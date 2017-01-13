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

def mean(numbers):
    a= np.array(numbers[0])
    for i in range(1,len(numbers)):
        a=a+np.array(numbers[i])
    return a / float(len(numbers))

# 算方差
def stdev(numbers):
    u = mean(numbers)
    b=np.matrix(numbers[0])-u
    c=np.matrix(numbers[0])-u
    c = transpose(c)
    a = np.dot(c,b)
    for i in range(1, len(numbers)):
        b = np.matrix(numbers[i])-u
        c = np.matrix(numbers[i])-u
        c = transpose(c)
        a = a+np.dot(c, b)
    # 算方差
    variance = a/ float(len(numbers) )
    return variance


def main(type):
    filename = 'data3_1_w1.csv'
    dataset = loadCsv(filename)
    dataset=np.array(dataset)
    if type==1:
        # 根据需要改成0,1;0,2;1,2
        dataset = np.delete(dataset, [0, 1], axis=1)
    if type==2:
        # 根据需要改成0,1,2
        dataset = np.delete(dataset, [2], axis=1)

    print mean(dataset)
    print stdev(dataset)

def test4():
    filename = 'data3_1_w2.csv'
    dataset = loadCsv(filename)
    dataset=np.array(dataset)
    E= stdev(dataset)
    print np.diag(E)

# def test4():
#     filename = 'data3_1_w2.csv'
#     dataset = loadCsv(filename)
#     dataset=np.array(dataset)
#     u = mean(dataset)
#     print "以下是样本中的均值:"
#     print u
#     a=np.array([[u[0],u[1],0],[u[0],0,u[2]],[0,u[1],u[2]]])
#     b=np.array([1,1,1])
#     x = solve(a, b)
#     print "以下是三个参数:"
#     print(1.0/x)


# main(3)
# main(2)
# main(1)
test4()