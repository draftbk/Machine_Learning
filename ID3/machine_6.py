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
  return dataset

def Entropy(s):
    x=[]
    total=len(s)
    for i in range(len(s[:,0])):
        x.append(s[i,0])
    p=[]
    n=[]
    for i in range(len(s)):
        if s[i,0] in p:
            pass
        else:
            p.append(s[i,0])
            n.append(x.count(s[i,0]))
    s=0
    for i in range(len(p)):
        a=1.0*n[i]/total
        s=s+a*log2(a)
    return -s

def gain(s,a):
    p = []
    n = []
    e=[]
    x=[]
    for i in range(len(s[:,0])):
        if s[i, a] in p:
            pass
        else:
            p.append(s[i, a])
            x=[]
            b=0
            for l in range(len(s[:,0])):
                if(s[i, a]==s[l, a]):
                    x.append(s[l, 0])
                    b=b+1
            n.append(b)
            e.append(Entropy(transpose(matrix(x))))
    c=0
    for i in range(len(p)):
        c=c+e[i]*(1.0*n[i]/len(s[:,0]))

    return (Entropy(s[:,0])-c)


def judge(a):
    if(a[0,2])=='H':
        print '分类为w1'
    if (a[0, 2]) == 'J':
        print '分类为w2'
    if (a[0, 2]) == 'I':
        print 'I,往下'
        if(a[0, 1])=='E':
            print '分类为w1'
        if (a[0, 1]) == 'G':
            print '分类为w1'
        if (a[0, 1]) == 'F':
            print '分类为w2'
        if (a[0, 1]) == 'D':
            print '分类为w2'
    return

def digui(w,c):
    print '...........'
    # print c
    a=0
    x=0
    for i in range(1,6):
        if i in c :
            pass
        else:
            if (gain(w, i) > x):
                x = gain(w, i)
                a = i
    c.append(a)
    # 下面
    s=w
    p = []
    n = []
    e = []
    y=[]
    for i in range(len(s[:,0])):
        if s[i, a] in p:
            pass
        else:
            p.append(s[i, a])
            y=[]
            b=0
            for l in range(len(s[:,0])):
                if(s[i, a]==s[l, a]):
                    y.append(s[l, 0])
                    b=b+1
            n.append(b)
            e.append(Entropy(transpose(matrix(y))))
    print '分支'
    print p
    print '节点对应的Gain'
    print e
    w2=matrix([['1','B','E','I','L','M']])
    for i in range(len(e)):
        if(e[i]>0):
            print p[i]+'向下分支'
            for l in range(len(s[:,0])):
                if(p[i]==s[l, a]):
                   w2=np.vstack((w2,s[l, :]))
            digui(w2,c)

def main():
    w = loadCsv('data6.csv')
    w = np.matrix(w)
    print gain(w,3)
    c = []
    digui(w,c)
    print '........'
    w1=np.matrix([['B','G','I','K','N']])
    print '判断一'
    judge(w1)
    w2 = np.matrix([['C', 'D', 'J', 'L', 'M']])
    print '判断二'
    judge(w2)
main()