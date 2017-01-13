# coding=utf-8
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy as np
from numpy import *


def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    # print numbers
    avg = mean(numbers)
    # 算方差
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    # 返回标准差
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def getU1(summaries,num):
    u1 =[]
    for classValue, classSummaries in summaries.iteritems():
        for i in range(num):
            mean, stdev = classSummaries[i]
            u1.append(mean)
        if i==num-1:
            break
    return u1

def getU2(summaries,num):
    u2 =[]
    for classValue, classSummaries in summaries.iteritems():
        for i in range(num):
            if i==0:
                u2=[]
            mean, stdev = classSummaries[i]
            u2.append(mean)
    return u2

def getE1(data):
    E=data[0:10]
    print '........'
    print E
    E = np.transpose(E)
    E = np.cov(E)
    return E

def getE2(data):
    E=data[10:20]
    print '........'
    print E
    E = np.transpose(E)
    E = np.cov(E)
    return E




def getdata(dataset,num):
    data=dataset
    print len(data)
    for i in range(len(data)):
        for x in range(4-num):
            data[i].pop(-1)
    return data



# 判别函数
def getP(E,u,x,d):
    # print '...P....'
    # print E
    # print u
    # print x
    # print d
    E = mat(E)
    x=np.array(x)
    return -(0.5) * np.dot(np.dot(np.transpose(x - u), np.linalg.inv(E)), (x - u)) - 0.5 * d * math.log(2 * math.pi,math.e) - 0.5 * math.log(np.linalg.det(E), math.e) + math.log(0.5, math.e)


def getErr(testSet, predictions):
    err = len(testSet)
    # print testSet
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            err -= 1
    print '错误的点的个数:'
    print err
    return (err / float(len(testSet))) * 100.0


def printBh(u1, u2, E1, E2):
    E1 = mat(E1)
    E2 = mat(E2)
    u2 = np.array(u2)
    u1 = np.array(u1)
    print '------------------bhat------------------'
    k= 0.125*np.dot(np.dot(np.transpose(u2-u1),np.linalg.inv(0.5*(E1+E2))),u2-u1)+0.5*math.log(np.linalg.det(0.5*(E1+E2))/math.sqrt(np.linalg.det(E1)*np.linalg.det(E2)),math.e)
    print 0.5*np.exp(-k)

def getPredictions(u1,u2,data,num):
    predictions = []
    E1 = getE1(data)
    E2 = getE2(data)
    printBh(u1,u2,E1,E2)
    # print 'e....'
    # print E1
    # print E2
    print 'data...'
    print data
    for i in range(len(data)):
        p1=getP(E1,u1,data[i],num)
        p2=getP(E2,u2,data[i],num)
        print p1
        print p2
        if p1>p2:
            predictions.append(1)
        else:
            predictions.append(2)
    return predictions


def start(dataset,num,testset):
    print "--------------这里是分界线-----------------"
    print '-----这里!!!!!!!!!'
    summaries = summarizeByClass(dataset)
    u1= getU1(summaries,num)
    u2= getU2(summaries, num)
    data= getdata(dataset,num)
    predictions=getPredictions(u1,u2,data,num)
    print predictions
    err=getErr(testset, predictions)
    print err



def main():
    filename = 'data2.csv'
    dataset = loadCsv(filename)
    testset=loadCsv(filename)
    start(dataset,1,testset)
    # start(dataset,2,testset)
    # start(dataset,1,testset)
main()