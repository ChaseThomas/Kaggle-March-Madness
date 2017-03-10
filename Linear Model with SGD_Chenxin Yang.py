import os
import csv
import numpy as np
import pandas as pd
class kg:
    def loaddata(address,filename):
        os.chdir(address)
        with open(filename) as csvfile:
            file = csv.reader(csvfile)
            data = []
            for row in file:
                data.append(row)
        data = np.array(data)
        data = data[1:,[2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
        data = np.array(list(data),dtype = 'int')
        return data
    def cmean(data,ID,col,distance=13):
        alldata = np.concatenate((data[data[:,0]==ID,col],data[data[:,2]==ID,col+distance]))
        return np.mean(alldata)
    def cleandata(data):
        allmean = dict()
        for i in range(np.shape(data)[0]):
            if str(data[i,0]) not in allmean:
                allmean[str(data[i,0])] = [kg.cmean(data,data[i,0],1,2),kg.cmean(data,data[i,0],4)/kg.cmean(data,data[i,0],5),kg.cmean(data,data[i,0],6)/kg.cmean(data,data[i,0],7),kg.cmean(data,data[i,0],8)/kg.cmean(data,data[i,0],9),kg.cmean(data,data[i,0],10),kg.cmean(data,data[i,0],11),kg.cmean(data,data[i,0],13)/kg.cmean(data,data[i,0],14),kg.cmean(data,data[i,0],16)]
        return allmean
    def T_sample(data,cleandata):
        sample = np.random.randint(np.shape(data)[0],size = 250)
        Q = data[sample,0:4]
        A = []
        for i in range(250):
            A.append([Q[i,1]]+[1]+cleandata[str(Q[i,0])]+cleandata[str(Q[i,2])])
        for i in range(250):
            A.append([Q[i,3]]+[1]+cleandata[str(Q[i,2])]+cleandata[str(Q[i,0])])
        return np.array(A)
    def SGD(data,cleandata,iteration,step,regular):
        theta = np.matrix(np.random.rand(17))
        for i in range(iteration):
            T_sample = kg.T_sample(data,cleandata)
            Y = np.matrix(T_sample[:,0])
            X = np.matrix(T_sample[:,1:])
            theta -= step*(2/500*np.dot(theta.dot(np.transpose(X))-Y,X) + 2*regular/500*theta)/(i+1)
        return theta        
    def accuracy(theta,data,cleandata):
        data = data[:,0:4]
        result = data[:,1]>data[:,3]
        testset1 = []
        for i in range(len(data)):
            testset1.append([1] + cleandata[str(data[i,0])] + cleandata[str(data[i,2])])
        testset2 = []
        for i in range(len(data)):
            testset2.append([1] + cleandata[str(data[i,2])] + cleandata[str(data[i,0])])
        testset1 = np.matrix(testset1)
        testset2 = np.matrix(testset2)
        testset1 = testset1.dot(np.transpose(theta))
        testset2 = testset2.dot(np.transpose(theta))
        predict = testset1>testset2
        predict = np.array(predict)
        predict = predict.reshape(len(result))
        A = np.sum(predict==result)/len(predict)
        return A
    
    
        
        

