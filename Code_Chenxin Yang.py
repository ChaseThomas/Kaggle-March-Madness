import os
import csv
import numpy as np
class Kaggle:
    def loaddata(address,filename):
        os.chdir(address)
        with open(filename) as csvfile:
            file = csv.reader(csvfile)
            data = []
            for row in file:
                data.append(row)
        return data
    def shootacc(data,ID):
        ndata = np.array(data)
        ndata = ndata[1:,[2,3,9,12]]
        ndata = np.array(list(ndata),dtype = "int")
        ndata = ndata[ndata[:,0]==ID,1:]
        ldata = np.array(data)
        ldata = ldata[1:,[4,5,22,25]]
        ldata = np.array(list(ldata),dtype = "int")
        ldata = ldata[ldata[:,0]==ID,1:]
        result = np.concatenate((ndata,ldata),axis = 0)
        return np.mean((result[:,0]-result[:,2])/result[:,1])
    def shaclist(data):
        result = []
        for i in range(len(data)-1):
            result.append(Kaggle.shootacc(data,int(data[i+1][2])))
        for i in range(len(data)-1):
            result.append(Kaggle.shootacc(data,int(data[i+1][4])))
        return np.array(result)
    def feature1(data,shaclist):
        data = np.array(data)
        data = data[1:,[20,33]]
        data = np.array(list(data),dtype = 'int')
        delta = data[:,0]-data[:,1]
        return np.concatenate((delta,-1*delta),axis = 0)*shaclist
    def feature2(data,shaclist):
        data = np.array(data)
        data = data[1:,[17,30]]
        data = np.array(list(data),dtype = 'int')
        delta = data[:,0]-data[:,1]
        return np.concatenate((delta,-1*delta),axis = 0)*shaclist
    def feature3(data,shaclist):
        data = np.array(data)
        data = data[1:,[18,31]]
        data = np.array(list(data),dtype = 'int')
        delta = data[:,0]-data[:,1]
        return np.concatenate((delta,-1*delta),axis = 0)*shaclist
    def fgratio(data,ID):
        data = np.array(data[1:])
        data = data[:,[2,4,8,9,21,22]]
        data = np.array(list(data),dtype = 'int')
        result1 = data[data[:,0]==ID,:]
        result1 = result1[:,[2,3]]
        result2 = data[data[:,1]==ID,:]
        result2 = result2[:,[4,5]]
        result = np.concatenate((result1,result2),axis = 0)
        return 1 - np.mean(result[:,0]/result[:,1])
    def ratiolist1(data):
        result = []
        for i in range(len(data)-1):
            result.append(Kaggle.fgratio(data,int(data[i+1][2])))
        for i in range(len(data)-1):
            result.append(Kaggle.fgratio(data,int(data[i+1][4])))
        return np.array(result)
    def ratiolist2(data):
        result = []
        for i in range(len(data)-1):
            result.append(Kaggle.fgratio(data,int(data[i+1][4])))
        for i in range(len(data)-1):
            result.append(Kaggle.fgratio(data,int(data[i+1][2])))
        return np.array(result)
    def feature4(data,shaclist,ratiolist1):
        data = np.array(data)
        data = data[1:,[14,15,27,28]]
        data = np.array(list(data),dtype = 'int')
        delta1 = data[:,0]-data[:,3]
        delta2 = data[:,2]-data[:,1]
        return np.concatenate((delta1,delta2),axis = 0)*shaclist*ratiolist1
    def feature5(data,shaclist,ratiolist2):
        data = np.array(data)
        data = data[1:,[14,15,27,28]]
        data = np.array(list(data),dtype = 'int')
        delta1 = data[:,2]-data[:,1]
        delta2 = data[:,0]-data[:,3]
        return np.concatenate((delta1,delta2),axis = 0)*shaclist*ratiolist2
    def ftratio(data,ID):
        data = np.array(data)
        data = data[1:,[2,4,12,13,25,26]]
        data = np.array(list(data),dtype = 'int')
        result1 = data[data[:,0]==ID,:]
        result1 = result1[:,[2,3]]
        result2 = data[data[:,1]==ID,:]
        result2 = result2[:,[4,5]]
        result = np.concatenate((result1,result2),axis = 0)
        return np.mean(result[:,0]/result[:,1])
    def ftratlist(data):
        result = []
        for i in range(len(data)-1):
            result.append(Kaggle.ftratio(data,int(data[i+1][2])))
        for i in range(len(data)-1):
            result.append(Kaggle.ftratio(data,int(data[i+1][4])))
        return np.array(result)
    def feature6(data,ftratlist):
        data = np.array(data)
        data = data[1:,[20,33]]
        data = np.array(list(data),dtype = "int")
        result = np.concatenate((data[:,1],data[:,0]),axis = 0)
        return result*ftratlist
    def feature7(data):
        data = np.array(data)
        data = data[1:,6]
        data = list(data)
        for i in range(len(data)):
            if data[i]=='H':
                data[i]=1
            elif data[i]=='A':
                data[i]=-1
            else:
                data[i] = 0
        data = np.array(data)
        result = np.concatenate((data,-data),axis = 0)
        return result
    def feature8():
        return np.ones(1828,dtype = np.int)
    def y(data):
        data = np.array(data)
        data = data[1:,[3,5]]
        data = np.array(list(data),dtype = 'int')
        result = np.concatenate((data[:,0],data[:,1]),axis = 0)
        return result
    def ts(data):
        shaclist = Kaggle.shaclist(data)
        ratiolist1 = Kaggle.ratiolist1(data)
        ratiolist2 = Kaggle.ratiolist2(data)
        ftratlist = Kaggle.ftratlist(data)
        feature1 = Kaggle.feature1(data,shaclist)
        feature2 = Kaggle.feature2(data,shaclist)
        feature3 = Kaggle.feature3(data,shaclist)
        feature4 = Kaggle.feature4(data,shaclist,ratiolist1)
        feature5 = Kaggle.feature5(data,shaclist,ratiolist2)
        feature6 = Kaggle.feature6(data,ftratlist)
        feature7 = Kaggle.feature7(data)
        feature8 = Kaggle.feature8()
        trainningset = np.stack((feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8))
        ts = np.matrix(trainningset)
        return ts
    def theta(data):
        y = Kaggle.y(data)
        ts = Kaggle.ts(data)
        sol = np.linalg.pinv(ts)
        sol = np.transpose(sol)
        theta = sol.dot(y)
        return theta[0]
    def meanstat(data, ID,order):
        data = np.array(data)[1:,:]
        if order == 1:
            index = [20,33]
        elif order == 2:
            index = [17,30]
        elif order == 3:
            index = [18,31]
        elif order == 4:
            index = [14,27]
        elif order == 5:
            index = [15,28]
        data = data[:,[2,4,index[0],index[1]]]
        data = np.array(list(data),dtype = 'int')
        data1 = data[data[:,0] == ID,2]
        data2 = data[data[:,1] == ID,3]
        return np.mean(np.concatenate((data1,data2),axis = 0))
    def predictscore(data, IDa, IDb, location,theta):
        shootacca = Kaggle.shootacc(data,IDa)
        shootaccb = Kaggle.shootacc(data,IDb)
        fgratioa = Kaggle.fgratio(data,IDa)
        fgratiob = Kaggle.fgratio(data,IDb)
        a1 = shootacca*(Kaggle.meanstat(data,IDa,1)-Kaggle.meanstat(data,IDb,1))
        a2 = shootacca*(Kaggle.meanstat(data,IDa,2)-Kaggle.meanstat(data,IDb,2))
        a3 = shootacca*(Kaggle.meanstat(data,IDa,3)-Kaggle.meanstat(data,IDb,3))
        a4 = shootacca*fgratioa*(Kaggle.meanstat(data,IDa,4)-Kaggle.meanstat(data,IDb,5))
        a5 = shootacca*fgratiob*(Kaggle.meanstat(data,IDb,4)-Kaggle.meanstat(data,IDa,5))
        a6 = Kaggle.ftratio(data,IDa)*Kaggle.meanstat(data,IDb,1)
        a7 = location
        a8 = 1

        b1 = shootaccb*(Kaggle.meanstat(data,IDb,1)-Kaggle.meanstat(data,IDa,1))
        b2 = shootaccb*(Kaggle.meanstat(data,IDb,2)-Kaggle.meanstat(data,IDa,2))
        b3 = shootaccb*(Kaggle.meanstat(data,IDb,3)-Kaggle.meanstat(data,IDa,3))
        b4 = shootaccb*fgratiob*(Kaggle.meanstat(data,IDb,4)-Kaggle.meanstat(data,IDa,5))
        b5 = shootaccb*fgratioa*(Kaggle.meanstat(data,IDa,4)-Kaggle.meanstat(data,IDb,5))
        b6 = Kaggle.ftratio(data,IDb)*Kaggle.meanstat(data,IDa,1)
        b7 = -location
        b8 = 1

        A = np.transpose(np.matrix([a1,a2,a3,a4,a5,a6,a7,a8]))
        B = np.transpose(np.matrix([b1,b2,b3,b4,b5,b6,b7,b8]))
        return [int(theta*A),int(theta*B)]
        
    
        
    
        
        
