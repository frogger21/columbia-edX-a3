import numpy as np
import pandas as pd
import scipy as sp
import sys
import random
import math
from scipy.stats import multivariate_normal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2019 April 7th JChang for ColumbiaX EdX Machine Learning class assignment
# K means and Gaussian Mixture Model (GMM)
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# inputs: X.csv: Each row corresponds to a single vector x(i) transposed
# FOR K-MEANS
# outputs: centroids-i.csv: 5 rows with kth row being the kth centroid

# FOR GMM
# Outputs: i = iteration, k = clusters
# pi-i.csv: 
# mu-i.csv
# sigma-i-k.csv

#input data (this is for columbiaEdX to test their scenarios automatically)
#X = np.genfromtxt(sys.argv[1], delimiter = ",") #input data where each row is x(i) vector transposed. so a matrix of inputs X

#~~~~~~~~~~~~~~~#
#               #    
# FUNCTIONS !   #
#               #
#~~~~~~~~~~~~~~~#

#~~ dot product
def DotProduct(zVec):
    #x^t * x where x is a vector
    temp = 0
    for z in zVec:
        temp += z*z        
    return(temp)
#~~end of DotProduct

#~~ arg min of ||x(i) - mew(k)||^2. Returns the cluster of the smallest of that
def ArgMinC(xVec,mewMatrix):
    smallest = 0+1 #range: [1 to K], this is just an initialization
    diff = xVec - mewMatrix[0,]
    smallestEuclidean = DotProduct(diff)
    #mewMatrix is a k * d matrix where the kth row is the kth cluster
    for k in range(1,mewMatrix.shape[0]):
        diff = xVec - mewMatrix[k,]
        euclidean = DotProduct(diff)
        if euclidean < smallestEuclidean:
            smallestEuclidean = euclidean
            smallest = k+1
        #end of if
    #end of for k  
    return(smallest)
#~~ end of ArgMinC ~~

#~~ Start KMeans ~~
def KMeans(data, nClusters, nIterations):
    #perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    
    columbia = 0 #0 for no, 1 for yes print to csv files on your current folder path
    
    #begin by randomly selecting 5 data points
    nX = data.shape[0] #number of observations of x(i)
    nD = data.shape[1] #number of dimensions in vector of x(i)
    cList = [0]*nX #c[i] where i is in 0 to nD-1 can be [1 to k] where k = nClusters (5 in this case)
    mews = np.zeros((nClusters,nD)) #5*d matrix in this case
    luckyNums = random.sample(list(range(0,nX)),nClusters) #randomly sample 5 numbers from 0 to n without replacement
    
    #initialize mews (each row in this matrix is a mew1 mew2 and so on)
    for z in range(nClusters):
        #get a matrix of the initialize mew
        mews[z,] = data[luckyNums[z],] #row z is the (z+1)th cluster: range [1 to k] where k = nClusters
    #end for z...
    
    #begin coordinate descent algorithm
    for i in range(nIterations):
        #UPDATE the c(i)s
        #each c[i] is the arg min of ||x(i)-mews(k)||^2 w.r.t to k.
        #computationally this is done by brute force checking the which mews[k] gives the smallest euclidean distance
        #recall that ||a-b||^2 = dotproduct of the vector (a-b)    
        for x in range(nX):
            cList[x] = ArgMinC(data[x,],mews) 
        
        #UPDATE the mews(k)s
        nk = [0] * nClusters
        for j in range(nClusters):
            for y in range(nX):
                if cList[y] == (j+1):
                    nk[j] += 1
        
        for j in range(nClusters):
            temp = [0] * nD
            for y in range(nX):
                if cList[y] == (j+1):
                    temp += data[y,]
            #end for y
            temp = [ZZ / nk[j] for ZZ in temp] #divide each element by nk[j]
            mews[j,] = temp
        #end for j
        
        centerslist = mews
        if columbia == 1:
            #FOR OUTPUT (for ColumbiaX EdX to check answers automatically)
            filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
            np.savetxt(filename, centerslist, delimiter=",")
    #end for i....
    
#~~ Done Kmeans ~~

#~~ Multivariate Gaussian distribution ~~
def MVgaussian(x, mean, sigma,D):
    d = D
    constantK = ((2*math.pi)**(-d/2))*((np.linalg.det(sigma))**(-1/2))
    invSigma = np.linalg.inv(sigma)
    reshapeX = np.matrix(x) #1xd
    reshapeMu = np.matrix(mean) #1xd
    Xvec = reshapeX-reshapeMu
    body1 = np.matmul(Xvec,invSigma)
    body2 = np.matmul(body1,Xvec.transpose())
    body3 = np.exp(body2*(-1/2)) 
    body4 = body3*constantK #inputs were vectors, output is singular
    return(body4)
#~~ done MV gaussian

def PosOrNeg():
    if random.random() > 0.5:
        return(1)
    else:
        return(-1)
#returns pos or negative 1
    
def sigmacheck(X,D):
    temp = np.zeros((D,D))
    boolean1 = 0
    for r in range(D):
        for c in range(D):
            if X[r,c] != temp[r,c]:
                boolean1 += 1
    return(boolean1)

#std/mean/range
def statsinfo(X,n):
    stdev = 0
    mean = 0
    maxx = 0
    miny = 0
    x = np.ravel(X)
    sumx = 0
    miny = x[0]
    maxx = x[0]
    for i in range(n):
        sumx += x[i]
        if x[i] < miny:
            miny = x[i]
        if x[i] > maxx:
            maxx = x[i]
    sumx /= n
    mean = sumx
    sumx = 0
    for i in range(n):
        sumx += (x[i] - mean)**(2)
    sumx /= n
    stdev = sumx**(1/2)
    return(stdev,mean,maxx,miny)
    
#~~ START EMGMM ~~    
def EMGMM(data,clusters,iterations,terminatingDifferential):
    #declare some variables
    terminatingDifferentialBoolean = False
    previousL = 0
    nX = data.shape[0]
    nD = data.shape[1]
    k = clusters
    mu = data[np.random.choice(nX,k,replace=False),:] #mu is a k*d matrix
    sigma = [np.eye(nD) for i in range(k)]
    pi = np.ones(k)/k #initialize each pi[k] to be 1/k
    phi = np.zeros((nX,k)) #intialize with 0 n*k matrix
    first = 0
    columbiaX = 0
    printon = 0
    printon2 = 1
    if printon == 1:
        print("Initial SIGMA START")
        for z in range(k):
            print(sigma[z])
        print("Initial mu")    
        print(mu)
        print("The initial pi")
        print(pi)
    #calculate first L
    for ii in range(nX):
        for K in range(k):
            previousL += pi[K]*MVgaussian(data[ii],mu[K],sigma[K],nD)
    for i in range(iterations):
        if printon == 1:
            print ("New ITERATION!: " + str(i+1))
        #
        # check for sigma to be not singular
        #
        temp = np.zeros((nD,nD))
        for K in range(k):
            if sigmacheck(sigma[K],nD) == 0:
                print("IN HERE")
                sigma[K] += np.diag([1e-6]*nD) #add small number to diagonal
        #~~~~~~~~~~~~~~#
        #~~ "E" step ~~#
        #~~~~~~~~~~~~~~#
        for n in range(nX):
            theX = data[n]
            phiDenominator = 0
            for K in range(k):
                #modify sigma a bit with 1e-6 added to diagonal
                sigmaTemp = sigma[K]+np.diag([1e-6]*nD) #this is to ensure no singularity matrix
                temp1 = MVgaussian(theX,mu[K],sigmaTemp,nD)
                temp2 = temp1*pi[K]
                phiDenominator += temp2
                if printon == 1 and K == (k-1) and first == 0 and phiDenominator == 0:
                    #it's a sad sad day that we made it in here
                    temp ="really sad =("
            if phiDenominator == 0 and first == 0:
                first += 1
                temp = "PhiDenom = 0 at iteration: " + str(i+1) + " n: " + str(n+1) + ", k: " + str(K+1)
                print(temp)
            if phiDenominator == 0:
                #print("phidenom = 0 at iteration: " + str(i+1) + ",n: " + str(n+1) + ",k: " + str(k+1))
                for K in range(k):
                    phi[n,K] = pi[K]
            else:
                for K in range(k):
                    sigmaTemp = sigma[K]+np.diag([1e-6]*nD)
                    phi[n,K] = (pi[K] * MVgaussian(theX,mu[K],sigmaTemp,nD))/phiDenominator
        if i >= 0 and printon == 1:
            temp = "phi iteration:" + str(i+1)
            print(temp)
            print(phi)
            print("end phi")
        
        #~~~~~~~~~~~~~~#
        #~~ "M" step ~~#
        #~~~~~~~~~~~~~~#
        #calculate nk
        nk =[0]*k
        for K in range(k):
            for n in range(nX):
                nk[K] += phi[n,K]
        if printon == 1:
            print("NK at iteration: " + str(i+1))
            print(nk)
        #update pi[k]
        for K in range(k):
            pi[K] = nk[K]/nX
        if printon == 1:
            print("iteration pi: " + str(i+1))
            print(pi)    
        #update mu[k]
        for K in range(k):
            temp = np.zeros((1,nD)) #1xd matrix
            for n in range(nX):
                temp += phi[n,K]*data[n]
            temp /= nk[K]
            mu[K] = temp 
            
        #update sigma[k]
    
        for K in range(k):
            temp2 = np.zeros((nD,nD)) #dxd matrix full of zeros
            for n in range(nX):
                demeaned = data[n]-mu[K] #1 x d matrix
                demeaned = np.matrix(demeaned) #1xd matrix
                demeanedT = demeaned.transpose() #d x 1 matrix
                temp2 += np.matmul(demeanedT,demeaned)*phi[n,K] #d x d matrix
            temp2 /= nk[K]
            sigma[K] = temp2
        if printon == 1:
            print("SIGMA ITERATION i: " + str(i+1))
            for z in range(k):
                print("k = " + str(z+1))
                print(sigma[z])
            print("MU at iteration: " + str(i+1))
            print(mu)
        if columbiaX == 1:
            #output for columbiaX edX output checking
            filename = "pi-" + str(i+1) + ".csv" 
            np.savetxt(filename, pi, delimiter=",") 
            ename = "mu-" + str(i+1) + ".csv"
            np.savetxt(ename, mu, delimiter=",")  #this must be done at every iteration
    
            for j in range(k): #k is the number of clusters 
                filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
                np.savetxt(filename, sigma[j], delimiter=",")
        if printon2 == 1:
            L = 0 #the likelihood we are trying to maximize
            for ii in range(nX):
                for K in range(k):
                    theX = data[ii]
                    mew = mu[K]
                    sigmaTemp = sigma[K]+np.diag([1e-6]*nD) #ensure no singularity
                    L += pi[K]*MVgaussian(theX,mew,sigmaTemp,nD)
            statement = "Likelihood at iteration [" + str(i+1) + "] = " + str(L)
            print(statement)
            Diff = L - previousL
            if(Diff < terminatingDifferential):
                terminatingDifferentialBoolean = True
            if(i == (iterations-1) or terminatingDifferentialBoolean == True):
                print("Final Mu (each row is for one cluster) k = " + str(k))
                print(mu)
                print("Final Covariance Matrices")
                print(sigma)
                if terminatingDifferentialBoolean == True:
                    print("We have converged!")
                    break
            previousL = L
    #end for i
    
#~~ END EMGMM ~~

#~~ MAIN PROCEDURE ~~
temp2 = "iris.csv"
Z = np.genfromtxt(temp2, delimiter=',')    
    
KMeans(Z,3,10)
EMGMM(Z,3,100,0.01)
