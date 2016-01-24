import sys
import os
import operator
import random
from util import *
from copy import *
from retmath import *
import numpy as np

def featureNormalization(features):
    means = [0.0 for x in features[0]]
    twonorms = [0.0 for x in features[0]]
    devs = [0.0 for x in features[0]]
    for f in features:
	for i in range(len(f)):
	    means[i] += f[i]
	    twonorms[i] += math.pow(f[i],2)
    
    for i in range(len(devs)):
	means[i] = means[i]/float(len(features))
	twonorms[i] = twonorms[i]/float(len(features))
	devs[i] = math.sqrt(twonorms[i]-math.pow(means[i],2))
    
    for n in range(len(features)):
	for i in range(1,len(features[n]),1):
	    if devs[i]==0:
		features[n][i] = 1.0
	    else:
		features[n][i] = (features[n][i]-means[i])/devs[i]
    
    return features

def correlation(dim,labels):
    xmean = sum(dim)/float(len(dim))
    ymean = sum(labels)/float(len(labels))
    x2norm = 0.0
    y2norm = 0.0
    xy2norm = 0.0
    for i in range(len(dim)):
	xy2norm += (dim[i]-xmean)*(labels[i]-ymean)
	x2norm += math.pow(dim[i]-xmean,2)
	y2norm += math.pow(labels[i]-ymean,2)
    if x2norm==0.0 or y2norm==0.0:
	return 0.0
    else:
	return xy2norm/math.sqrt(x2norm)/math.sqrt(y2norm)

def featureSelection(trainFs,devFs,testFs,labels,num):
    
    dimensions = np.matrix(trainFs).T.tolist()
    labels = np.matrix(labels).T.tolist()[0]
    
    correlations = {}
    for i in range(1,len(dimensions)):
	dim = dimensions[i]
	correlations[i] = math.fabs(correlation(dim,labels))
    corrRank = sorted(correlations.iteritems(),key=operator.itemgetter(1),reverse=True)
    print corrRank[:10]
    indexes = [corr[0] for corr in corrRank[0:num]]
    indexes.insert(0,0)
    
    #print indexes
   
    select_trainFs = []
    select_devFs = []
    select_testFs = []
    for i in range(len(trainFs)):
	newFeature = [trainFs[i][index] for index in indexes]
	select_trainFs.append(newFeature)
    for i in range(len(devFs)):
	newFeature = [devFs[i][index] for index in indexes]
	select_devFs.append(newFeature)
    for i in range(len(testFs)):
	newFeature = [testFs[i][index] for index in indexes]
	select_testFs.append(newFeature)
    
    return select_trainFs, select_devFs, select_testFs

def featureTransform(features,dim):
    pass
    #for f in range(len(features)):
    #	feature = features[f]
    #	for n in range(len(feature)):

def logistic_regression(features,labels,lamda):
    logicLabels = []
    for l in labels:
	logicLabels.append([logit(l[0])])
    return linear_regression(features,logicLabels,lamda)

def logit(l):
    if l==1:
	return math.log(1)-math.log(0.000000000000001)
    else:
	return math.log(l)-math.log(1-l)

def logistic(p):
    if -1*p>400:
	return 0.0
    elif -1*p<-400:
	return 1.0
    return 1/(1+math.exp(-1*p))
    
def linear_regression(features,labels,lamda):
    
    X = np.matrix(features)
    Y = np.matrix(labels)
    diag = np.diag([lamda for i in range(X.shape[1])])
    theta = (X.T*X+diag).I*(X.T)*Y
    #print theta 
    return theta

def logistic_predict(features,theta):
    Px = linear_predict(features,theta).tolist()
    plst = [[logistic(px[0])] for px in Px]
    return np.matrix(plst)

def linear_predict(features,theta):
    X = np.matrix(features)
    P = X*theta
    return P

def norm1err(P,Y):
    plst = P.tolist()
    ylst = Y
    norm1err = 0.0
    cnt = 0.0
    for i in range(len(plst)):
	cnt += 1.0
	#print plst[i][0], ylst[i][0]
	norm1err += math.fabs(plst[i][0]-ylst[i][0])
    return norm1err/cnt

def norm2err(P,Y):
    plst = P.tolist()
    ylst = Y
    norm2err = 0.0
    for i in range(len(plst)):
	norm2err += math.pow(plst[i][0]-ylst[i][0],2)
    return math.sqrt(norm2err/float(len(plst)))


