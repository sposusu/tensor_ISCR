import sys
import os
import operator
from util import *
from regression import *

trainX,trainY = readStateFeatures(sys.argv[1])
devX,devY = readStateFeatures(sys.argv[2])
testX,testY = readStateFeatures(sys.argv[3])

trainX = featureNormalization(trainX)
devX = featureNormalization(devX)
testX = featureNormalization(testX)

numFeature = [10,20,50,70,78]
lamdas = [pow(10,i) for i in range(5,4,-1)]

for lamda in lamdas:
    for num in numFeature:
	
	select_trainX, select_devX, select_testX = \
		featureSelection(trainX,devX,testX,trainY,num)
	print len(select_trainX),len(select_trainX[0])
	#select_trainX = trainX
	#select_devX = devX
	#select_testX = testX

	theta = logistic_regression(select_trainX,trainY,lamda)
   
	trainP = logistic_predict(select_trainX,theta)
	devP = logistic_predict(select_devX,theta)
	testP = logistic_predict(select_testX,theta)

	print '-------------------------------------------------------'
	
	"""
	print 'theta:\t'
	for t in theta.tolist():
	    print t[0],
	print
	"""
	
	print '(%d,%d)\ttrain\t n1 err:\t%.8f' % \
		(lamda,num,norm1err(trainP,trainY))	
	print '(%d,%d)\tdev\t n1 err:\t%.8f' % \
		(lamda,num,norm1err(devP,devY))
	print '(%d,%d)\ttest\t n1 err:\t%.8f' % \
		(lamda,num,norm1err(testP,testY))

	del select_trainX
	del select_devX
	del select_testX
	del theta
	del trainP
	del devP
	del testP


