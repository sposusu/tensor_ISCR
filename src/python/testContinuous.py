import os
import sys
import operator
from util import *
from MDP import *
from retrieval import *
from expansion import *
from action import *

lex = readLex(sys.argv[1])
idir = sys.argv[2]

background = readBackground(sys.argv[3],lex)
inv_index = readInvIndex(sys.argv[4])
doclengs = readDocLength(sys.argv[5])
answers = readAnswer(sys.argv[6],lex)
docmodeldir = sys.argv[7]
polDir = sys.argv[8]
wtsDir = sys.argv[9]

numofgauss = int(sys.argv[10])
var = float(sys.argv[11])
lamda = float(sys.argv[12])

iter = int(sys.argv[13])

# define action set
actionset = genActionSet()
costTable = genCostTable()

# Continuous MDP Policy 
dply = ContinuousPolicy(actionset,numofgauss,var,5,1.0,lamda)

# Simulater User
simulator = Simulator()
simulator.addAction(ActionType.FEEDBACK_BY_DOC,returnSelectedDoc)
simulator.addAction(ActionType.FEEDBACK_BY_KEYTERM,\
	returnKeytermYesNo)
simulator.addAction(ActionType.FEEDBACK_BY_REQUEST,\
	returnSelectedRequest)
simulator.addAction(ActionType.FEEDBACK_BY_TOPIC,\
	returnSelectedTopic)


MAP = 0.0
AvgReturn = 0.0
for f in range(1,11):
    test_queries ,test_indexes  = readFoldQueries(idir+'test.fold'+str(f))
    dply.open(polDir+'theta.fold'+str(f)+'.'+str(numofgauss)+'.'+str(var)+\
	    '.'+str(lamda),iter)
    weights = readWeights(wtsDir+'theta.fold'+str(f))
    means = readWeights(wtsDir+'mean.fold'+str(f))
    devs = readWeights(wtsDir+'dev.fold'+str(f))
    
    for j in range(len(test_queries)):
	print 'testing ',test_indexes[j]
	simulator.setSessionAnswer(answers[test_indexes[j]])
	testMDP = MDP(deepcopy(simulator),actionset,dply,\
		test_indexes[j],costTable)
	testMDP.configIRMDP(test_queries[j],\
		answers[test_indexes[j]],background,inv_index,\
		doclengs,1000,docmodeldir,10,10,1,100,500,False,\
		weights,means,devs)
	AP,Return = testMDP.test()
	MAP += AP
	AvgReturn += Return
	del testMDP

MAP/=163.0
AvgReturn/=163.0

print 'iteration:%d MAP:%.4f Return %.2f' \
	%(iter,MAP,AvgReturn)


