import os
import sys
import operator
from util import *
from MDP_nonumpy import *
from retrieval import *
from expansion import *
from action import *

lex = readLex(sys.argv[1])
train_queries,train_indexes = readFoldQueries(sys.argv[2])
test_queries ,test_indexes  = readFoldQueries(sys.argv[3])

background = readBackground(sys.argv[4],lex)
inv_index = readInvIndex(sys.argv[5])
doclengs = readDocLength(sys.argv[6])
answers = readAnswer(sys.argv[7],lex)
docmodeldir = sys.argv[8]
fout = file(sys.argv[9],'w')


# define action set
actionset = genActionSet()
costTable = genCostTable()

# Discrete MDP Policy 
dply = DiscretePolicy(actionset,5,0.7)
    
testIterations = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]

# Training
for it in range(100):
    
    for i in range(len(train_queries)):

	numofexamples = len(train_queries)*it+i
	if numofexamples>10000:
	    break
	print 'training iterations :',numofexamples
	
	dply.epsln = 0.2 + float(numofexamples)/10000.0*(0.7-0.2)

	# Simulater User
	simulator = Simulator()
	simulator.setSessionAnswer(answers[train_indexes[i]])
	simulator.addAction(ActionType.FEEDBACK_BY_DOC,returnSelectedDoc)
	simulator.addAction(ActionType.FEEDBACK_BY_KEYTERM,\
		returnKeytermYesNo)
	simulator.addAction(ActionType.FEEDBACK_BY_REQUEST,\
		returnSelectedRequest)
	simulator.addAction(ActionType.FEEDBACK_BY_TOPIC,\
		returnSelectedTopic)

	# MDP training init
	mdp = MDP(deepcopy(simulator),actionset,dply,\
		train_indexes[i],costTable)
	mdp.configIRMDP(train_queries[i],answers[train_indexes[i]],\
		background,inv_index,doclengs,1000,docmodeldir,10,10,1)
	mdp.ValueIteration()
	del mdp

	# update policy and testing
	if numofexamples in testIterations:
	    dply.updatePolicy()
	    dply.printReturnTable(numofexamples)
	    dply.epsln = 1.0
	    MAP = 0.0
	    AvgReturn = 0.0
	    for j in range(len(test_queries)):
		print 'testing ',j
		simulator.setSessionAnswer(answers[test_indexes[j]])
		testMDP = MDP(deepcopy(simulator),actionset,dply,\
			test_indexes[j],costTable)
		testMDP.configIRMDP(test_queries[j],\
			answers[test_indexes[j]],background,inv_index,\
			doclengs,1000,docmodeldir,10,10,1)
		AP,Return = testMDP.test()
		MAP += AP
		AvgReturn += Return
		del testMDP
	    MAP/=float(len(test_queries))
	    AvgReturn/=float(len(test_queries))
	    print 'Iteration %d: MAP:%.4f Return %.2f' \
		    %(numofexamples,MAP,AvgReturn)
	    del MAP
	    del AvgReturn

	# output policy table
	if numofexamples in testIterations:
	    dply.printPolicyTable(fout,numofexamples)
	
	del simulator

fout.close()

