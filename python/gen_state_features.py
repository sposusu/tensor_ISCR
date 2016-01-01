import os
import sys
import operator
from util import *
from MDP_nonumpy import *
from retrieval import *
from expansion import *
from action import *

lex = readLex(sys.argv[1])
queries,indexes = readFoldQueries(sys.argv[2])

background = readBackground(sys.argv[3],lex)
inv_index = readInvIndex(sys.argv[4])
doclengs = readDocLength(sys.argv[5])
answers = readAnswer(sys.argv[6],lex)
docmodeldir = sys.argv[7]

fout = file(sys.argv[8],'a')
#fout = sys.stdout

# define action set
actionset = genActionSet()
costTable = genCostTable()

# Discrete MDP Policy 
dply = DiscretePolicy(actionset,5,0.7)
    
# Training
for it in range(100):
    
    for i in range(len(queries)):

	numofexamples = len(queries)*it+i
	print 'Iterations :',numofexamples
	
	dply.epsln = 0.0 

	# Simulater User
	simulator = Simulator()
	simulator.setSessionAnswer(answers[indexes[i]])
	simulator.addAction(ActionType.FEEDBACK_BY_DOC,returnSelectedDoc)
	simulator.addAction(ActionType.FEEDBACK_BY_KEYTERM,\
		returnKeytermYesNo)
	simulator.addAction(ActionType.FEEDBACK_BY_REQUEST,\
		returnSelectedRequest)
	simulator.addAction(ActionType.FEEDBACK_BY_TOPIC,\
		returnSelectedTopic)

	# MDP generate features
	mdp = MDP(deepcopy(simulator),actionset,dply,\
		indexes[i],costTable)
	mdp.configIRMDP(queries[i],answers[indexes[i]],\
		background,inv_index,doclengs,1000,docmodeldir,10,10,1)
	mdp.generateFeatures(fout)
	
	del mdp
	del simulator

fout.close()

