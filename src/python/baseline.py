import os
import sys
import operator
from util import *
from MDP import *
from retrieval import *
from expansion import *
from action import *

lex = readLex(sys.argv[1])
queries = readQuery(sys.argv[2],lex)
background = readBackground(sys.argv[3],lex)
inv_index = readInvIndex(sys.argv[4])
doclengs = readDocLength(sys.argv[5])
answers = readAnswer(sys.argv[6],lex)
docmodeldir = sys.argv[7]
baseline = sys.argv[8]

# define action set
actionset = genActionSet()
costTable = genCostTable()

# Simulation
MAPs = [0.0,0.0,0.0,0.0,0.0,0.0]
AvgReturns = [0.0,0.0,0.0,0.0,0.0,0.0]
counts = [0.0,0.0,0.0,0.0,0.0,0.0]
for i in range(len(queries)):
    
    print 'Processing query %d ...' % (i+1)
    # Simulater User
    simulator = Simulator()
    simulator.setSessionAnswer(answers[i])
    simulator.addAction(ActionType.FEEDBACK_BY_DOC,returnSelectedDoc)
    simulator.addAction(ActionType.FEEDBACK_BY_KEYTERM,returnKeytermYesNo)
    simulator.addAction(ActionType.FEEDBACK_BY_REQUEST,returnSelectedRequest)
    simulator.addAction(ActionType.FEEDBACK_BY_TOPIC,returnSelectedTopic)
    
    # Deterministic Policy 
    dply = DeterministicPolicy(actionset,5)
    if baseline=='DOC':
	dply.configDetermine(ActionType.FEEDBACK_BY_DOC,5)
    elif baseline=="KEYTERM":
	dply.configDetermine(ActionType.FEEDBACK_BY_KEYTERM,5)
    elif baseline=="REQUEST":
	dply.configDetermine(ActionType.FEEDBACK_BY_REQUEST,5)
    elif baseline=="TOPIC":
	dply.configDetermine(ActionType.FEEDBACK_BY_TOPIC,5)
    
    # MDP init
    mdp = MDP(simulator,actionset,dply,i,costTable)
    mdp.configIRMDP(queries[i],answers[i],background,inv_index,\
	    doclengs,1000,docmodeldir,10,10,1)
    APs,Returns = mdp.simulate()
    
    for i in range(len(APs)):
	if APs[i]==0:
	    APs[i] = APs[i-1]
	MAPs[i] += APs[i]
	counts[i] += 1.0
	AvgReturns[i] += Returns[i]

    del mdp
    del simulator
    del dply
    #print 'Simulating... %d' % i

MAPs = [MAPs[i]/counts[i] for i in range(len(MAPs))]
AvgReturns = [AvgReturns[i]/counts[i] for i in range(len(AvgReturns))]

print MAPs
print AvgReturns


