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

# define action set
actionset = {}
actionset[ActionType.FEEDBACK_BY_DOC] = feedbackByDoc
actionset[ActionType.FEEDBACK_BY_KEYTERM] = feedbackByKeyterm
actionset[ActionType.FEEDBACK_BY_REQUEST] = feedbackByRequest
actionset[ActionType.FEEDBACK_BY_TOPIC] = feedbackByTopic
actionset[ActionType.SHOW_RESULTS] = showResults

# Simulation
MAPs = [0.0,0.0,0.0]
counts = [0.0,0.0,0.0]

for i in range(len(queries)):
    
    # Simulater User
    simulator = Simulator()
    simulator.setSessionAnswer(answers[i])
    simulator.addAction(ActionType.FEEDBACK_BY_DOC,returnSelectedDoc)
    simulator.addAction(ActionType.FEEDBACK_BY_KEYTERM,returnKeytermYesNo)
    simulator.addAction(ActionType.FEEDBACK_BY_REQUEST,returnSelectedRequest)
    simulator.addAction(ActionType.FEEDBACK_BY_TOPIC,returnSelectedTopic)
    
    # Deterministic Policy 
    dply = DeterministicPolicy(actionset,1)
    dply.configDetermine(ActionType.FEEDBACK_BY_TOPIC,1)
    
    # MDP init
    mdp = MDP(simulator,actionset,dply,i)
    mdp.configIRMDP(queries[i],answers[i],background,inv_index,\
	    doclengs,1000,docmodeldir,10,10,1)
   
    while len(mdp.topicRanking)!=0:
	APs = mdp.simulate()
	print i,APs[1],APs[0],APs[1]-APs[0]
	mdp.reset()

    del mdp
    del simulator
    del dply
    """
    for i in range(len(APs)):
	if APs[i]==0:
	    APs[i] = APs[i-1]
	MAPs[i] += APs[i]
        counts[i] += 1.0
    """
    #print 'Simulating... %d' % i



