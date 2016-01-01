import sys
import os
import operator
import random
from util import *
from action import *
from copy import *
import numpy as np
from retmath import *

class Policy:

    def __init__(self,actions,horizon=-1):
	self.actions = actions
	self.horizon = horizon

    def getAction(self,state,horizon):
	raise NotImplementedError("Subclass must implement abstract method")

class DeterministicPolicy(Policy):

    def __init__(self,actions,horizon=-1):
	Policy.__init__(self,actions,horizon)
	self.determine = None

    def getAction(self,state,horizon):
	if horizon>=self.horizon:
	    return ActionType.SHOW_RESULTS,\
		    self.actions[ActionType.SHOW_RESULTS]
	return self.determineType, self.determine
 
    def configDetermine(self,a,horizon):
	self.determineType = a
	self.determine = self.actions[a]
	self.horizon = horizon

class RandomPolicy(Policy):

    def __init__(self,actions,horizon=-1):
	Policy.__init__(self,actions,horizon)

    def getAction(self,state,horizon):
	if horizon==self.horizon:
	    return ActionType.SHOW_RESULTS,\
		    self.actions[ActionType.SHOW_RESULTS]
	chosenAction = random.choice(self.actions.keys())
	return chosenAction, self.actions[chosenAction]

class DiscretePolicy(Policy):

    def __init__(self,actions,horizon=-1,epsln=1.0):
	Policy.__init__(self,actions,horizon)
	
	self.epsln = epsln
	self.actions = actions
	self.policyMap = {}
	self.accuReturn = {}
	self.accuTimes = {}

	mapSegments = [float(i)/10 for i in range(0,10)]
	for map in mapSegments:
	    for t in range(self.horizon+1):
		statetup = (map,t)
		for a in self.actions.iterkeys():
		    self.accuReturn[(statetup,a)] = 0.0
		    self.accuTimes[(statetup,a)] = 0.0
		self.policyMap[statetup] = \
			self.actions[ActionType.SHOW_RESULTS]
			#random.choice(self.actions.values())
			#self.actions[ActionType.SHOW_RESULTS]

    def open(self,fname,iteration):
	fin = file(fname)
	lines = fin.readlines()
	for i in range(0,len(lines),61):
	    iter = lines[i].split()[1].split(':')[-1]
	    if not iter==str(iteration):
		continue
	    for j in range(i+1,i+61):
		line = lines[j]
		tmp = line.split('\t')[0].replace('(','').\
			replace(')','').split(',')
		statetup = (float(tmp[0]),int(tmp[1]))
		funcname = line.split('\t')[-1].split()[1]
		#print statetup, funcname
		if(funcname=='feedbackByDoc'):
		    self.policyMap[statetup] = \
			    self.actions[ActionType.FEEDBACK_BY_DOC]
		elif(funcname=='feedbackByKeyterm'):
		    self.policyMap[statetup] = \
			    self.actions[ActionType.FEEDBACK_BY_KEYTERM]
		elif(funcname=='feedbackByRequest'):
		    self.policyMap[statetup] = \
			    self.actions[ActionType.FEEDBACK_BY_REQUEST]
		elif(funcname=='feedbackByTopic'):
		    self.policyMap[statetup] = \
			    self.actions[ActionType.FEEDBACK_BY_TOPIC]
		elif(funcname=='showResults'):
		    self.policyMap[statetup] = \
			    self.actions[ActionType.SHOW_RESULTS]


    def BellmanUpdate(self,statei,statej,actionA,ri):
	key = (statei.totuple(),actionA)
	self.accuReturn[key] += ri + self.maxQ(statej)
	self.accuTimes[key] += 1.0

    def maxQ(self,state):
	maxQ = -99999999
	for a in self.actions.iterkeys():
	    key = (state.totuple(),a)
	    Q = 0
	    if self.accuTimes[key]!=0:
		Q = self.accuReturn[key]/self.accuTimes[key]
	    if Q>maxQ:
		maxQ = Q
	return maxQ

    def getAction(self,state,horizon):
	# finite horizon
	if horizon==self.horizon:
	    return ActionType.SHOW_RESULTS,\
		    self.actions[ActionType.SHOW_RESULTS]
	
	# Epsilon Soft Policy
	soft = random.uniform(0,1)
	if soft>self.epsln:
	    restactions = deepcopy(self.actions.values())
	    restactions.remove(self.policyMap[state.totuple()])
	    chosenFunction = random.choice(restactions)
	    for key,val in self.actions.iteritems():
		if val==chosenFunction:
		    return key, chosenFunction
	else:
	    chosenFunction = self.policyMap[state.totuple()]
	    for key,val in self.actions.iteritems():
		if val==chosenFunction:
		    return key, chosenFunction

    def configPolicyMap(self,policyMap):
	self.policyMap = policyMap

    def updatePolicy(self):
	for statetup in self.policyMap.iterkeys():
	    selectedAction = self.policyMap[statetup]
	    maxValue = -999999
	    for a in self.actions.iterkeys():
		avg = -999999
		if self.accuTimes[(statetup,a)]!=0:
		    avg = self.accuReturn[(statetup,a)]/\
			    self.accuTimes[(statetup,a)]
		if avg>maxValue:
		    maxValue = avg
		    selectedAction = self.actions[a]
	    if maxValue<=0:
		self.policyMap[statetup] = self.actions[ActionType.SHOW_RESULTS]
	    else:
		self.policyMap[statetup] = selectedAction

    def printReturnTable(self,iteration):
	sortReturn = sorted(self.accuReturn.iteritems(),\
		key=operator.itemgetter(0))
	print '------ iteration:'+str(iteration)+' ------\n'
	for statetup,val in sortReturn:
	    if self.accuTimes[statetup]!=0:
		print str(statetup)+'\t'+str(val/self.accuTimes[statetup])
	    else:
		print str(statetup)+'\t0'

    def printPolicyTable(self,fout,iteration):
	print 'Printing policy...'
	sortPolicy = sorted(self.policyMap.iteritems(),\
		key=operator.itemgetter(0))
	fout.write('------ iteration:'+str(iteration)+' ------\n')
	for statetup,action in sortPolicy:
	    fout.write(str(statetup)+'\t'+str(action)+'\n')

class ContinuousPolicy(Policy):

    def __init__(self,actions,numofgauss,var,horizon=-1,epsln=1.0,lamda=0.0):
	Policy.__init__(self,actions,horizon)
	
	self.epsln = epsln
	self.actions = actions
	self.numofgauss = numofgauss
	self.var = var
	self.lamda = lamda

	if self.numofgauss<=5:
	    self.minimumNum = 10
	else:
	    self.minimumNum = 20

	# init basis 
	self.basis = []
	for i in range(self.numofgauss):
	    base = {}
	    base['mean'] = float(i)/float(self.numofgauss)
	    base['var'] = var
	    self.basis.append(base)

	# init parameters
	self.thetas = {}
	self.phis = {}
	self.labels = {}
	#self.matrixMasses = {}
	#self.bellmanMasses = {}
	for t in range(self.horizon+1):
	    for a in self.actions.iterkeys():
		self.thetas[(t,a)] = np.matrix([[0.0] \
			for i in range(self.numofgauss)])	
		self.phis[(t,a)] = [[] for i in range(self.numofgauss)]
		self.labels[(t,a)] = []
		
		# init mass
		#self.matrixMasses[(t,a)] = np.matrix(\
		#	[[0.0 for i in range(5)] for j in range(5)])
		#self.bellmanMasses[(t,a)] = \
		#	np.matrix([[0.0] for i in range(5)])

    def open(self,fname,iteration):
	fin = file(fname)
	lines = fin.readlines()
	for i in range(0,len(lines),30*self.numofgauss+1):
	    iter = lines[i].split()[1].split(':')[-1]
	    if not iter==str(iteration):
		continue
	    for j in range(0,30):
		index = i+1+j*self.numofgauss
		theta = []
		tokens = lines[index].split('\t')
		key = (int(tokens[0].split(',')[0].replace('(','')),\
			int(tokens[0].split(',')[1].replace(')',''))) 

		theta.append([float(tokens[1].replace('[','').replace(']',''))])
		for idx in range(index+1,index+self.numofgauss):
		    theta.append([float(lines[idx].replace('[','').replace(']',''))])
		self.thetas[key] = np.matrix(theta)
		#print theta

    def BellmanUpdate(self,statei,statej,actionA,ri):
	key = (statei.horizon,actionA)
	phiJ = [gaussian(base['mean'],base['var'],statei.eval)\
		for base in self.basis] 
	maxQ,bestaction = self.maxQ(statej)
	self.labels[key].append([ri+maxQ])
	for i in range(len(phiJ)):
	    self.phis[key][i].append(phiJ[i])
	
	#self.matrixMasses[(statei.horizon,actionA)] += phiJ*phiJ.T
	#self.bellmanMasses[(statei.horizon,actionA)] += phiJ.T.dot(\
	#	np.matrix([[ri + maxQ] for i in range(5)]))
    
    def fit(self):
	for t in range(self.horizon+1):
	    for a in self.actions.iterkeys():
		key = (t,a)
		if len(self.labels[key])<10:
		    continue
		phiMat = np.matrix(self.phis[key])
		Y = np.matrix(self.labels[key])
		#print phiMat
		#print Y
		#print phiMat*(phiMat.T)
		del self.thetas[key]
		diag = np.diag([self.lamda for i in range(phiMat.shape[0])])
		self.thetas[key] = (phiMat*(phiMat.T)+diag).I*phiMat*Y
		
		#del self.phis[key]
		#del self.labels[key]
		#self.phis[key] = [[] for i in range(5)]
		#self.labels[key] = []
		#self.thetas[(t,a)] = self.matrixMasses[(t,a)].I*\
		#	self.bellmanMasses[(t,a)]
		print key
		print self.thetas[key]

    def clearData(self):
	for t in range(self.horizon+1):
	    for a in self.actions.iterkeys():
		del self.phis[(t,a)]
		del self.labels[(t,a)] 
		self.phis[(t,a)] = [[] for i in range(self.numofgauss)]
		self.labels[(t,a)] = []
		
		#self.matrixMasses[(t,a)] = np.matrix(\
		#	[[0.0 for i in range(5)] for j in range(5)])
		#self.bellmanMasses[(t,a)] = \
		#	np.matrix([[0.0] for i in range(5)])
    
    def maxQ(self,state):
	maxQ = -99999999
	bestAction = ActionType.SHOW_RESULTS
	for a in self.actions.iterkeys():
	    key = (state.horizon,a)
	    theta = self.thetas[key]
	    phi = np.matrix([[gaussian(base['mean'],base['var'],\
		    state.eval)] for base in self.basis])	
	    Q = (theta.T*phi).tolist()[0][0]
	    if Q>maxQ:
		maxQ = Q
		bestAction = a
	if maxQ<=0:
	    bestAction = ActionType.SHOW_RESULTS
	return maxQ,bestAction

    def getAction(self,state,horizon):
	# finite horizon
	if horizon==self.horizon:
	    return ActionType.SHOW_RESULTS,\
		    self.actions[ActionType.SHOW_RESULTS]
	
	# decide action with max Q
	maxQ,bestaction = self.maxQ(state)

	# Epsilon Soft Policy
	soft = random.uniform(0,1)
	#if self.epsln==1.0:
	#    print state.eval,state.horizon,maxQ, bestaction
	if soft>self.epsln:
	    restactions = deepcopy(self.actions.values())
	    restactions.remove(self.actions[bestaction])
	    chosenFunction = random.choice(restactions)
	    for key,val in self.actions.iteritems():
		if val==chosenFunction:
		    return key, chosenFunction
	else:
	    chosenFunction = self.actions[bestaction]
	    for key,val in self.actions.iteritems():
		if val==chosenFunction:
		    return key, chosenFunction
    
    def printThetaParams(self,fout,iteration):
	print 'Printing policy...'
	sortThetas = sorted(self.thetas.iteritems(),\
		key=operator.itemgetter(0))
	fout.write('------ iteration:'+str(iteration)+' ------\n')
	for statetup,theta in sortThetas:
	    fout.write(str(statetup)+'\t'+str(theta)+'\n')

