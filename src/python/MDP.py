import os
import sys
import operator
from util import *
from action import *
from retrieval import *
from action import *
from simulator import *
from policy import *
from state import *

class MDP:

    def __init__(self,sim,actionSet,policy,index,costTable):
	self.simulator = sim
	self.actions = actionSet
	self.policy = policy
	self.index = index
	self.costTable = costTable

    """
    def __del__(self):	
	del self.simulator
	del self.actions
	del self.policy
	del self.index
	del self.costTable
	del self.query
	del self.answer
	del self.back
	del self.inv_index
	del self.doclengs
	del self.alpha_d
	del self.docmodeldir
	del self.iteration
	del self.mu
	del self.delta
	del self.positive_docs
	del self.positive_leng
	del self.negative_docs
	del self.negative_leng
	del self.posprior
	del self.negprior
	del self.keytermlst
	del self.self.topicRanking
	del self.topiclst
	del self.requestlst
	del self.statequeue
	del self.topicnumwords
	del self.topicleng
    """

    def configIRMDP(self,query,ans,back,inv_index,doclengs,alpha_d,\
	    docmodeldir,em_iter,mu,delta,topicleng=100,topicnumword=500,\
	    oracle=True,weights=[],means=[],devs=[]):
	self.query = query
	self.answer = deepcopy(ans)
	self.back = back
	self.inv_index = inv_index
	self.doclengs = doclengs
	self.alpha_d = alpha_d
	self.docmodeldir = docmodeldir
	self.iteration = em_iter
	self.mu = mu
	self.delta = delta

	# feedback models
	self.positive_docs = []
	self.positive_leng = []
	self.negative_docs = []
	self.negative_leng = []
	self.posprior = deepcopy(query)
	self.negprior = {}
	pieces = docmodeldir.split('/')
	lastindex = len(pieces)-2
	cpsID = pieces[lastindex-1]+'.'+pieces[lastindex]
	self.keytermlst = readKeytermlist(cpsID,query)
	self.requestlst = readRequestlist(cpsID,ans)
	self.topiclst = readTopicWords(cpsID)
	self.topicRanking = readTopicList(cpsID,self.index)[:5]
	self.statequeue = []

	self.topicleng = topicleng
	self.topicnumwords = topicnumword

	# state estimation
	self.oracle = oracle
	self.weights = weights
	self.means = means
	self.devs = devs

    def genStateQueue(self,reserveModel=False):
	# Generate Firstpass
	firstpass = retrieve(self.query,self.back,self.inv_index,\
		self.doclengs,self.alpha_d)
	if self.oracle:
	    self.statequeue.append(State(firstpass,self.answer,\
		    ActionType.ACTION_NONE,0,self.docmodeldir,\
		    self.doclengs,self.back,self.iteration,self.mu,\
		    self.delta,self.inv_index,self.alpha_d))
	else:	
	    if isinstance(self.policy,DiscretePolicy): # discrete
		self.statequeue.append(DiscreteState(firstpass,\
			self.answer,ActionType.ACTION_NONE,0,\
			self.docmodeldir,self.doclengs,self.back,\
			self.iteration,self.mu,self.delta,\
			self.inv_index,self.alpha_d,\
			self.statequeue,self.weights))
		self.statequeue[-1].setModels(self.posprior,self.negprior,\
			self.posprior,self.negprior)
		self.statequeue[-1].estimate(self.statequeue,\
			self.weights,self.means,self.devs)
	    
	    else: # continuous
		self.statequeue.append(ContinuousState(firstpass,\
			self.answer,ActionType.ACTION_NONE,0,\
			self.docmodeldir,self.doclengs,self.back,\
			self.iteration,self.mu,self.delta,\
			self.inv_index,self.alpha_d,\
			self.statequeue,self.weights))
		self.statequeue[-1].setModels(self.posprior,self.negprior,\
			self.posprior,self.negprior)
		self.statequeue[-1].estimate(self.statequeue,\
			self.weights,self.means,self.devs)
	   
	self.curtHorizon= 0
	actionType, action = \
		self.policy.getAction(self.statequeue[-1],self.curtHorizon)
	while actionType!=ActionType.SHOW_RESULTS:
	    #print 'Simulation ..',self.statequeue[-1].ap
	    curtState = self.statequeue[-1]
	    posmodel, negmodel = action(curtState.ret,self.answer,\
		    self.simulator,self.posprior,self.negprior,\
		    self.positive_docs,self.negative_docs,\
		    self.positive_leng,self.negative_leng,\
		    self.doclengs,self.back,self.iteration,
		    self.mu,self.delta,self.docmodeldir,\
		    self.keytermlst,self.requestlst,self.topiclst,\
		    self.topicRanking,self.topicleng,self.topicnumwords)
	    ret = retrieveCombination(posmodel,negmodel,self.back,\
		    self.inv_index,self.doclengs,self.alpha_d,0.1)
	    self.curtHorizon += 1
	    if self.oracle:
		self.statequeue.append(State(ret,self.answer,actionType,\
		    self.curtHorizon,self.docmodeldir,self.doclengs,\
		    self.back,self.iteration,self.mu,self.delta,\
		    self.inv_index,self.alpha_d))
	    else:
		if isinstance(self.policy,DiscretePolicy): # discrete
		    self.statequeue.append(DiscreteState(ret,\
			    self.answer,actionType,self.curtHorizon,\
			    self.docmodeldir,self.doclengs,self.back,\
			    self.iteration,self.mu,self.delta,\
			    self.inv_index,self.alpha_d,\
			    self.statequeue,self.weights))
		    self.statequeue[-1].setModels(posmodel,negmodel,\
			    self.posprior,self.negprior)
		    self.statequeue[-1].estimate(self.statequeue,\
			    self.weights,self.means,self.devs)
		else: # continuous
		    self.statequeue.append(ContinuousState(ret,\
			    self.answer,actionType,self.curtHorizon,\
			    self.docmodeldir,self.doclengs,self.back,\
			    self.iteration,self.mu,self.delta,\
			    self.inv_index,self.alpha_d,\
			    self.statequeue,self.weights))
		    self.statequeue[-1].setModels(posmodel,negmodel,\
			    self.posprior,self.negprior)
		    self.statequeue[-1].estimate(self.statequeue,\
			    self.weights,self.means,self.devs)
	    actionType, action = \
		    self.policy.getAction(self.statequeue[-1],self.curtHorizon)

    def ValueIteration(self):
	
	self.genStateQueue()

	# Policy update
	for i in range(len(self.statequeue)-1):
	    statei = self.statequeue[i]
	    statej = self.statequeue[i+1]
	    actionA = statej.actionType
	    ri = 0.0
	    ri = self.costTable[actionA] + self.costTable['lambda'] *\
		    (self.statequeue[i+1].ap-self.statequeue[i].ap)
	    self.policy.BellmanUpdate(statei,statej,actionA,ri)
    
    def FittedValueIteration(self):
	
	self.genStateQueue()

	# Policy update
	for i in range(len(self.statequeue)-1):
	    statei = self.statequeue[i]
	    statej = self.statequeue[i+1]
	    actionA = statej.actionType
	    ri = self.costTable[actionA] + self.costTable['lambda'] *\
		    (self.statequeue[i+1].ap-self.statequeue[i].ap)
	    self.policy.BellmanUpdate(statei,statej,actionA,ri)

    def test(self):	
	APs,Returns = self.simulate()
	return APs[-1],Returns[-1]

    def simulate(self):
	
	self.genStateQueue()

	APs = [state.ap for state in self.statequeue]
	Returns = [0 for state in self.statequeue]
	for i in range(len(Returns)):
	    for j in range(0,i+1):
		Returns[i] += self.costTable[self.statequeue[j].actionType]
	    Returns[i] += self.costTable['lambda'] * \
		    (self.statequeue[i].ap-self.statequeue[0].ap) 
	return APs, Returns

    def reset(self):
	del self.statequeue
	del self.positive_docs
	del self.positive_leng
	del self.negative_docs
	del self.negative_leng
	del self.posprior
	del self.negprior
	
	self.statequeue = []
	self.positive_docs = []
	self.positive_leng = []
	self.negative_docs = []
	self.negative_leng = []
	self.posprior = deepcopy(self.query)
	self.negprior = {}
	self.curtHorizon = 1



