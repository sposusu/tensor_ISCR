import os
import sys
import operator
from util import *
from action import *
from retrieval import *
from action import *
from simulator import *
from policy_nonumpy import *
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
	    docmodeldir,em_iter,mu,delta,topicleng=100,topicnumword=500):
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

    def genStateQueue(self,reserveModel=False):
	# Generate Firstpass
	firstpass = retrieve(self.query,self.back,self.inv_index,\
		self.doclengs,self.alpha_d)
	self.statequeue.append(State(firstpass,self.answer,\
		ActionType.ACTION_NONE,0))
	self.curtHorizon= 0
	actionType, action = \
		self.policy.getAction(self.statequeue[-1],self.curtHorizon)
	self.statequeue[-1].setModels(self.posprior,self.negprior,self.posprior,self.negprior)
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
	    self.statequeue.append(State(ret,self.answer,actionType,\
		    self.curtHorizon))

	    if reserveModel:
		self.statequeue[-1].setModels(posmodel,negmodel,self.posprior,self.negprior)
	    
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
		    (self.statequeue[i+1].eval-self.statequeue[i].eval)
	    self.policy.BellmanUpdate(statei,statej,actionA,ri)
    
    def FittedValueIteration(self):
	
	self.genStateQueue()

	# Policy update
	for i in range(len(self.statequeue)-1):
	    statei = self.statequeue[i]
	    statej = self.statequeue[i+1]
	    actionA = statej.actionType
	    ri = self.costTable[actionA] + self.costTable['lambda'] *\
		    (self.statequeue[i+1].eval-self.statequeue[i].eval)
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

    def generateFeatures(self,fout):

	self.genStateQueue(True)
	
	actions = [state.actionType for state in self.statequeue[0:]]
	for i in range(len(self.statequeue)):
	    state = self.statequeue[i]
	    
	    # Clarity, WIG, NCQ 
	    docs = []
	    doclengs = []
	    k = 0
	    # read puesdo rel docs
	    for docID,score in state.ret:
		if k>=20:
		    break
		model = readDocModel(self.docmodeldir+\
			IndexToDocName(docID))
		docs.append(model)
		doclengs.append(self.doclengs[docID])
		k+=1
	    
	    irrel = cross_entropies(state.posmodel,self.back) - \
		    0.1*cross_entropies(state.negmodel,self.back)
	    
	    ieDocT10 = {}
	    ieDocT20 = {}
	    WIG10 = 0.0
	    WIG20 = 0.0
	    NQC10 = 0.0
	    NQC20 = 0.0
	    for i in range(len(docs)):
		doc = docs[i]
		leng = doclengs[i]
		if i<10:
		    WIG10 += (state.ret[i][1]-irrel)/10.0
		    NQC10 += math.pow(state.ret[i][1]-irrel,2)/10.0
		    for wordID,prob in doc.iteritems():
			if ieDocT10.has_key(wordID):
			    ieDocT10[wordID] += prob*leng
			else:
			    ieDocT10[wordID] = prob*leng
		else:
		    WIG20 += (state.ret[i][1]-irrel)/10.0
		    NQC20 += math.pow(state.ret[i][1]-irrel,2)/10.0
		    for wordID,prob in doc.iteritems():
			if ieDocT20.has_key(wordID):
			    ieDocT20[wordID] += prob*leng
			else:
			    ieDocT20[wordID] = prob*leng
	    ieDocT10 = renormalize(ieDocT10)
	    ieDocT20 = renormalize(ieDocT20)
	    
	    if math.fabs(WIG10)>1000:
		continue

	    # output MAP label
	    fout.write(str(state.ap)+'\t')
	    
	    # dialogue turn 
	    fout.write(str(state.horizon)+' ')
	    
	    # actions taken
	    for a in range(1,6):
		if a<state.horizon+1:
		    fout.write(str(actions[a])+' ')
		else:
		    fout.write('0 ')

	    # prior entropy
	    fout.write(str(entropy(state.posprior))+' ')

	    # model entropy
	    fout.write(str(entropy(state.posmodel))+' ')

	    # write clarity
	    fout.write(str(-1*cross_entropies(ieDocT10,self.back))+' ')
	    fout.write(str(-1*cross_entropies(ieDocT20,self.back))+' ')
	    
	    # write WIG
	    fout.write(str(WIG10)+' ')
	    fout.write(str(WIG20)+' ')
	    
	    # write NQC
	    fout.write(str(math.sqrt(NQC10))+' ')
	    fout.write(str(math.sqrt(NQC20))+' ')

	    # query feedback
	    Ns = [10,20,50]
	    posmodel = expansion({},docs[:10],doclengs[:10],\
		    self.back,self.iteration,self.mu,self.delta) 
	    oret = retrieveCombination(posmodel,state.negprior,\
		    self.back,self.inv_index,self.doclengs,\
		    self.alpha_d,0.1)
	    N = min([state.ret,oret])
	    for i in range(len(Ns)):
		if N<Ns[i]: 
		    Ns[i]=N

	    qf10 = 0.0
	    qf20 = 0.0
	    qf50 = 0.0
	    for docID1,rel1 in oret[:Ns[0]]:
		for docID2,rel2 in state.ret[:Ns[0]]:
		    if docID1==docID2:
			qf10 += 1.0
			break
	    for docID1,rel1 in oret[:Ns[1]]:
		for docID2,rel2 in state.ret[:Ns[1]]:
		    if docID1==docID2:
			qf20 += 1.0
			break
	    for docID1,rel1 in oret[:Ns[2]]:
		for docID2,rel2 in state.ret[:Ns[2]]:
		    if docID1==docID2:
			qf50 += 1.0
			break
	    fout.write(str(qf10)+' ')
	    fout.write(str(qf20)+' ')
	    fout.write(str(qf50)+' ')

	    # retrieved number
	    fout.write(str(len(state.ret))+' ')

	    # query scope
	    fout.write(str(QueryScope(state.posprior,self.inv_index))+' ')
	    fout.write(str(QueryScope(state.posmodel,self.inv_index))+' ')

	    # idf stdev
	    fout.write(str(idfDev(state.posprior,self.inv_index))+' ')
	    fout.write(str(idfDev(state.posmodel,self.inv_index))+' ')

	    # cross entropy - prior, model, background
	    fout.write(str(-1*cross_entropies(posmodel,state.posmodel))+' ')
	    fout.write(str(-1*cross_entropies(state.posprior,state.posmodel))+' ')
	    fout.write(str(-1*cross_entropies(state.posprior,self.back))+' ')
	    fout.write(str(-1*cross_entropies(state.posmodel,self.back))+' ')
	    del posmodel

	    # IDF score, max, average
	    maxIdf, avgIdf = IDFscore(state.posmodel,self.inv_index)
	    fout.write(str(maxIdf)+' ')
	    fout.write(str(avgIdf)+' ')

	    # Statistics, top 5, 10, 20, 50, 100 
	    means, vars = Variability(state.ret,[5,10,20,50,100])
	    for mean in means:
		fout.write(str(-1*mean)+' ')
	    for var in vars:
		fout.write(str(var)+' ')

	    # fit to exp distribution
	    lamdas = [0.1,0.01,0.001]
	    for lamda in lamdas:
		fout.write(str(FitExpDistribution(state.ret,lamda))+' ')
	    
	    # fit to gauss distribution
	    Vars = [100,200,500]
	    for v in Vars:
		fout.write(str(FitGaussDistribution(state.ret,0,v))+' ')

	    # top N scores
	    for key, val in state.ret[:50]:
		fout.write(str(-1*val)+' ')
	    fout.write('\n') 

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

