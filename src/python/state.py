from util import *
from retrieval import *
import math
import pdb
from expansion import *

class State:

    def __init__(self,ret,ans,actionType,horizon,docmodeldir='',doclengs={},
	    back={},dir = [],iteration=10,mu=1,delta=1,inv_index={},alpha_d=0.1):
        self.ret = ret
        self.ans = ans
        self.ap = evalAP(ret,ans)
        self.eval = self.ap
        self.actionType = actionType
        self.horizon = horizon
        self.posmodel = None
        self.negmodel = None
        self.posprior = None
        self.negprior = None
        self.dir = dir
        self.docmodeldir = docmodeldir
        self.doclengs = doclengs
        self.back = back
        self.iteration = iteration
        self.mu = mu
        self.delta = delta
        self.inv_index = inv_index
        self.alpha_d = alpha_d
    """
    def totuple(self):
        bin = math.floor(self.eval*10)/10.0
        if bin==1.0:
            bin = 0.9
        return (bin,self.horizon)
    """
    def setModels(self,pos,neg,pprior,nprior):
        self.posmodel = pos
        self.negmodel = neg
        self.posprior = pprior
        self.negprior = nprior

    def featureExtraction(self,state):

	feature = []

	# Extract Features
	docs = []
        doclengs = []
	k = 0

	# read puesdo rel docs
	for docID,score in state.ret:
	    if k>=20:
		break

	model = readDocModel(self.dir + self.docmodeldir+\
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

	# bias term
	feature.append(1.0)

	# dialogue turn
	feature.append(float(state.horizon))

	# actions taken
	#for i in range(0,4):
	#    feature.append(0.0)
	#for a in actions[1:]:
	#    feature[a+1] += 1.0

	# prior entropy
	feature.append(entropy(state.posprior))

	# model entropy
	feature.append(entropy(state.posmodel))

	# write clarity
	feature.append(-1*cross_entropies(ieDocT10,self.back))
	feature.append(-1*cross_entropies(ieDocT20,self.back))

	# write WIG
	feature.append(WIG10)
	feature.append(WIG20)

	# write NQC
	feature.append(math.sqrt(NQC10))
	feature.append(math.sqrt(NQC20))

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
	feature.append(qf10)
	feature.append(qf20)
	feature.append(qf50)

	# retrieved number
	feature.append(len(state.ret))

	# query scope
	feature.append(QueryScope(state.posprior,self.inv_index))
	feature.append(QueryScope(state.posmodel,self.inv_index))

	# idf stdev
	feature.append(idfDev(state.posprior,self.inv_index))
	feature.append(idfDev(state.posmodel,self.inv_index))

	# cross entropy - prior, model, background
	feature.append(-1*cross_entropies(posmodel,state.posmodel))
	feature.append(-1*cross_entropies(state.posprior,state.posmodel))
	feature.append(-1*cross_entropies(state.posprior,self.back))
	feature.append(-1*cross_entropies(state.posmodel,self.back))
	del posmodel

	# IDF score, max, average
	maxIdf, avgIdf = IDFscore(state.posmodel,self.inv_index)
	feature.append(maxIdf)
	feature.append(avgIdf)

	# Statistics, top 5, 10, 20, 50, 100
	means, vars = Variability(state.ret,[5,10,20,50,100])
	for mean in means:
	    feature.append(-1*mean)
	for var in vars:
	    feature.append(var)

	# fit to exp distribution
	lamdas = [0.1,0.01,0.001]
	for lamda in lamdas:
	    feature.append(FitExpDistribution(state.ret,lamda))

	# fit to gauss distribution
	Vars = [100,200,500]
	for v in Vars:
	    feature.append(FitGaussDistribution(state.ret,0,v))

	# top N scores
	for key, val in state.ret[:49]:
	    feature.append(-1*val)

	#print feature
	return feature
"""
class DiscreteState(State):

    def __init__(self,ret,ans,actionType,horizon,docmodeldir,\
	    doclengs,back,iteration,mu,delta,inv_index,alpha_d,\
	    statequeue,weights):
	State.__init__(self,ret,ans,actionType,horizon,docmodeldir,\
		doclengs,back,iteration,mu,delta,inv_index,alpha_d)

    def estimate(self,statequeue,weights,means,devs):
	feature = self.featureExtraction(statequeue)
	self.eval = 0.0
	for i in range(len(weights)):
	    if devs[i]==0.0:
		self.eval += weights[i]*(feature[i]-0.9993*means[i])
	    else:
		self.eval += weights[i]*(feature[i]-means[i])/devs[i]
	#print self.eval,self.ap
	if self.eval < 0:
	    self.eval = 0.0
	elif self.eval >10.0:
	    self.eval = 0.0
	elif self.eval >1.0:
	    self.eval = 1.0

class ContinuousState(State):

    def __init__(self,ret,ans,actionType,horizon,docmodeldir,\
	    doclengs,back,iteration,mu,delta,inv_index,alpha_d,\
	    weights):
	State.__init__(self,ret,ans,actionType,horizon,docmodeldir,\
		doclengs,back,iteration,mu,delta,inv_index,alpha_d)

    def estimate(self,statequeue,weights,means,devs):
	feature = self.featureExtraction(statequeue)
	self.eval = 0.0

	for i in range(len(weights)):
	    if devs[i]==0.0:
		self.eval += weights[i]*(feature[i]-0.9993*means[i])
	    else:
		self.eval += weights[i]*(feature[i]-means[i])/devs[i]
	#print self.eval,self.ap
	if self.eval < 0:
	    self.eval = 0.0
	elif self.eval >10.0:
	    self.eval = 0.0
	elif self.eval >1.0:
	    self.eval = 1.0
"""
