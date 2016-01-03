import pdb

from action import feedbackByDoc, feedbackByTopic, feedbackByKeyterm, feedbackByRequest, showResults
from action import genCostTable
from simulator import Simulator
from retrieve import retrieveCombination
from util import readLex, readFoldQueries, readBackground, readInvIndex, \
        readDocLength, readAnswer, readKeytermlist, readRequestlist, \
        readTopicWords, readTopicList

#python python/continuousMDP.py PTV.lex 10fold/query/CMVN/train.fold1 10fold/query/CMVN/test.fold1 background/onebest.CMVN.bg index/onebest/PTV.onebest.CMVN.index doclength/onebest.CMVN.length PTV.ans docmodel/onebest/CMVN/ cmdp/theta/onebest.CMVN/theta.fold1 10 0.01

class Environment(object):
  def __init__(self,lex,train_quries,test_queries,background,inv_index,\
        doclengs,answers,docmodeldir,alpha_d=1000,em_iter=10, mu=10,\
        delta=1,topicleng=100, topicnumword=500):
    """
    Initialize Environment with
      lex, background model, inverted index, document lengths, and answers

    Todo:
      Agent should receive train_queries & test_queries
    """
    self.simulator = Simulator()

    # Initialize
    self.lex = readLex(lex)
    self.train_queries, self.train_indexes = readFoldQueries(train_queries)
    self.test_queries, self.test_indexes = readFoldQueries(test_queries)
    self.background = readBackground(background,self.lex)
    self.inv_index = readInvIndex(inv_index)
    self.doclengs = readDocLength(doclengs)
    self.answers = readAnswer(answers,self.lex)

    # Document Model Directory, varies with query
    self.docmodeldir = docmodeldir

    # parameters
    self.alpha_d = alpha_d
    self.iteration = em_iter
    self.mu = mu
    self.delta = delta
    self.topicleng = topicleng
    self.topicnumword = topicnumword

  def query(self,query,ans,index):
    """
    Query before retrieving anything
    """
    self.query = query

    cpsID = '.'.join(self.docmodeldir.split('/')[1:-1])

    self.keytermlist = readKeytermlist(cpsID,query)
  	self.requestlist = readRequestlist(cpsID,ans)
  	self.topiclist = readTopicWords(cpsID)
  	self.topicRanking = readTopicList(cpsID,index)[:5]

    # feedback models
  	self.positive_docs = []
  	self.positive_leng = []
  	self.negative_docs = []
  	self.negative_leng = []
  	self.posprior = self.query
  	self.negprior = {}

    self.curtHorizon = 0.

    self.firstpass = True

    self.initAP = None

  def retrieve(self, action):
    """
    Has to have a query before retrieving anything
    Input: action
    Return: (1) State: 1 dim vector
        (2) Reward: 1 real value
    """
    assert self.query != None

    posmodel, negmodel = action(curtState.ret,self.answer,\
		  self.simulator,self.posprior,self.negprior,\
		  self.positive_docs,self.negative_docs,\
		  self.positive_leng,self.negative_leng,\
		  self.doclengs,self.back,self.iteration,
		  self.mu,self.delta,self.docmodeldir,\
		  self.keytermlst,self.requestlst,self.topiclst,\
		  self.topicRanking,self.topicleng,self.topicnumwords)

    ret = retrieveCombination(posmodel,negmodel,self.back,\
      self.inv_index,self.doclengs,self.alpha_d,beta=0.1)

    self.curtHorizon += 1

    """
    Since dictionaries can hold function has keys,
    We can write a new cost table

    Todo:
      New cost table
    """

    if self.firstpass is True:
      pass


def featureExtraction(self,statequeue):

  actions = [state.actionType for state in statequeue[0:]]

  state = statequeue[-1]
  feature = []

  # Extract Features
  docs = []
  doclengs = []
  k = 0
  # read puesdo rel docs
  for docID,score in state.ret:
    if k >= 20:
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

  # bias term
  feature.append(1.0)

  # dialogue turn
  feature.append(float(state.horizon))

  # actions taken
  #for i in range(0,4):
  #  feature.append(0.0)
  #for a in actions[1:]:
  #  feature[a+1] += 1.0

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


if __name__ == "__main__":
  lex = 'PTV.lex'
  train_queries = '10fold/query/CMVN/train.fold1'
  test_queries = '10fold/query/CMVN/test.fold1'
  background = 'background/onebest.CMVN.bg'
  inv_index = 'index/onebest/PTV.onebest.CMVN.index'
  doclengs = 'doclength/onebest.CMVN.length'
  answers = 'PTV.ans'
  docmodeldir = 'docmodel/onebest/CMVN/'

  env = Environment(lex,train_queries,test_queries,\
          background,inv_index,\
          doclengs,answers,docmodeldir)
  env.query(env.train_queries[1],env.answers[1],env.train_indexes[1])
