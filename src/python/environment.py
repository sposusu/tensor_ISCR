import pdb

from action import feedbackByDoc, feedbackByTopic, feedbackByKeyterm, feedbackByRequest, showResults
from action import genCostTable,genActionSet
from simulator import Simulator
from state import ContinuousState
from retrieval import retrieveCombination,retrieve
from util import readLex, readFoldQueries, readBackground, readInvIndex, \
        readDocLength, readAnswer, readKeytermlist, readRequestlist, \
        readTopicWords, readTopicList

#python python/continuousMDP.py PTV.lex 10fold/query/CMVN/train.fold1 10fold/query/CMVN/test.fold1 background/onebest.CMVN.bg index/onebest/PTV.onebest.CMVN.index doclength/onebest.CMVN.length PTV.ans docmodel/onebest/CMVN/ cmdp/theta/onebest.CMVN/theta.fold1 10 0.01

class Environment(object):
  def __init__(self,lex,background,inv_index,\
        doclengs,answers,docmodeldir,dir,alpha_d=1000,beta=0.1,em_iter=10,mu=10,\
        delta=1,topicleng=100, topicnumword=500):
    """
    Initialize Environment with
      lex, background model, inverted index, document lengths, and answers

    Todo:
      Agent should receive train_queries & test_queries
    """
    self.simulator = Simulator()
    self.costTable = genCostTable()
    self.actionSet = genActionSet()

    # Initialize
    print lex
    print 'dir = ',dir
    self.lex = readLex(dir+lex)
    self.back = readBackground(dir+background,self.lex)
    self.inv_index = readInvIndex(dir+inv_index)
    self.doclengs = readDocLength(dir+doclengs)
    self.answers = readAnswer(dir+answers,self.lex)

    # Document Model Directory, varies with query
    self.docmodeldir = docmodeldir

    # parameters
    self.alpha_d = alpha_d
    self.beta = beta
    self.iteration = em_iter
    self.mu = mu
    self.delta = delta
    self.topicleng = topicleng
    self.topicnumword = topicnumword

  def setSession(self,query,index):
    """
    Set query and answer for this session
    return state: 1 dim-vector for initialization
    """
    self.query = query
    cpsID = '.'.join(self.docmodeldir.split('/')[1:-1])
    self.keytermlist = readKeytermlist(cpsID,query)
    self.requestlist = readRequestlist(cpsID,self.answers[index]) #  !!
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

    firstpass = retrieve(self.query,self.back,self.inv_index,self.doclengs,self.alpha_d)

    state = ContinuousState(firstpass,\
            self.answer,ActionType.ACTION_NONE,0,\
            self.docmodeldir,self.doclengs,self.back,\
            self.iteration,self.mu,self.delta,\
            self.inv_index,self.alpha_d,\
            self.statequeue,self.weights)

    state.setModels(self.posprior,self.negprior,\
            self.posprior,self.negprior)

    feature = state.featureExtraction(state)
    
    return feature

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
      self.inv_index,self.doclengs,self.alpha_d, self.beta )

    state = ContinuousState(ret,\
                self.answer,actionType,self.curtHorizon,\
                self.docmodeldir,self.doclengs,self.back,\
                self.iteration,self.mu,self.delta,\
                self.inv_index,self.alpha_d,\
                self.statequeue,self.weights)

    state.setModels(posmodel,negmodel,\
                self.posprior,self.negprior)

    feature = state.featureExtraction(state)

    self.curtHorizon += 1
    
    actionA = statej.actionType
    reward = self.costTable[actionA] + self.costTable['lambda'] * (state.AP - lastAP)
    lastAP = state.AP
    
    return feature,reward

  def game_over():
    if self.curtHorizon >= 5:
      return False

    self.query = None
    return True

  def _evalAP(ret,ans):
    if len(ans) == 0:
      return 0.0
    AP = 0.0
    cnt = 0.0
    get = 0.0
    for docID, val in ret:
      cnt += 1.0
      if ans.has_key(docID):
        get += 1.0
        AP += float(get)/float(cnt)
    AP /= float(len(ans))
    return AP

if __name__ == "__main__":
  lex = 'PTV.lex'
  train_queries = '10fold/query/CMVN/train.fold1'
  test_queries = '10fold/query/CMVN/test.fold1'
  background = 'background/onebest.CMVN.bg'
  inv_index = 'index/onebest/PTV.onebest.CMVN.index'
  doclengs = 'doclength/onebest.CMVN.length'
  answers = 'PTV.ans'
  docmodeldir = 'docmodel/onebest/CMVN/'

  train = readFoldQueries(train_queries)

  pdb.set_trace()

  env = Environment(lex,train_queries,test_queries,\
          background,inv_index,\
          doclengs,answers,docmodeldir)

  env.setSession(env.train_queries[1],env.answers[1],env.train_indexes[1])
