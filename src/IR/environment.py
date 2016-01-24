import pdb
from action import feedbackByDoc, feedbackByTopic, feedbackByKeyterm, feedbackByRequest, showResults
from action import genCostTable,genActionSet,ActionType
from simulator import Simulator
from retrieval import retrieveCombination,retrieve
from util import readLex, readFoldQueries, readBackground, readInvIndex, \
        readDocLength, readAnswer, readKeytermlist, readRequestlist, \
        readTopicWords, readTopicList
from state import State

#python python/continuousMDP.py PTV.lex 10fold/query/CMVN/train.fold1 10fold/query/CMVN/test.fold1 background/onebest.CMVN.bg index/onebest/PTV.onebest.CMVN.index doclength/onebest.CMVN.length PTV.ans docmodel/onebest/CMVN/ cmdp/theta/onebest.CMVN/theta.fold1 10 0.01

class Environment(object):
  def __init__(self,lex,background,inv_index,doclengs,answers,docmodeldir,dir,\
        alpha_d=1000,beta=0.1,em_iter=10,mu=10,delta=1,topicleng=100, topicnumword=500):
    """
      Description:
        Initialize Environment with
          lex,
          background model,
          inverted index,
          document lengths,
          answers
          directory
          other parameters
    """
    self.simulator = Simulator()
    self.simulator.addActions()
    self.costTable = genCostTable()
    self.actionSet = genActionSet()

    # Initialize
    self.lex = readLex(dir+lex)
    self.back = readBackground(dir+background,self.lex)
    self.inv_index = readInvIndex(dir+inv_index)
    self.doclengs = readDocLength(dir+doclengs)
    self.answers = readAnswer(dir+answers,self.lex)

    # Document Model Directory, varies with query
    self.dir = dir
    self.docmodeldir = docmodeldir

    # Parameters
    self.alpha_d = alpha_d
    self.beta = beta
    self.iteration = em_iter
    self.mu = mu
    self.delta = delta
    self.topicleng = topicleng
    self.topicnumword = topicnumword

  def setSession(self,query,index):
    """
      Description:
        Sets query and answer for this interactive session

      Return:
        state: 1 dim feature vector ( firstpass result )
    """

    self.query = query
    self.ans_index = index

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

    # Parameters for step ( interaction )
    self.curtHorizon = 0
    self.lastAP = 0.
    self.terminal = False

    firstpass = retrieve(self.query,self.back,self.inv_index,self.doclengs,self.alpha_d)

    self.prev_ret = firstpass

    # Initialize Retrieval Results with no action
    state = State(firstpass,\
            self.answers[self.ans_index], ActionType.ACTION_NONE,0,\
            self.docmodeldir,self.doclengs,self.back,\
            self.dir,self.iteration,self.mu,self.delta,\
            self.inv_index,self.alpha_d)

    state.setModels(self.posprior,self.negprior,\
                    self.posprior,self.negprior)

    feature = state.featureExtraction(state)
    return feature

  def step(self, action_type):
    """
      Description:
        Has to have a query before calling this function.

      Input:
        (1) action: integer value ( >= 0 )
      Return:
        (1) State: 1 dim vector
        (2) Reward: 1 real value
    """
    assert self.query != None
    assert 0 <= action_type <= 5, 'Action_type not found!'

    ##################
    #  Show Results  #
    ##################
    if action_type == 4:
        self.terminal = True
        reward = self.costTable[ action_type ] # + self.costTable['lambda'] * (self.lastAP - self.lastAP)
        feature = None
    else:
        action = self.actionSet[ action_type ]

        posmodel, negmodel = action(self.prev_ret,self.answers[self.ans_index],\
    		  self.simulator,self.posprior,self.negprior,\
    		  self.positive_docs,self.negative_docs,\
    		  self.positive_leng,self.negative_leng,\
    		  self.doclengs,self.back,self.iteration,
    		  self.mu,self.delta,self.docmodeldir,\
    		  self.keytermlist,self.requestlist,self.topiclist,\
    		  self.topicRanking,self.topicleng,self.topicnumword)

        ret = retrieveCombination(posmodel,negmodel,self.back,\
          self.inv_index,self.doclengs,self.alpha_d, self.beta )

        state = State(ret,self.answers[self.ans_index],action_type,self.curtHorizon,\
                    self.docmodeldir,self.doclengs,self.back,\
                    self.dir,self.iteration,self.mu,self.delta,\
                    self.inv_index,self.alpha_d)

        state.setModels(posmodel,negmodel,self.posprior,self.negprior)

        feature = state.featureExtraction(state)

        reward = self.costTable[action_type] + self.costTable['lambda'] * (state.ap - self.lastAP)

        self.lastAP = state.ap
        self.prev_ret = ret
        self.curtHorizon += 1

    return reward,feature

  def game_over(self):
    if self.curtHorizon >= 5 or self.terminal:
        self.query = None
        self.ans_index = None
        return True
    return False

if __name__ == "__main__":

  dir = '../../ISDR-CMDP/'
  lex = 'PTV.lex'
  train_queries = '10fold/query/CMVN/train.fold1'
  test_queries = '10fold/query/CMVN/test.fold1'
  background = 'background/onebest.CMVN.bg'
  inv_index = 'index/onebest/PTV.onebest.CMVN.index'
  doclengs = 'doclength/onebest.CMVN.length'
  answers = 'PTV.ans'
  docmodeldir = 'docmodel/onebest/CMVN/'

  train,train_indexes = readFoldQueries(dir + train_queries)

  env = Environment(lex,background,inv_index, \
    doclengs,answers,docmodeldir,dir,alpha_d=1000,beta=0.1,em_iter=10,mu=10,\
    delta=1,topicleng=100, topicnumword=500)
  pdb.set_trace()
  env.setSession(train[1],train_indexes[1])
