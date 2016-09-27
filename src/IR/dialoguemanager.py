from collections import defaultdict
import cPickle as pickle
import logging
import operator
import pdb

# For Feature Extraction
from actionmanager import ActionManager
from searchengine import SearchEngine
from statemachine import StateMachine
import numpy as np

class DialogueManager(object):
  def __init__(self,lex,background,inv_index,doclengs,dir,docmodeldir,feat):

    # Search Engine
    self.searchengine = SearchEngine(
                                lex         = lex,
                                background  = background,
                                inv_index   = inv_index,
                                doclengs    = doclengs,
                                dir         = dir
                                )

    # State Machine, feature extraction and AP estimation
    self.statemachine = StateMachine(
                              background  = self.searchengine.background,
                              inv_index   = self.searchengine.inv_index,
                              doclengs    = self.searchengine.doclengs,
                              dir         = dir,
                              docmodeldir = docmodeldir,
                              feat        = feat
                              )

    # Action Manager, performs query expansion
    self.actionmanager = ActionManager(
                              background  = self.searchengine.background,
                              doclengs    = self.searchengine.doclengs,
                              dir         = dir,
                              docmodeldir = docmodeldir
                              )

    # Training or Testing
    self.test_flag = True

  def __call__(self, query, ans, test_flag = False):
    """
      Set inital parameters for every session

      Return:
        state(firstpass): 1 dim vector
    """

    self.query = query
    #self.ans   = ( None if test_flag else ans )
    self.ans   = ans

    # Interaction Parameters, action and current turn number
    self.cur_action  = -1 # Action None
    self.curtHorizon = 0

    # Previous retrieved results and MAP
    self.ret     = None

    self.lastMAP = 0.
    self.MAP     = 0.

    # Termination indicator
    self.terminal = False

    # Training or Testing
    self.test_flag = test_flag

  def gen_state_feature(self):
    assert self.actionmanager.posmodel != None

    # Search Engine Retrieves Result
    self.ret = self.searchengine.retrieve( self.actionmanager.posmodel,\
                                            self.actionmanager.negmodel )
    self.curtHorizon += 1

    # Feature Extraction
    # State Estimation
    feature, estimatedMAP = self.statemachine(
                                              ret         = self.ret,
                                              action_type = self.cur_action,
                                              curtHorizon = self.curtHorizon,

                                              posmodel    = self.actionmanager.posprior,
                                              negmodel    = self.actionmanager.negprior,
                                              posprior    = self.actionmanager.posprior,
                                              negprior    = self.actionmanager.negprior
                                              )

    # Record mean average precision
    # Train state estimator
    #if not self.test_flag:
    self.lastMAP = self.MAP
    self.MAP = self.evalAP(self.ret,self.ans)

    return feature

  def request(self,action_type):
    '''

      Sends request to simulator for more query info

    '''
    self.cur_action = action_type
    request = {}
    request['ret']    = self.ret
    request['action'] = self.actionmanager.actionTable[ action_type ]
    return request

  def expand_query(self,response):
    '''

      Passes response to action manager for query expansion

    '''
    posmodel, negmodel = self.actionmanager.expand_query(response)

    self.posmodel = posmodel
    self.negmodel = negmodel

    return posmodel, negmodel

  def evalAP(self,ret,ans):
    tp = [ float(ans.has_key(docID)) for docID,val in ret ]
    atp = np.cumsum(tp)
    precision = [  atp[idx] / (idx+1) * tp[idx] for idx,(docID,val) in enumerate(ret)  ]
    return ( sum(precision)/len(ans) if len(ans) else 0. )

  def evalMAP(self,ret,ans):
    APs = [ evalAP(ret[i],ans[i]) for i in xrange(len(ret)) ]
    print "warning!! MAP"
    return sum(APs)/len(APs)

  def calculate_reward(self):
    if self.terminal:
      reward = self.actionmanager.costTable[ 4 ]  # 0
    else:
      #reward_std = self.actionmanager.noiseTable[ self.cur_action ]
      
      reward = self.actionmanager.costTable[ self.cur_action ] +\
               self.actionmanager.costTable['lambda'] * (self.MAP - self.lastMAP)

    return reward

  def show(self):
    self.terminal = True
    params = {'ret': self.ret }
    return params

  def game_over(self):
    if self.terminal or self.curtHorizon >= 5:
      self.query = None
      self.ans   = None
      self.actionmanager.posmodel = None
      return True, self.MAP
    return False, self.MAP

  """
  def save_features(self,filename):
    with open(filename + '.pkl','w') as f:
      pickle.dump((self.se_features,self.se_MAPs),f)
  """

"""

  Read functions for Dialogue Manager

"""

"""
dir = '../../ISDR-CMDP/'
def readKeytermlist(cpsID,fileIDs):
  if cpsID=='lattice.tandem':
    cpsID = 'onebest.tandem'
  elif cpsID=='lattice.CMVN':
    cpsID = 'onebest.CMVN'

  keyterms = defaultdict(float)

  for fileID,prob in fileIDs.iteritems():
    try:
      with open(dir + 'keyterm/' + cpsID + '/' + str(fileID) ) as fin:
        for i in range(100):
          pair = fin.readline().strip('\n').split('\t')
          keyterms[int(pair[0])] += prob*float(pair[1])
    except:
      print 'File does not exist'
  sortedKeytermlst = sorted(keyterms.iteritems(), key=operator.itemgetter(1),reverse = True)
  return sortedKeytermlst

def readRequestlist(cpsID,fileIDs):
  requests = defaultdict(float)
  for fileID,prob in fileIDs.iteritems():
    try:
      with open(dir + 'request/' + cpsID + '/' + str(fileID)) as fin:
        for line in fin.readlines():
          pair = line.strip('\n').split('\t')
          requests[ int(pair[0]) ] += float(pair[1])
    except:
      print 'File does not exist'
  return sorted(requests.iteritems(),key=operator.itemgetter(1),reverse=True)

def readTopicWords(cpsID):
  topicWordList = []
  for i in range(128):
    words = {}
    with open( dir + 'lda/' + cpsID + '/' + str(i) ) as fin:
      for line in fin.readlines():
        pair = line.split('\t')
        if len(pair) >= 2:
          words[ int(pair[0]) ] = float(pair[1])
    topicWordList.append(words)
  return topicWordList

def readTopicList(cpsID,qID):
  with open(dir+ 'topicRanking/' + cpsID + '/' + str(qID)) as fin:
    f = lambda x: (int(float(x[0])),float(x[1]))
    rankings = map(f,[ line.split('\t') for line in fin.readlines() ])
    return rankings

"""
if __name__ == "__main__":
  pass
