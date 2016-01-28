from collections import defaultdict
import math
import os
import operator
import pdb

# For Feature Extraction
from actionmanager import genCostTable
from expansion import expansion
from searchengine import SearchEngine
from statemachine import StateMachine
from util import *

"""
  Todo: action manager
"""
class DialogueManager(object):
  def __init__(self,lex,background,inv_index,doclengs,dir,docmodeldir,\
              topicleng=50, topicnumword=1000):

    # Cost Table
    self.costTable = genCostTable()

    # Search Engine
    self.searchengine = SearchEngine(
                                lex = lex,
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
                              docmodeldir = docmodeldir
                              )

    self.background  = self.searchengine.background
    self.doclengs    = self.searchengine.doclengs
    self.dir         = dir
    self.docmodeldir = docmodeldir
    self.cpsID       = '.'.join(docmodeldir.split('/')[1:-1])
    self.topiclist   = readTopicWords(self.cpsID)


    # Parameters for expansion and action
    self.topicleng = topicleng
    self.topicnumword = topicnumword

    # Training or Testing
    self.test_flag = True

  def __call__(self, query, ans, test_flag = False):
    """
      Set inital parameters for every session

      Return:
        state(firstpass): 1 dim vector
    """
    # Query and answer (for calculating average precision)
    self.posmodel = query
    self.negmodel = None

    self.ans = ( None if test_flag else ans )

    # Initialize Feedback models for this session
    self.posdocs  = []
    self.poslengs  = []
    self.negdocs  = []
    self.neglengs = []

    self.posprior = self.posmodel
    self.negprior = {}

    # Interaction Parameters
    self.cur_action  = 5 # Action None
    self.curtHorizon = 0

    # Previous retrieved results and MAP
    self.ret    = None

    self.lastMAP = 0.
    self.MAP     = 0.

    # Termination indicator
    self.terminal = False

    # Training or Testing
    self.test_flag = test_flag

  def get_retrieved_result(self):
    assert self.posmodel != None
    # Search Engine Retrieves Result
    self.ret = self.searchengine.retrieve(self.posmodel,self.negmodel)

    # Feature extraction and estimation performed by State Machine
    self.curtHorizon += 1
    feature, estimatedMAP = self.statemachine(
                                              ret = self.ret,
                                              action_type = self.cur_action,
                                              curtHorizon = self.curtHorizon,
                                              posmodel = self.posprior,
                                              negmodel = self.negprior,
                                              posprior = self.posprior,
                                              negprior = self.negprior
                                              )
    # Record mean average precision
    self.lastMAP = self.MAP
    self.MAP = ( estimatedMAP if self.test_flag else self.evalMAP(self.ret,self.ans) )

    return feature

  def request(self,action_type):
    '''
      Sends request to simulator for more query info
    '''
    self.cur_action = action_type
    return (self.ret, self.cur_action)

  def expand_query(self,response):
    '''
      Receives response from simulator
      PS: 204 is HTTP status code for empty content
    '''
    if self.cur_action == 0:
      doc = response
      if doc:
        self.posdocs.append(self.docmodeldir+IndexToDocName(doc))
      	self.poslengs.append(self.doclengs[doc])
    elif self.cur_action == 1:
      if response is not 204:
        keyterm = response[0]
        boolean = response[1]
        if boolean:
          self.posprior[ keyterm ] = 1.
        else:
          self.negprior[ keyterm ] = 1.
    elif self.cur_action == 2:
      request = response
      self.posprior[request] = 1.0
    elif self.cur_action == 3:
      topicIdx = response
      self.posdocs.append(pruneAndNormalize(self.topiclist[topicIdx],self.topicnumword))
      self.poslengs.append(self.topicleng)
    elif self.cur_action == 4:
      # This condition shouldn't happen, since we blocked this in environment.py
      assert response == None
      assert 0, 'Should not show result in dialogue manager expand query function!'
      self.terminal = True

    posmodel = expansion(renormalize(self.posprior),self.posdocs,self.poslengs,self.background)
    negmodel = expansion(renormalize(self.negprior),self.negdocs,self.doclengs,self.background)

    self.posmodel = posmodel
    self.negmodel = negmodel

    return posmodel, negmodel

  def evalMAP(self,ret,ans):
    Precision_sum = sum([ 1.0/(idx+1) if ans.has_key(docID) else 0. for idx,(docID,val) in enumerate(ret) ])
    return ( Precision_sum/len(ans) if len(ans) else 0. )

  def calculate_reward(self):
    reward = self.costTable[ self.cur_action ] + self.costTable['lambda'] * (self.MAP - self.lastMAP)
    return reward

  def game_over(self):
    if self.terminal or self.curtHorizon >= 5:
      self.query = self.ans = None
      return True
    return False

"""

  Read functions for Dialogue Manager

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


if __name__ == "__main__":
  pass
