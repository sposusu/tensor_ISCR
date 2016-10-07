from collections import defaultdict
import cPickle as pickle
import logging
import operator
import os

# For Feature Extraction
from actionmanager import ActionManager
from searchengine import SearchEngine
from statemachine import StateMachine
import numpy as np

class DialogueManager(object):
  def __init__(self, data_dir, feature_type):

    docmodels_pickle    = os.path.join(data_dir,'docmodels.pickle')
    topiclist_pickle    = os.path.join(data_dir,'lda','topiclist.pickle')

    data_name = data_dir.split('/')[-1]

    lex_file        = os.path.join(data_dir,data_name + '.lex')
    background_file = os.path.join(data_dir,data_name + '.background')
    inv_index_file  = os.path.join(data_dir, data_name + '.index')
    doclengs_file   = os.path.join(data_dir, data_name + '.doclength')

    # Search Engine
    self.searchengine = SearchEngine(
                                    lex_file        = lex_file,
                                    background_file = background_file,
                                    inv_index_file  = inv_index_file,
                                    doclengs_file   = doclengs_file
                                )

    # State Machine
    docmodel_dir    = os.path.join(data_dir,'docmodel')

    self.statemachine = StateMachine(
                              background    = self.searchengine.background,
                              inv_index     = self.searchengine.inv_index,
                              doclengs      = self.searchengine.doclengs,
                              docmodel_dir  = docmodel_dir,
                              feat          = feature_type
                              )

    # Action Manager
    #topic_dir       = os.path.join(data_dir,'lda')
    self.actionmanager = ActionManager(
                              background  = self.searchengine.background,
                              doclengs    = self.searchengine.doclengs,
                              data_dir    = data_dir
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
    feature = self.statemachine(  ret         = self.ret,
                                  action_type = self.cur_action,
                                  curtHorizon = self.curtHorizon,

                                  posmodel    = self.actionmanager.posprior,
                                  negmodel    = self.actionmanager.negprior,
                                  posprior    = self.actionmanager.posprior,
                                  negprior    = self.actionmanager.negprior  )

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

if __name__ == "__main__":
    data_dir = '/home/ubuntu/InteractiveRetrieval/data/reference'
    searchengine_pickle = os.path.join(data_dir,'searchengine.pickle')
    docmodels_pickle    = os.path.join(data_dir,'docmodels.pickle')
    topiclist_pickle    = os.path.join(data_dir,'lda','topiclist.pickle')
    feat = 'all'
    dm = DialogueManager(searchengine_pickle, docmodels_pickle, topiclist_pickle,feat)
    import pdb; pdb.set_trace()
