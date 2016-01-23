from collections import defaultdict
import math
import os
import operator
import pdb
# For Feature Extraction
from expansion import expansion
from retmath import *
from retrieval import *

dir = '../../ISDR-CMDP/'

"""
  Put search engine into dialogue manager?
"""

class DialogueManager(object):
  def __init__(self,background,inv_index,doclengs,answers,dir,docmodeldir,\
              iteration=10,mu=10,delta=1,topicleng=100, topicnumword=500):

    self.statemachine = StateMachine(
                              background = background,
                              inv_index = inv_index,
                              doclengs = doclengs,
                              dir = dir,
                              docmodeldir = docmodeldir
                              )

    self.answers = answers
    self.dir = dir
    self.docmodeldir = docmodeldir
    self.cpsID = '.'.join(docmodeldir.split('/')[1:-1])

    # Parameters for expansion and action
    self.iteration = iteration
    self.mu = mu
    self.delta = delta
    self.topicleng = topicleng
    self.topicnumword = topicnumword

    # Since we call it only once, put it in constructor
    self.topiclist = readTopicWords(self.cpsID)

  def __call__(self,query,ans_index):
    """
      Set inital parameters for every session

      Return:
        state(firstpass): 1 dim vector
    """
    self.query = query
    self.ans = self.answers[ans_index]

    # Information for actions
    self.keytermlist = readKeytermlist(self.cpsID,query)
    self.requestlist = readRequestlist(self.cpsID, self.ans)
    #self.topiclist = readTopicWords(cpsID) # Move to constructor, since we call it only once
    self.topicRanking = readTopicList(self.cpsID,ans_index)[:5]

    # Initialize Feedback models for this session
    self.positive_docs = []
    self.positive_leng = []
    self.negative_docs = []
    self.negative_leng = []

    self.posprior = self.query
    self.negprior = {}


    # Interaction Parameters
    self.cur_action = 5 # Action None
    self.curtHorizon = 0
    self.lastAP = 0.
    self.terminal = False

    # Previous retrieved results
    self.prev_ret = None

    return self
  # To be modified:
  def get_retrieved_result(self,ret):
    self.prev_ret = ret
    self.lastAP = evalAP(ret,self.ans)

  def featureExtraction(self):
    self.statemachine(
                    ret = self.prev_ret,
                    ans = self.ans,
                    action_type = self.cur_action,
                    curtHorizon = self.curtHorizon,
                    posmodel = self.posprior,
                    negmodel = self.negprior,
                    posprior = self.posprior,
                    negprior = self.negprior
                    )

    return self.statemachine.featureExtraction()

class StateMachine(object):
  def __init__(self,background,inv_index,doclengs,dir,docmodeldir,\
              iteration=10,mu=1,delta=1,alpha=0.1):

    self.background = background
    self.inv_index = inv_index
    self.doclengs = doclengs
    self.dir = dir
    self.docmodeldir = docmodeldir

    # Parameters for expansion
    self.iteration = iteration
    self.mu = mu
    self.delta = delta
    self.inv_index = inv_index
    self.alpha = alpha

  def __call__(self,ret,ans,action_type,curtHorizon,\
                posmodel,negmodel,posprior,negprior):
    """
      Return:
        feature: 1 dim vector
        AP: average precision of retrieved result
    """

    self.ret = ret
    self.ans = ans
    self.action_type = action_type
    self.curtHorizon = curtHorizon
    self.posmodel = posmodel
    self.negmodel = negmodel
    self.posprior = posprior
    self.negprior = negprior

    return self

  def featureExtraction(self):
    feature = []

    # Extract Features
    docs = []
    doclengs = []
    k = 0

    # read puesdo rel docs
    for docID,score in self.ret:
      if k>=20:
        break
      model = readDocModel(self.dir + self.docmodeldir + IndexToDocName(docID))
      docs.append(model)
      doclengs.append(self.doclengs[docID])
      k += 1

    irrel = cross_entropies(self.posmodel,self.background) - \
            0.1*cross_entropies(self.negmodel,self.background)

    ieDocT10 = defaultdict(float)
    ieDocT20 = defaultdict(float)

    WIG10 = 0.0
    WIG20 = 0.0
    NQC10 = 0.0
    NQC20 = 0.0
    for i in range(len(docs)):
      doc = docs[i]
      leng = doclengs[i]
      if i < 10:
        WIG10 += (self.ret[i][1]-irrel)/10.0
        NQC10 += math.pow(self.ret[i][1]-irrel,2)/10.0
        for wordID,prob in doc.iteritems():
          ieDocT10[ wordID ] += prob * leng
      else:
        WIG20 += (self.ret[i][1]-irrel)/10.0
        NQC20 += math.pow(self.ret[i][1]-irrel,2)/10.0
        for wordID,prob in doc.iteritems():
          ieDocT20[ wordID ] += prob * leng

    ieDocT10 = renormalize(ieDocT10)
    ieDocT20 = renormalize(ieDocT20)

    # bias term
    feature.append(1.0)

    # dialogue turn
    feature.append(float(self.curtHorizon))

    # prior entropy
    feature.append(entropy(self.posprior))

    # model entropy
    feature.append(entropy(self.posmodel))

    # write clarity
    feature.append(-1*cross_entropies(ieDocT10,self.background))
    feature.append(-1*cross_entropies(ieDocT20,self.background))

    # write WIG
    feature.append(WIG10)
    feature.append(WIG20)

    # write NQC
    feature.append(math.sqrt(NQC10))
    feature.append(math.sqrt(NQC20))

    # query feedback
    Ns = [10,20,50]
    posmodel = expansion({},docs[:10],doclengs[:10],\
                          self.background,self.iteration,self.mu,self.delta)
    oret = retrieveCombination(posmodel,self.negprior,\
                              self.background,self.inv_index,self.doclengs,\
                              self.alpha,0.1)
    N = min([self.ret,oret])
    for i in range(len(Ns)):
      if N<Ns[i]:
        Ns[i]=N

    qf10 = 0.0
    qf20 = 0.0
    qf50 = 0.0
    for docID1,rel1 in oret[:Ns[0]]:
      for docID2,rel2 in self.ret[:Ns[0]]:
        if docID1==docID2:
          qf10 += 1.0
          break
    for docID1,rel1 in oret[:Ns[1]]:
      for docID2,rel2 in self.ret[:Ns[1]]:
        if docID1==docID2:
          qf20 += 1.0
          break
    for docID1,rel1 in oret[:Ns[2]]:
      for docID2,rel2 in self.ret[:Ns[2]]:
        if docID1==docID2:
          qf50 += 1.0
          break

    feature.append(qf10)
    feature.append(qf20)
    feature.append(qf50)

    # retrieved number
    feature.append(len(self.ret))

    # query scope
    feature.append(QueryScope(self.posprior,self.inv_index))
    feature.append(QueryScope(self.posmodel,self.inv_index))

    # idf stdev
    feature.append(idfDev(self.posprior,self.inv_index))
    feature.append(idfDev(self.posmodel,self.inv_index))

    # cross entropy - prior, model, background
    feature.append(-1*cross_entropies(posmodel,self.posmodel))
    feature.append(-1*cross_entropies(self.posprior,self.posmodel))
    feature.append(-1*cross_entropies(self.posprior,self.background))
    feature.append(-1*cross_entropies(self.posmodel,self.background))
    del posmodel

    # IDF score, max, average
    maxIdf, avgIdf = IDFscore(self.posmodel,self.inv_index)
    feature.append(maxIdf)
    feature.append(avgIdf)

    # Statistics, top 5, 10, 20, 50, 100
    means, vars = Variability(self.ret,[5,10,20,50,100])
    for mean in means:
      feature.append(-1*mean)
    for var in vars:
      feature.append(var)

    # fit to exp distribution
    lamdas = [0.1,0.01,0.001]
    for lamda in lamdas:
      feature.append(FitExpDistribution(self.ret,lamda))

    # fit to gauss distribution
    Vars = [100,200,500]
    for v in Vars:
      feature.append(FitGaussDistribution(self.ret,0,v))

    # top N scores
    for key, val in self.ret[:49]:
      feature.append(-1*val)

    return feature

  def evalAP(ret,ans):
    relevant = sum(1.0 for docID, val in ret.iteritems() if ans.has_key(docID))
    return ( relevant / len(ans) if len(ans) else 0.0 )

"""

  Read functions for Dialogue Manager

"""
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

def readDocModel(fname):
  model = {}
  with open(fname) as fin:
    for line in fin.readlines():
      [ t1, t2 ] = line.split('\t')
      model[ int(t1) ] = float(t2)
  return model

if __name__ == "__main__":
  pass
