"""
  Action Manager is pulled out dialogue manager
  because it would be easier to add more actions in a separate file
"""

from collections import defaultdict
import json
import operator
import pdb

from util import  IndexToDocName, pruneAndNormalize
from util import  readDocModel, readTopicList, readTopicWords
from util import  renormalize

# Define Cost Table
def genCostTable():
  values = [ -30., -10., -50., -20., 0., 0., 1000. ]
  costTable = dict(zip(range(6)+['lambda'],values))
  return costTable

def genActionTable():
  at = {}
  at[-1] = 'none'
  at[0]  = 'doc'
  at[1]  = 'keyterm'
  at[2]  = 'request'
  at[3]  = 'topic'
  at[4]  = 'show'
  return at

class ActionManager(object):
  def __init__(self, background, doclengs, dir, docmodeldir, \
                    topicleng=50, topicnumword=1000):

    self.background  = background
    self.doclengs    = doclengs

    self.dir         = dir
    self.docmodeldir = docmodeldir

    self.cpsID       = '.'.join(docmodeldir.split('/')[1:-1])
    self.topiclist   = readTopicWords(self.cpsID)

    # Parameters for expansion and action
    self.topicleng    = topicleng
    self.topicnumword = topicnumword

    # Since agent only returns integer as action
    self.actionTable  = genActionTable()
    self.costTable    = genCostTable()
    
  def __call__(self, query):
    self.posmodel = query
    self.negmodel = None

    self.posdocs  = []
    self.poslengs = []
    self.negdocs  = []
    self.neglengs = []

    self.posprior = self.posmodel
    self.negprior = {}


  def expand_query(self,params):
    assert isinstance(params,dict)

    action = params['action']

    if action == 'none':
      query = params['query']
      self.__call__(query)

    elif action == 'doc':
      doc = params['doc']
      if doc:
        self.posdocs.append(self.docmodeldir+IndexToDocName(doc))
      	self.poslengs.append(self.doclengs[doc])

    elif action == 'keyterm':
      if 'keyterm' in params:
        keyterm, isrel = params['keyterm']
        if isrel:
          self.posprior[ keyterm ] = 1.
        else:
          self.negprior[ keyterm ] = 1.

    elif action == 'request':
      request = params['request']
      self.posprior[ request ] = 1.0

    elif action == 'topic':
      topicIdx = params['topic']
      try:
        self.topiclist[ topicIdx ]
        self.posdocs.append(pruneAndNormalize(self.topiclist[ topicIdx ],self.topicnumword))
      except:
        print 'Index out of range error'
      self.poslengs.append(self.topicleng)

    elif action == 'show':
      # This condition shouldn't happen, since we blocked this in environment.py
      assert 0

    posmodel = expansion(renormalize(self.posprior),self.posdocs,self.poslengs,self.background)
    negmodel = expansion(renormalize(self.negprior),self.negdocs,self.doclengs,self.background)

    self.posmodel = posmodel
    self.negmodel = negmodel

    return posmodel, negmodel

def expansion(prior,docnames,doclengs,back,iteration=10,mu=10,delta=1):
  dir = '../../ISDR-CMDP/'

  models = []
  alphas = []

  for name in docnames:
    if isinstance(name, str):
      models.append(readDocModel(dir+name))
    else:
      models.append(name)
    alphas.append(0.5)

  N = float(len(docnames))

  # init query model
  query = defaultdict(float)
  for model in models:
    for word,val in model.iteritems():
      query[word] += val/N

  # EM expansion
  for it in range(iteration):
    # E step, Estimate P(Zw,d=1)
    aux = {}
    for m in range(len(models)):
      model = models[m]
      alpha = alphas[m]
      aux[m] = defaultdict(float)
      for word,val in model.iteritems():
        aux[m][word] = alpha*query[word]/(alpha*query[word] + (1-alpha)*back[word])

    # M step
    # Estimate alpha_D
    tmpmass = defaultdict(float)
    for m in range(len(models)):
      alphas[m] = 0.
      model = models[m]
      for word,val in model.iteritems():
        alphas[m]     += aux[m][word]*val*doclengs[m]
        tmpmass[word] += aux[m][word]*val*doclengs[m]
      alphas[m] /= doclengs[m]

    # Estimate expanded query model
    qexpand = defaultdict(float)
    for word,val in prior.iteritems():
      qexpand[word] = mu * val
    for word,val in tmpmass.iteritems():
      qexpand[word] += tmpmass[word]

    # Normalize expanded model
    Z = 0.0
    for word,val in qexpand.iteritems():
      Z += val
    for word in qexpand.iterkeys():
      qexpand[word] = qexpand[word]/Z

    query = qexpand
    mu *= delta

  qsort = sorted(query.iteritems(),key=operator.itemgetter(1),reverse=True)
  query = dict(qsort[0:100])

  return query
