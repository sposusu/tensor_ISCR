"""

  Action Manager is pulled out dialogue manager
  because it would be easier to add more actions in a separate file

"""

from collections import defaultdict
from copy import deepcopy
import json
import logging
import operator
import os

from util import  renormalize, pruneAndNormalize
import reader


########################
#      Cost Table      #
########################

# Define Cost Table
def genCostTable(survey):
  if not survey: # Human defined costs
    values = [ -30., -10., -50., -20., 0., 0., 1000. ]
  else: # Survey Values
    values = [ -23., -13., -19., -23., 0., 0., 1000. ]
  costTable = dict(zip(range(6)+['lambda'],values))
  return costTable

def genNoiseTable():
  values = [ 11., 7., 10., 11., 0.001, 0.001, 0001. ]
  noiseTable = dict(zip(range(6)+['lambda'],values))
  return noiseTable

def genActionTable():
  at = {}
  at[-1] = 'firstpass'
  at[0]  = 'doc'
  at[1]  = 'keyterm'
  at[2]  = 'request'
  at[3]  = 'topic'
  at[4]  = 'show'
  return at

##########################
#     Action Manager     #
##########################
class ActionManager(object):
    def __init__(self, background, doclengs, data_dir, survey,\
                    topicleng=100, topicnumword=500):

        self.background  = background
        self.doclengs    = doclengs

        topic_dir         = os.path.join(data_dir,'lda')
        self.topiclist    = reader.readTopicWords(topic_dir)

        self.docmodel_dir  = os.path.join(data_dir,'docmodel')

        # Parameters for expansion and action
        self.topicleng    = topicleng
        self.topicnumword = topicnumword

        # Since agent only returns integer as action
        self.actionTable  = genActionTable()
        self.costTable    = genCostTable(survey)
        self.noiseTable   = genNoiseTable()

    def __call__(self, query):
        self.posmodel = deepcopy(query)
        self.negmodel = None

        self.posdocs  = []
        self.poslengs = []
        self.negdocs  = []
        self.neglengs = []

        self.posprior = deepcopy(self.posmodel)
        self.negprior = {}

    def expand_query(self,params):
        assert isinstance(params,dict),params

        action = params['action']

        if action == 'firstpass':
            query = params['query']
            self.__call__(query)

        elif action == 'doc':
            doc = params['doc']
            if doc is not None:
                doc = int(doc)
                self.posdocs.append(os.path.join(self.docmodel_dir,reader.IndexToDocName(doc)))
            	self.poslengs.append(self.doclengs[doc])

        elif action == 'keyterm':
            if 'keyterm' in params: # len(keytermlist) is not 0
                keyterm = int(params['keyterm'])
                isrel = params['isrel'] == 'True'
                if isrel:
                    self.posprior[ keyterm ] = 1.
                else:
                    self.negprior[ keyterm ] = 1.

        elif action == 'request':
            request = int(params['request'])
            self.posprior[ request ] = 1.0

        elif action == 'topic':
            if params['topic'] is not None:
                topicIdx = int(params['topic'])
                self.topiclist[ topicIdx ]
                self.posdocs.append(pruneAndNormalize(self.topiclist[ topicIdx ],self.topicnumword))
                self.poslengs.append(self.topicleng)
        else:
            raise ValueError("Invalid action {}".format(action))

        posmodel = self.expansion(renormalize(self.posprior),self.posdocs,self.poslengs,self.background)
        negmodel = self.expansion(renormalize(self.negprior),self.negdocs,self.doclengs,self.background)

        self.posmodel = posmodel
        self.negmodel = negmodel

        return posmodel, negmodel

    def expansion(self,prior,docnames,doclengs,back,iteration=10,mu=10,delta=1):
        models = []
        alphas = []

        for name in docnames:
            if isinstance(name, str) or isinstance(name,int):
                docmodel_path = name
                models.append(reader.readDocModel(docmodel_path))
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
