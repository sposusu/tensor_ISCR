from collections import defaultdict
import math

import theano
import theano.tensor as T
import lasagne
import numpy as np

from expansion import expansion
from retmath import *
from retrieval import retrieveCombination
from util import IndexToDocName

"""
  Todo: build approximator (NN) for statemachine
"""

class Approximator(object):
  def __init__(self):
    pass
  def train_online(self):
    pass
  def train_offline(self):
    pass
  def predict(self):
    pass
"""
def Approximator():
  def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 89),
                             input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
                             l_in, num_units=100,
                             nonlinearity=lasagne.nonlinearities.rectify,
                             W=lasagne.init.GlorotUniform())

    l_out = lasagne.layers.DenseLayer(
                             l_hid1, num_units=1,
                             nonlinearity=lasagne.nonlinearities.softmax)
    return l_out
  return None
"""
class StateMachine(object):
  def __init__(self,background,inv_index,doclengs,dir,docmodeldir,\
              iteration=10,mu=1,delta=1,alpha=0.1):

    self.background = background
    self.inv_index = inv_index
    self.doclengs = doclengs
    self.dir = dir
    self.docmodeldir = docmodeldir

    # Model for state estimation
    self.approximator = Approximator()

    # Parameters for expansion
    self.iteration = iteration
    self.mu = mu
    self.delta = delta
    self.alpha = alpha

  def __call__(self,ret,action_type,curtHorizon,\
                posmodel,negmodel,posprior,negprior):
    """
      Return:
        feature: 1 dim vector
        estimatedMAP: average precision of retrieved result
    """

    self.ret = ret
    self.action_type = action_type
    self.curtHorizon = curtHorizon
    self.posmodel = posmodel
    self.negmodel = negmodel
    self.posprior = posprior
    self.negprior = negprior

    feature = self.featureExtraction()
    estimatedMAP = self.approximator.predict(feature)

    return feature, estimatedMAP

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
            0.1 * cross_entropies(self.negmodel,self.background)

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


def readDocModel(fname):
  model = {}
  with open(fname) as fin:
    for line in fin.readlines():
      [ t1, t2 ] = line.split('\t')
      model[ int(t1) ] = float(t2)
  return model
