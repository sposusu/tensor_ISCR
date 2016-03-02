from collections import defaultdict
import math
import pdb

import h5py
import lasagne
import numpy as np
import theano
import theano.tensor as T
import random

from expansion import expansion
from retmath import *
from retrieval import retrieveCombination
from stateestimation import dnn
from util import IndexToDocName

"""
  Todo: Online training for approximator
  Review feature extraction in old code
"""

normalize_feature = False

se_prefix = '../Data/stateestimation/'
weights_file = se_prefix + 'dnn.npz'

class Approximator(object):
  def __init__(self):
    # Functions
    self.network, self.train_fn, self.val_fn, self.test_fn = dnn()

    with np.load( weights_file ) as f:
      param_values = [f['arr_%d' % i] for i in range(len(f.files))]
      lasagne.layers.set_all_param_values(self.network, param_values)


  def predict_one(self,feature):
    assert len(feature) == 89
    feature      = np.asarray(feature).reshape(1,1,1,89)
    estimatedMAP = float(self.test_fn(feature)[0])
    return estimatedMAP

class StateMachine(object):
  def __init__(self,background,inv_index,doclengs,dir,docmodeldir,\
              iteration=10,mu=1,delta=1,alpha=0.1):

    self.background = background
    self.inv_index  = inv_index
    self.doclengs   = doclengs

    self.dir         = dir
    self.docmodeldir = docmodeldir

    # Model for state estimation
    self.approximator = Approximator()

    # Mean and std for feature normalization
    with h5py.File("../Data/stateestimation/norm.h5") as norm:
      self.mean = norm['mean'][:]
      self.std = norm['std'][:]

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
    feature = self.featureExtraction(ret,action_type,curtHorizon,\
                                    posmodel,negmodel,posprior,negprior)
    estimatedMAP = 0.
#    estimatedMAP = self.approximator.predict_one(feature)

    feature = np.asarray(feature).reshape(1,len(feature))

    if normalize_feature:
      with np.errstate(divide='ignore', invalid='ignore'):
        feature = np.true_divide(feature-self.mean , self.std)
    	feature[feature == np.inf] = 0
    	feature = np.nan_to_num(feature)

    return feature, estimatedMAP

  def featureExtraction(self,ret,action_type,curtHorizon,\
                          posmodel,negmodel,posprior,negprior):
    feature = []
    my_feature = []
    # Extract Features
    docs = []
    doclengs = []
    k = 0

    # read puesdo rel docs
    for docID,score in ret:
      if k>=20:
        break
      model = readDocModel(self.dir + self.docmodeldir + IndexToDocName(docID))
      docs.append(model)
      doclengs.append(self.doclengs[docID])
      k += 1

    irrel = cross_entropies(posmodel,self.background) - \
            0.1 * cross_entropies(negmodel,self.background)

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
        WIG10 += (ret[i][1]-irrel)/10.0
        NQC10 += math.pow(ret[i][1]-irrel,2)/10.0
        for wordID,prob in doc.iteritems():
          ieDocT10[ wordID ] += prob * leng
      else:
        WIG20 += (ret[i][1]-irrel)/10.0
        NQC20 += math.pow(ret[i][1]-irrel,2)/10.0
        for wordID,prob in doc.iteritems():
          ieDocT20[ wordID ] += prob * leng

    ieDocT10 = renormalize(ieDocT10)
    ieDocT20 = renormalize(ieDocT20)

    # bias term
    # feature.append(1.0)

    # dialogue turn
    feature.append(float(curtHorizon))

    # prior entropy
    feature.append(entropy(posprior))

    # model entropy
    feature.append(entropy(posmodel))

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
    pos = expansion({},docs[:10],doclengs[:10],\
                          self.background,self.iteration,self.mu,self.delta)
    oret = retrieveCombination(pos,negprior,\
                              self.background,self.inv_index,self.doclengs,\
                              self.alpha,0.1)
    N = min([ret,oret])
    for i in range(len(Ns)):
      if N<Ns[i]:
        Ns[i]=N

    qf10 = 0.0
    qf20 = 0.0
    qf50 = 0.0
    for docID1,rel1 in oret[:Ns[0]]:
      for docID2,rel2 in ret[:Ns[0]]:
        if docID1==docID2:
          qf10 += 1.0
          break
    for docID1,rel1 in oret[:Ns[1]]:
      for docID2,rel2 in ret[:Ns[1]]:
        if docID1==docID2:
          qf20 += 1.0
          break
    for docID1,rel1 in oret[:Ns[2]]:
      for docID2,rel2 in ret[:Ns[2]]:
        if docID1==docID2:
          qf50 += 1.0
          break

    feature.append(qf10)
    feature.append(qf20)
    feature.append(qf50)

    # retrieved number
    #feature.append(len(ret))

    # query scope
    feature.append(QueryScope(posprior,self.inv_index))
    feature.append(QueryScope(posmodel,self.inv_index))

    # idf stdev
    feature.append(idfDev(posprior,self.inv_index))
    feature.append(idfDev(posmodel,self.inv_index))

    # cross entropy - prior, model, background
    feature.append(-1*cross_entropies(pos,posmodel))
    feature.append(-1*cross_entropies(posprior,posmodel))
    feature.append(-1*cross_entropies(posprior,self.background))
    feature.append(-1*cross_entropies(posmodel,self.background))
    del pos

    # IDF score, max, average
    maxIdf, avgIdf = IDFscore(posmodel,self.inv_index)
    feature.append(maxIdf)
    feature.append(avgIdf)

    # Statistics, top 5, 10, 20, 50, 100
    means, vars = Variability(ret,[5,10,20,50,100])
    for mean in means:
      feature.append(-1*mean)
    for var in vars:
      feature.append(var)

    # fit to exp distribution
    lamdas = [0.1,0.01,0.001]
    for lamda in lamdas:
      feature.append(FitExpDistribution(ret,lamda))

    # fit to gauss distribution
    Vars = [100,200,500]
    for v in Vars:
      feature.append(FitGaussDistribution(ret,0,v))

    # top N scores
    for key, val in ret[:49]:
      feature.append(-1*val)

    return feature


def readDocModel(fname):
  model = {}
  with open(fname) as fin:
    for line in fin.readlines():
      [ t1, t2 ] = line.split('\t')
      model[ int(t1) ] = float(t2)
  return model

if __name__ == "__main__":
  train_fn, test_fn = neuralnetwork()

  filepath = '../../Data/stateestimation/features_89.h5'
  h5f = h5py.File(filepath,'r')
  X_train = h5f['features'][:].reshape(37962,1,1,89)
  y_train = h5f['MAPs'][:].reshape(37962,1)
  train_fn(X_train[:2],y_train[:2])
