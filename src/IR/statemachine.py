from collections import defaultdict
import math
import logging
import os

import numpy as np
import random

from expansion import expansion
from retmath import *
from retrieval import retrieveCombination
import reader

class StateMachine(object):
  def __init__(self,background,inv_index,doclengs,data_dir,\
              iteration=10,mu=1,delta=1,alpha=0.1,feat="all"):

    self.background     = background
    self.inv_index      = inv_index
    self.doclengs       = doclengs
    self.docmodel_dir   = os.path.join(data_dir,'docmodel')

    # Parameters for expansion
    self.iteration = iteration
    self.mu        = mu
    self.delta     = delta
    self.alpha     = alpha

    # feature
    assert feat in ["all","raw","wig","nqc"]
    self.feat = feat
    if feat == "all":
      self.feat_len = 87
    elif feat == "raw":
      self.feat_len = 49
    elif feat == "wig" or feat == "nqc":
      self.feat_len = 5

    #print("feature length = {}".format(self.feat_len))

  def __call__(self,ret,action_type,curtHorizon,\
                posmodel,negmodel,posprior,negprior):
    """
      Return:
        feature: 1 dim vector
        estimatedMAP: average precision of retrieved result
    """
    feature = self.featureExtraction(ret,action_type,curtHorizon,\
                                    posmodel,negmodel,posprior,negprior)

    feature = np.asarray(feature).reshape(1,len(feature))

    return feature

  def featureExtraction(self,ret,action_type,curtHorizon,\
                          posmodel,negmodel,posprior,negprior):

    top_scores = [ -1 * x[1] for x in ret[:49] ]
    if self.feat == "raw":
        return top_scores
    elif self.feat == "wig" or self.feat == "nqc":
        metrics = []
        # read puesdo rel docs
        docs = []
        doclengs = []
        for docID,score in ret[:50]:
            docmodel_path = os.path.join(self.docmodel_dir,reader.IndexToDocName(docID))
            model = reader.readDocModel(docmodel_path)

            docs.append(model)
            doclengs.append(self.doclengs[docID])

        # Calculate wig & nqc
        irrel = cross_entropies(posmodel,self.background) - \
            0.1 * cross_entropies(negmodel,self.background)

        # Calculate at 10, 20, 30, 40 ,50
        cal_num_list = [0,10,20,30,40,50]
        for start, end in zip( cal_num_list[:-1], cal_num_list[1:] ):
            metric = 0.
            for r in ret[start:end]:
                if self.feat == "wig":
                    metric += ( r[1] - irrel ) / 10.
                else:
                    metric += math.pow(r[1]-irrel,2) / 10.
                metrics.append(metric)
        return metrics

    feature = []

    # Extract Features
    docs = []
    doclengs = []
    k = 0

    # read puesdo rel docs
    for docID,score in ret:
      if k>=20:
        break
      #model = self.docmodels[IndexToDocName(docID)]
      docmodel_path = os.path.join(self.docmodel_dir, reader.IndexToDocName(docID))
      model = reader.readDocModel(docmodel_path)

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
    top_scores = [ -1 * x[1] for x in ret[:49] ]
    feature += top_scores

    return feature
