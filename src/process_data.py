import cPickle as pickle
import os
import pdb

import h5py
import numpy

from IR.util import readFoldQueries,readAnswer, readLex

dir='../../ISDR-CMDP/'
lex = 'PTV.lex'

answers = 'PTV.ans'

data_dir = '10fold/query/CMVN/'

data_lex = readLex(dir+lex)
data_answers = readAnswer(dir+answers,data_lex)

newdir = '../Data/query/'
pkl = '.pkl'

def main():
  fin = '../Data/stateestimation/' + 'feature_89.pkl'
  fout = '../Data/stateestimation/' + 'feature_89_epoch_1000.h5'
  removeDuplicateFeatures(fin,fout)

def genQueryAnswer():
  for filename in os.listdir(dir+data_dir):
    with open( newdir + filename + pkl , 'w' ) as f:
        queries, indexes = readFoldQueries(dir+data_dir+filename)
        ans = [ data_answers[idx] for idx in indexes ]
        pickle.dump((queries,ans,indexes),f)

def removeDuplicateFeatures(fin,fout):
  features, MAPs = pickle.load(open(fin))

  data = zip(features,MAPs)
  result = []
  for d in data:
    if d not in result:
      result.append(d)

  try:
    indep_features, indep_MAPs = result

    features = np.asarray(indep_features)
    MAPs = np.asarray(indep_MAPs)


    h5f = h5py.File(fout,'w')
    h5f.create_dataset('features',data=features)
    h5f.create_dataset('MAPs',data=MAPs)
    h5f.close()
  except:
    pdb.set_trace()


if __name__ == "__main__":
  main()
