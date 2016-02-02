import cPickle as pickle
import os

from IR.util import readFoldQueries,readAnswer, readLex

dir='../../ISDR-CMDP/'
lex = 'PTV.lex'
#train_data = '10fold/query/CMVN/train.fold1'
#test_data = '10fold/query/CMVN/test.fold1'
#background = 'background/onebest.CMVN.bg'
#inv_index = 'index/onebest/PTV.onebest.CMVN.index'
#o = readInvIndex(dir+inv_index)
#print o
#doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'
#docmodeldir = 'docmodel/onebest/CMVN/'
#train_queries,train_indexes = readFoldQueries(dir+train_data)
#test_queries ,test_indexes  = readFoldQueries(dir+test_data)

data_dir = '10fold/query/CMVN/'

data_lex = readLex(dir+lex)
data_answers = readAnswer(dir+answers,data_lex)

newdir = '../Data/query/'
pkl = '.pkl'

def main():
  pass

def genQueryAnswer():
  for filename in os.listdir(dir+data_dir):
    with open( newdir + filename + pkl , 'w' ) as f:
        queries, indexes = readFoldQueries(dir+data_dir+filename)
        ans = [ data_answers[idx] for idx in indexes ]
        pickle.dump((queries,ans,indexes),f)

if __name__ == "__main__":
  main()
