import pdb

from python import util
from python import retrieval
from python.searchengine import SearchEngine

dir = '../../ISDR-CMDP/'
lex = 'PTV.lex'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'
docmodeldir = 'docmodel/onebest/CMVN/'

# Queries
train_data = '10fold/query/CMVN/train.fold1'
test_data = '10fold/query/CMVN/test.fold1'

train_queries,train_indexes = util.readFoldQueries(dir+train_data)
test_queries ,test_indexes  = util.readFoldQueries(dir+test_data)

alpha = 1000
beta = 0.1

# Initialize Retrieval Engine
SE = SearchEngine(lex,background,inv_index,doclengs,answers,docmodeldir,dir)
SE(alpha,beta)

# Read data, using old code from util.py
lex = util.readLex(dir+lex)
background = util.readBackground(dir+background,lex)
inv_index = util.readInvIndex(dir+inv_index)
doclengs = util.readDocLength(dir+doclengs)
answers = util.readAnswer(dir+answers,lex)

def test_retrieve():
  assert all( SE.retrieve( q ) == retrieval.retrieve( q ,background,inv_index,doclengs,alpha) for q in train_queries )

def test_retrieve_neg():
  " I am too lazy to save negative models for testing. Haha"
  pass

if __name__ == "__main__":
  pass
