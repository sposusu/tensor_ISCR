import operator
import os

from nose.tools import *

from python import searchengine
from python import dialoguemanager as dm
from python import util

dir='../../ISDR-CMDP/'
lex = 'PTV.lex'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'
docmodeldir = 'docmodel/onebest/CMVN/'


train_data = '10fold/query/CMVN/train.fold1'
test_data = '10fold/query/CMVN/test.fold1'
train_queries,train_indexes = util.readFoldQueries(dir+train_data)
test_queries ,test_indexes  = util.readFoldQueries(dir+test_data)

docs = os.listdir(dir+docmodeldir)

global cpsID
cpsId = '.'.join(docmodeldir.split('/')[1:-1])

#ans = searchengine.readAnswer(dir+answers,l)

def test_readLex():
  assert searchengine.readLex(dir+lex) == util.readLex(dir+lex)
  global l
  l = searchengine.readLex(dir+lex)

def test_readBackground():
  assert searchengine.readBackground(dir+background,l) == util.readBackground(dir+background,l)

def test_readInvIndex():
  assert searchengine.readInvIndex(dir+inv_index) == util.readInvIndex(dir+inv_index)

def test_readDocLength():
  assert searchengine.readDocLength(dir+doclengs) == util.readDocLength(dir+doclengs)

def test_readAnswer():
  assert searchengine.readAnswer(dir+answers,l) == util.readAnswer(dir+answers,l)
  global ans
  ans = searchengine.readAnswer(dir+answers,l)

def test_readKeyTermList():
  assert all( dm.readKeytermlist(cpsId,q) == util.readKeytermlist(cpsId,q) for q in train_queries )

def test_readRequestList():
  assert all( dm.readRequestlist(cpsId,ans[idx]) == util.readRequestlist(cpsId,ans[idx]) for idx in train_indexes )

def test_readTopicWords():
  assert dm.readTopicWords(cpsId) == util.readTopicWords(cpsId)

def test_readTopicList():
  assert all(dm.readTopicList(cpsId,idx)[:5] == util.readTopicList(cpsId,idx)[:5] for idx in train_indexes )

def test_readDocModel():
  assert all(dm.readDocModel(dir+docmodeldir+docID) == util.readDocModel(dir+docmodeldir+docID) for docID in docs )
