import unittest

from python import documentmodel
from python import util

dir='../../ISDR-CMDP/'
lex = 'PTV.lex'
train_data = '10fold/query/CMVN/train.fold1'
test_data = '10fold/query/CMVN/test.fold1'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
#o = readInvIndex(dir+inv_index)
#print o
doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'
docmodeldir = 'docmodel/onebest/CMVN/'


global l
def test_readLex():
    assert documentmodel.readLex(dir+lex) == util.readLex(dir+lex)
    l = documentmodel.readLex(dir+lex)

def test_readBackground():
    assert documentmodel.readBackground(dir+background,l) == util.readBackground(dir+background,l)

def test_readInvIndex():
    assert documentmodel.readInvIndex(dir+inv_index) == util.readInvIndex(dir+inv_index)

def test_readDocLength():
    assert documentmodel.readDocLength(dir+doclengs) == util.readDocLength(dir+doclengs)

def test_readAnswer():
    assert documentmodel.readAnswer(dir+answers,l) == util.readAnswer(dir+answers,l)
