import sys
import os
import operator
from util import *
from retrieval import *
from expansion import *

# python preliminary.py lexicon query back. inv_index docleng answer
lex = readLex(sys.argv[1])
queries = readQuery(sys.argv[2],lex)
background = readBackground(sys.argv[3],lex)
inv_index = readInvIndex(sys.argv[4])
doclengs = readDocLength(sys.argv[5])
answers = readAnswer(sys.argv[6],lex)
docmodeldir = sys.argv[7]

ret = batchRetrieve(queries,background,inv_index,doclengs,1000)
MAP,APs = evalMAP(ret,answers,True)
print 'Baseline exp\t=>\tMAP:\t%2.4f' % MAP

expanded_queries = PRFExpansion(queries,ret,doclengs,background,docmodeldir,10,10,10,1)

retx = batchRetrieve(expanded_queries,background,inv_index,doclengs,1000)
MAP, APs = evalMAP(retx,answers,True);
print 'PRF T10 exp\t=>\tMAP:\t%2.4f' % MAP


