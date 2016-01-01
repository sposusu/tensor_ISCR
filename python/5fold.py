import sys
import os
import operator
from util import *

lex = readLex(sys.argv[1])
queries = readQuery(sys.argv[2],lex)
odir = sys.argv[3]

qIndexes = [i for i in range(len(queries))]

# 5 fold CV
testSize = len(queries)/10+1
cvSize = len(queries)/10+1

for fold in range(0,10,1):
    start = fold*testSize
    # write test set
    writeQueryFold(odir+'test.fold'+str(fold+1),\
	    queries[start:start+testSize],qIndexes[start:start+testSize])
    
    """
    # write train set
    trainset = []
    trainIndex = []
    trainset.extend(queries[:start])
    trainIndex.extend(qIndexes[:start])
    trainset.extend(queries[start+testSize:])
    trainIndex.extend(qIndexes[start+testSize:])
    writeQueryFold(odir+'train.fold'+str(fold+1),trainset,trainIndex)
    """

    if fold!=9:
	# write dev set
	writeQueryFold(odir+'dev.fold'+str(fold+1),queries[start+testSize:start+2*testSize],qIndexes[start+testSize:start+2*testSize])
	trainset = []
	trainset.extend(queries[:start])
	trainset.extend(queries[start+2*testSize:])
	trainIndex = []
	trainIndex.extend(qIndexes[:start])
	trainIndex.extend(qIndexes[start+2*testSize:])
	# write train set
	writeQueryFold(odir+'train.fold'+str(fold+1),trainset,trainIndex)
    else:
	devset = queries[0:cvSize]
	devIndex = qIndexes[cvSize:start]
	trainset = queries[cvSize:start]
	trainIndex = qIndexes[cvSize:start]
	writeQueryFold(odir+'dev.fold'+str(fold+1),devset,devIndex)
	writeQueryFold(odir+'train.fold'+str(fold+1),trainset,trainIndex)



