import sys
import os
import operator
from util import *

inv_index = readInvIndex(sys.argv[1])
oDir = sys.argv[2]

documents = {}
for i in range(1,5048,1):
    documents[i] = {}

for word, docs in inv_index.iteritems():
    for d,val in docs.iteritems():
	documents[d][word] = val

for docid, words in documents.iteritems():
    fname = oDir + IndexToDocName(docid)
    print 'Outputing :%s...' % fname
    fout = file(fname,'w')
    for wordid, val in words.iteritems():
	fout.write(str(wordid)+'\t'+str(val)+'\n')
    fout.close()

