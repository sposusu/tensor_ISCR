import sys
import os
import operator
from util import *

lex = readLex(sys.argv[1])
answers = readAnswer(sys.argv[2],lex)
doclengs = readDocLength(sys.argv[3])
fout = file(sys.argv[4],'w')
docmodeldir = sys.argv[5]

for i in range(len(answers)):
    ans = answers[i]
    words = {}
    for docID in ans.iterkeys():
	fname = docmodeldir + IndexToDocName(docID)
	docmodel = readDocModel(fname)
	for term,prob in docmodel.iteritems():
	    if words.has_key(term):
		words[term] += prob*doclengs[docID]
	    else:
		words[term] = prob*doclengs[docID]
    
    fout.write(str(i)+'\tX\t')
    for word,cnt in words.iteritems():
	for c in range(int(round(cnt))):
	    fout.write(str(word)+' ')
    fout.write('\n')



