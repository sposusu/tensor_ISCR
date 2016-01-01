import sys
import os
import operator
from util import *
import math

docmodeldir = sys.argv[1]
doclengs = readDocLength(sys.argv[2])
fout = file(sys.argv[3],'w')

for root, Dir, Files in os.walk(sys.argv[1]):
    for f in Files:
	fname = os.path.join(root,f)
	docID = docNameToIndex(fname.split('/')[-1])
	dataset = fname.split('/')[-3]
	scaling = 1
	if dataset=='lattice':
	    scaling = 10

	doc = readDocModel(fname)
	fout.write(IndexToDocName(docID)+'\tX\t')
	for term, val in doc.iteritems():
	    cnt = int(round(val*scaling*doclengs[docID]))
	    for i in range(cnt):
		fout.write(str(term)+' ')
	fout.write('\n')
fout.close()


