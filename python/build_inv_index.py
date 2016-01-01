import sys
import os
import operator
from util import *

idir = sys.argv[1]
fout = file(sys.argv[4],'w')

lst = readList(sys.argv[2])
lex = readLex(sys.argv[3])
cnt = len(lex)

emptyLst = []
for d in lex.iterkeys():
    emptyLst.append([])

inv_index = dict(zip(lex.values(),emptyLst))
for n in lst: 
    print 'indexing %s' % n
    fname = idir + n
    docID = docNameToIndex(n)
    fin = file(fname)
    doclen = int(float(fin.readline()))
    for pair in fin.readlines():
	if not lex.has_key(pair.split()[0]):
	    lex[pair.split()[0]] = cnt
	    cnt = len(lex)
	    inv_index[lex[pair.split()[0]]] = []
	    os.system('echo \''+pair.split()[0]+'\' >> '+sys.argv[3])
	wordID = lex[pair.split()[0]]
	val = pair.split()[1]
	inv_index[wordID].append(str(docID)+':'+val)
    fin.close()

sorted_inv = sorted(inv_index.iteritems(),key=operator.itemgetter(0))
for wordID,indexes in sorted_inv:
    fout.write(str(wordID)+'\t'+' '.join(indexes)+'\n')
fout.close()

