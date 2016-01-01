import sys
import os
import operator
from util import *

fidx = sys.argv[1]
odir = sys.argv[2]
inv_index = readCleanInvIndex(fidx)

for term1 in inv_index.iterkeys():
    if os.path.isfile(odir+str(term1)):
	print 'Jump through %s' % str(term1)
	continue
    docs1 = inv_index[term1]
    jaccard = {}
    for term2 in inv_index.iterkeys():
	if term1==term2:
	    continue
	docs2 = inv_index[term2]
	jaccard[term2] = float( len(set(docs1)&set(docs2)) )\
		/float( len(set(docs1)|set(docs2)) )

    listToPrint = sorted(jaccard.iteritems(),key=operator.itemgetter(1),reverse=True)
    print 'Writing file :%s ...' % (odir+str(term1))
    fout = file(odir+str(term1),'w')
    for term,val in listToPrint[1:100]:
	if val!=0:
	    fout.write(str(term) + '\t' + str(val)+'\n')
    fout.close()

