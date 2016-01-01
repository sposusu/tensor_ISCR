import sys
import os
import operator
import math
from util import *

inv_index = readInvIndex(sys.argv[1])
lengs = readDocLength(sys.argv[3])
odir = sys.argv[4]
for root, Dir, Files in os.walk(sys.argv[2]):
    for f in Files:
	fname = os.path.join(root,f)
	docmodel = readDocModel(fname)
	for key in docmodel.iterkeys():
	    docmodel[key] *= math.log(1+len(inv_index[key]))
    
	sorteds = sorted(docmodel.iteritems(),key=operator.itemgetter(1),reverse=True)

	f = docNameToIndex(fname.split('/')[-1])
	fout = file(odir+str(f),'w')
	for key,val in sorteds:
	    fout.write(str(key)+'\t'+str(val)+'\n')
	fout.close()
    

