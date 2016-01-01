import sys
import os
import operator
from util import *

idir = sys.argv[1]
lst = readList(sys.argv[2])
fout = file(sys.argv[3],'w')

for n in lst:
    fname = idir + n
    fin = file(fname)
    doclen = fin.readline()
    fout.write(n+' '+doclen)
    fin.close()
fout.close()







