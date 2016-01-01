import sys
import os
import operator


fin = file(sys.argv[1])
odir = sys.argv[2]

lines = fin.readlines()

i = 0
for q in range(163):
    fout = file(odir + str(q),'w')
    ranking = {}
    for t in range(128):
	tokens = [float(val) for val in lines[i].split()]
	i+=1
	ranking[tokens[0]] = tokens[-1]
    for topic,improve in sorted(ranking.iteritems(),\
	    key=operator.itemgetter(1),reverse=True)[:20]:
	fout.write(str(topic)+'\t'+str(improve)+'\n')
    fout.close()



