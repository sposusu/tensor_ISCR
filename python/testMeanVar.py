import sys
import os
import operator
from util import *


idir = sys.argv[1]
odir = sys.argv[2]


for f in range(1,11):
    
    features,labels = readStateFeatures(idir+'train.feature.fold'+str(f))

    means = [0.0 for x in features[0]]
    twonorms = [0.0 for x in features[0]]
    devs = [0.0 for x in features[0]]
    
    for feature in features:
	for i in range(len(feature)):
	    means[i] += feature[i]
	    twonorms[i] += math.pow(feature[i],2)

    for i in range(len(devs)):
	means[i] = means[i]/float(len(features))
	twonorms[i] = twonorms[i]/float(len(features))
	devs[i] = math.sqrt(twonorms[i]-math.pow(means[i],2))

    fout = file(odir+'mean.fold'+str(f),'w')
    for mean in means:
	fout.write(str(mean)+' ')
    fout.write('\n')
    fout.close()
    
    fout = file(odir+'dev.fold'+str(f),'w')
    for dev in devs:
	fout.write(str(dev)+' ')
    fout.write('\n')
    fout.close()



