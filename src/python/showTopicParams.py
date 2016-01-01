import os
import sys

cpsID = sys.argv[1]

topicleng = [10,20,50,100,200]
topicnumwords = [10,20,50,80,100,150,200,300,400,500,1000]

for tlng in topicleng:
    for twrd in topicnumwords:
	fname = '/home/shawnwun/ISDR-CMDP/tune/' + cpsID + '.'+\
		str(tlng)+'.'+str(twrd)
	
	fin = file(fname)
	lines = fin.readlines()
	
	print lines[-1],
	



