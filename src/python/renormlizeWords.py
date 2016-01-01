import os
import sys
from util import *

topicWordList = readTopicWords(sys.argv[1])

for i in range(len(topicWordList)):
    N = sum(topicWordList[i].values())

    fout = file('/home/shawnwun/ISDR-CMDP/lda/'+sys.argv[1]+'/'+str(i),'w')
    for term, prob in topicWordList[i].iteritems():
	fout.write(str(term)+'\t'+str(prob/N)+'\n')
    fout.close()



