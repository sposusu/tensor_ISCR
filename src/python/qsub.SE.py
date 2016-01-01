import os
import sys
import operator
import time

feature = ['CMVN','tandem']
dataset = ['onebest']

fold = [str(i) for i in range(1,11)]
#machine = ['ral','ral','ral','ith','tir','tal','eth','nef']

machine = ['ort','ith','nef','tir','eth']
#machine = ['nef','eth','ith','ort','ral','tir','i','j','k','p','q','amd']

index = 0
for d in dataset:
    for f in feature:
	oDir = 'SE/scripts/'+d+'.'+f+'/'
	for fd in fold:
	    fname = oDir + 'SE.fold'+fd+'.sh'
	    fout = file(fname,'w')
	    fout.write('cd $PBS_O_WORKDIR\n')
	    fout.write('train=SE/features/'+d+'.'+f+'/train.feature.fold'+fd+'\n')
	    fout.write('dev=SE/features/'+d+'.'+f+'/dev.feature.fold'+fd+'\n')
	    fout.write('test=SE/features/'+d+'.'+f+'/test.feature.fold'+fd+'\n')
	    
	    fout.write('log=SE/log/'+d+'.'+f+'/log.fold'+fd+'\n')
	    fout.write('python python/state_estimate.py $train $dev $test > $log\n')
	    fout.close()	
	    
	    cmd = 'qsub -l nodes='+machine[index]+' '+fname
	    #cmd = 'qsub '+fname
	    print cmd
	    os.system(cmd)
	    index += 1
	    if index>=len(machine):
		index = 0


