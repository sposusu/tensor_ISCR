import os
import sys
import operator
import time

#dataset = ['onebest','lattice']
feature = ['tandem']
dataset = ['lattice']
fold = [str(i) for i in range(1,11)]
machine = ['ral','nef','eth','ith','ort','tir','thul','tal','amd','k']
#machine = ['p','q','amd','i','j','k','l','e']


param = ['2000']

index = 0
for d in dataset:
    for f in feature:
	for p in param:
	    oDir = 'exescripts/'+d+'.'+f+'/'
	    for fd in fold:
		fname = oDir + 'DMDP.fold'+fd+'.sh'
		fout = file(fname,'w')
		fout.write('cd $PBS_O_WORKDIR\n')
		fout.write('TRAIN_Q=10fold/query/'+f+'/train.fold'+fd+'\n')
		fout.write('TEST_Q=10fold/query/'+f+'/test.fold'+fd+'\n')
		fout.write('BG=background/'+d+'.'+f+'.bg\n')
		fout.write('INVIDX=index/'+d+'/PTV.'+d+'.'+f+'.index\n')
		fout.write('DOCLEN=doclength/'+d+'.'+f+'.length\n')
		fout.write('MODEL=docmodel/'+d+'/'+f+'/\n')
		fout.write('PLY=policy/'+p+'/'+d+'.'+f+'/policy.fold'+fd+'\n')
		fout.write('LOG=log/'+p+'/'+d+'.'+f+'/log.fold'+fd+'\n\n')
		fout.write('python python/discreteMDP.py PTV.lex $TRAIN_Q $TEST_Q $BG $INVIDX $DOCLEN PTV.ans $MODEL $PLY > $LOG\n')
		fout.close()	
	    
		cmd = 'qsub -l nodes='+machine[index]+' '+fname
		#cmd = 'qsub '+fname
		print cmd
		os.system(cmd)
		index += 1
		if index>=len(machine):
		    index = 0


