import os
import sys
import operator
import time

#dataset = ['onebest','lattice']
feature = ['tandem']#,'tandem']
#feature = ['tandem']
dataset = ['lattice']
fold = [str(i) for i in range(1,11)]

#machine = ['thul','eth','tal','nef','ith','tir','ort','ral']
machine = ['thul','tal','ral','ort','ith','tir','thul','tal']

#gauss = ['10','8','5']
gauss = ['10']
vars = ['0.0625']#,'0.0625']
lamdas = ['0.01']#,'0.0001']

param = ['1000']

index = 0
for d in dataset:
    for f in feature:
	for p in param:
	    for g in gauss:
		for v in vars:
		    for l in lamdas:
			oDir = 'cmdp/exescripts/'+d+'.'+f+'/'
			for fd in fold:
			    fname = oDir + 'CMDP.fold'+fd+'.'+g+'.'+v+'.'+l+'.sh'
			    fout = file(fname,'w')
			    fout.write('cd $PBS_O_WORKDIR\n')
			    fout.write('TRAIN_Q=10fold/query/'+f+'/train.fold'+fd+'\n')
			    fout.write('TEST_Q=10fold/query/'+f+'/test.fold'+fd+'\n')
			    fout.write('BG=background/'+d+'.'+f+'.bg\n')
			    fout.write('INVIDX=index/'+d+'/PTV.'+d+'.'+f+'.index\n')
			    fout.write('DOCLEN=doclength/'+d+'.'+f+'.length\n')
			    fout.write('MODEL=docmodel/'+d+'/'+f+'/\n')
			    fout.write('PLY=cmdp/theta/'+p+'/'+d+'.'+f+'/theta.fold'+fd+'.'+g+'.'+v+'.'+l+'\n')
			    fout.write('LOG=cmdp/log/'+p+'/'+d+'.'+f+'/log.fold'+fd+'.'+g+'.'+v+'.'+l+'\n\n')
			    fout.write('python python/continuousMDP.py PTV.lex $TRAIN_Q $TEST_Q $BG $INVIDX $DOCLEN PTV.ans $MODEL $PLY '+g+' '+v+' '+l+' > $LOG\n')
			    fout.close()	
	    
			    cmd = 'qsub -l nodes='+machine[index]+' '+fname
			    #cmd = 'qsub '+fname
			    print cmd
			    os.system(cmd)
			    index += 1
			    if index>=len(machine):
				index = 0


