import os
import sys
import operator
import time

#dataset = ['onebest','lattice']
feature = ['tandem']#,'tandem']
dataset = ['lattice']
#fold = [str(i) for i in range(1,11)]
machine = ['ral','tal','thul','tir','ort','ith','ith','nef']

gauss = ['10']
vars = ['0.0625']
lamdas = ['0.01']

param = ['1000']

iteration = ['50','100','200','500','1000','2000','5000','10000']

index = 0

for iter in iteration:
    for d in dataset:
	for f in feature:
	    for p in param:
		for g in gauss:
		    for v in vars:
			for l in lamdas:
			    
			    oDir = 'cmdp/exescripts/'+d+'.'+f+'/'
			    
			    fname = oDir + 'testCMDP.'+iter+'.sh'
			    fout = file(fname,'w')
			    fout.write('cd $PBS_O_WORKDIR\n')
			    fout.write('Q_FOLD=10fold/query/'+f+'/\n')
			    fout.write('BG=background/'+d+'.'+f+'.bg\n')
			    fout.write('INVIDX=index/'+d+'/PTV.'+d+'.'+f+'.index\n')
			    fout.write('DOCLEN=doclength/'+d+'.'+f+'.length\n')
			    fout.write('MODEL=docmodel/'+d+'/'+f+'/\n')
			    fout.write('PLY=cmdp/theta/'+p+'/'+d+'.'+f+'/\n')
			    fout.write('SE=SE/theta/'+d+'.'+f+'/\n\n')
			    fout.write('LOG=testing/CMDP/'+p+'/'+d+'.'+f+'/log.'+iter+'\n\n')
			    fout.write('python python/testContinuous.py PTV.lex $Q_FOLD $BG $INVIDX $DOCLEN PTV.ans $MODEL $PLY $SE '+g+' '+v+' '+l+' '+iter+' > $LOG\n')
			    fout.close()	
	    
			    cmd = 'qsub -l nodes='+machine[index]+' '+fname
			    #cmd = 'qsub '+fname
			    print cmd
			    os.system(cmd)
			    index += 1
			    if index>=len(machine):
				index = 0


