import os
import sys
import operator
import time

feature = ['CMVN']#,'tandem']
dataset = ['lattice']

fold = [str(i) for i in range(1,11)]
machine = ['e','p','q','amd','i','j','k','l']
#machine = ['ith','tir','tal','eth','nef','thul']

sets = ['dev','test']
#sets = ['dev','test','train']

#machine = ['p','q','amd','e']
#machine = ['nef','eth','ith','ort','ral','tir','i','j','k','p','q','amd']

index = 0
for d in dataset:
    for f in feature:
	for s in sets:
	    oDir = 'SE/scripts/'+d+'.'+f+'/'
	    for fd in fold:
		fname = oDir + 'gen.'+s+'.fold'+fd+'.sh'
		fout = file(fname,'w')
		fout.write('cd $PBS_O_WORKDIR\n')
		fout.write('Q=10fold+dev/query/'+f+'/'+s+'.fold'+fd+'\n')
		fout.write('BG=background/'+d+'.'+f+'.bg\n')
		fout.write('INVIDX=index/'+d+'/PTV.'+d+'.'+f+'.index\n')
		fout.write('DOCLEN=doclength/'+d+'.'+f+'.length\n')
		fout.write('MODEL=docmodel/'+d+'/'+f+'/\n')
		fout.write('F=SE/features/'+d+'.'+f+'/'+s+'.feature.fold'+fd+'\n')
		fout.write('python python/gen_state_features.py PTV.lex $Q $BG $INVIDX $DOCLEN PTV.ans $MODEL $F\n')
		fout.close()	
	    
		cmd = 'qsub -l nodes='+machine[index]+' '+fname
		#cmd = 'qsub '+fname
		print cmd
		os.system(cmd)
		index += 1
		if index>=len(machine):
		    index = 0


