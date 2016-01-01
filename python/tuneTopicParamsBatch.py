import os
import sys

dataset = sys.argv[1]
feature = sys.argv[2]
cpsID = dataset+'.'+feature

fname = '/home/shawnwun/ISDR-CMDP/exescripts/'+cpsID+'/'+\
	'qsub.'+cpsID+'.sh'
fsp = file(fname,'w')
fsp.write('cd $PBS_O_WORKDIR\n')
fsp.write('python python/tuneTopicParams.py PTV.lex '+\
	'PTV.qry.model.'+feature+' background/'+cpsID+'.bg '+\
	'index/'+dataset+'/PTV.'+cpsID+'.index '+\
	'doclength/'+cpsID+'.length PTV.ans '+\
	'docmodel/'+dataset+'/'+feature+'/ > tune/'+cpsID+'.tune\n')

fsp.close()
os.system('qsub '+fname)

