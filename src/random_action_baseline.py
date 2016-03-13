import sys
from IR.environment import *
import cPickle as pickle
from sklearn.cross_validation import KFold
import numpy as np

recognitions = [ ('onebest','CMVN'),
                 ('onebest','tandem'),
                 ('lattice','CMVN'),
                 ('lattice','tandem') ]

rec_type = recognitions[0]
fold = 1

def setEnvironment():
  print 'Creating Environment and compiling State Estimator...'

  dir='../../ISDR-CMDP/'
  lex = 'PTV.lex'
  background = 'background/' + '.'.join(rec_type) + '.bg'
  inv_index = 'index/' + rec_type[0] + '/PTV.' + '.'.join(rec_type) + '.index'
  doclengs = 'doclength/' + '.'.join(rec_type) + '.length'
  docmodeldir = 'docmodel/' + '/'.join(rec_type) + '/'
  print '...done'
  return Environment(lex,background,inv_index,doclengs,docmodeldir,dir)

def load_data():
  newdir = '../Data/query/'
  print 'loading queries from ',newdir,'...'
  data = pickle.load(open(newdir+'data.pkl','r'))
  kf = KFold(163, n_folds=10)
  tr,tx = list(kf)[int(fold)-1]
  training_data = [ data[i] for i in tr ]
  testing_data = [ data[i] for i in tx ]
  print '...done'
  return training_data, testing_data


def random_action_baseline():
  filename =  'result/' + '.'.join(rec_type) + '_random_action_baseline.log'
  f = open(filename,'w')
  f.write('Index\tMAP\tReturn\n')

  env = setEnvironment()
  tr, tx = load_data()
  data = tx + tr
  repeat = 100
  EAPs = np.zeros(163)
  EReturns = np.zeros(163)

  for idx,(q, ans, ans_index) in enumerate(data):
    print 'Query ',idx
    APs = np.zeros(repeat)
    Returns = np.zeros(repeat)
    for i in xrange(repeat):
      cur_return = 0.

      terminal = False
      init_state = env.setSession(q,ans,ans_index,True)
      while( not terminal ):
        act = np.random.randint(5)
        reward, state = env.step(act)
        cur_return += reward
        terminal, AP = env.game_over()
      print AP,'\t',cur_return
      APs[i] = AP
      Returns[i] = cur_return
    EAPs[idx] = np.mean(APs)
    EReturns[idx] = np.mean(Returns)
    print '\n',EAPs[idx],'\t',EReturns[idx],'\n'
    f.write( '{}\t{}\t{}\n'.format(idx,EAPs[idx],EReturns[idx]) )
    f.flush()
  f.write('\nResults\n{}\t{}'.format( np.mean(EAPs),np.mean(EReturns) ) )
  f.close()
  print 'MAP : ',np.mean(EAPs),'\tReturn : ',np.mean(EReturns)

if __name__ == "__main__":
  random_action_baseline()
