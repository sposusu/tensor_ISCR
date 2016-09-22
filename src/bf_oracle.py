import sys
from IR.environment import *
import cPickle as pickle
from sklearn.cross_validation import KFold
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

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

tr,tx = load_data()
data = tx + tr

def get_seqs():
  seqs = [[4]]
  # 1
  for a in xrange(4):
    seqs.append([a])

  # 2
  for a in xrange(4):
    for b in xrange(4):
      seqs.append([a,b])

  # 3
  for a in xrange(4):
    for b in xrange(4):
      for c in xrange(4):
        seqs.append([a,b,c])

  # 4
  for a in xrange(4):
    for b in xrange(4):
      for c in xrange(4):
        for d in xrange(4):
          seqs.append([a,b,c,d])
  return seqs

best_returns = - np.ones(163)
best_seqs = defaultdict(list)
APs = np.zeros(163)
seqs = get_seqs()
env = setEnvironment()
logname = 'result/' + '.'.join(rec_type) + '_best_seq_return.log'
lf = open(logname,'w')
lf.write('Index\tBest sequence\tBest return\tAP\n')
lf.flush()
def test_one_action(idx):
  q, ans, ans_index = data[idx]
  for seq in seqs:
    cur_return = 0.
    init_state = env.setSession(q,ans,ans_index,True)
    for act in seq:
      reward, state = env.step(act)
      cur_return += reward
    terminal, AP = env.game_over()
    sys.stderr.write('\rQuery {}    Actions Sequence {}    Return = {}'.format(idx,seq,cur_return))

    if cur_return > best_returns[idx]:
      best_returns[idx] = cur_return
      best_seqs[idx] = seq
      APs[idx] = AP
  lf.write( '{}\t{}\t{}\t{}\n'.format(idx, best_seqs[idx], best_returns[idx], APs[idx]))
  lf.flush()
  return best_seqs[idx],best_returns[idx],APs[idx]

def test_action():
  pool = Pool(10)
  for i in xrange(16):
    print i,'0~',i,'9'
    print pool.map(test_one_action, range(i*10,(i+1)*10))
  print '160~162'
  print pool.map(test_one_action, range(160,163))
 
  filename = 'result/' + '.'.join(rec_type) + '_best_seq_return.pkl'
  with open(filename,'w') as f:
    pickle.dump( (best_returns, best_seqs,APs),f )
  print 'MAP = ', np.mean(APs),'Return = ',np.mean(best_returns)

if __name__ == "__main__":
  test_action()
