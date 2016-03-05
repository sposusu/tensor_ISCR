import cPickle as pickle
from collections import defaultdict
import datetime
import logging
import os
import pdb
import random
import time
from termcolor import cprint
print_red = lambda x: cprint(x, 'red')
print_blue = lambda x: cprint(x, 'blue')
print_yellow = lambda x: cprint(x, 'yellow')
import numpy as np
import progressbar
from progressbar import ProgressBar, Percentage, Bar, ETA

from DQN import q_network
import DQN.agent as agent
from IR.environment import *
from IR.util import readFoldQueries,readLex,readInvIndex

##########################
#       filename         #
##########################

recognitions = [ ('onebest','CMVN'), 
		             ('onebest','tandem'),
                 ('lattice','CMVN'),
                 ('lattice','tandem') ]

rec_type = recognitions[2]

train_data = 'train.fold1.pkl'
test_data  = 'test.fold1.pkl'

dir='../../ISDR-CMDP/'
#data_dir = '10fold/query/CMVN'
#answers = 'PTV.ans'

lex = 'PTV.lex'
#background = 'background/onebest.CMVN.bg'
background = 'background/' + '.'.join(rec_type) + '.bg'
#inv_index = 'index/onebest/PTV.onebest.CMVN.index'
inv_index = 'index/' + rec_type[0] + '/PTV.' + '.'.join(rec_type) + '.index'
#doclengs = 'doclength/onebest.CMVN.length'
doclengs = 'doclength/' + '.'.join(rec_type) + '.length'
#docmodeldir = 'docmodel/onebest/CMVN/'
docmodeldir = 'docmodel/' + '/'.join(rec_type) + '/'

newdir = '../Data/query/'

training_data = pickle.load(open(newdir+train_data,'r'))
testing_data  = pickle.load(open(newdir+test_data,'r'))

def list2tuple(data):
  result = []
  for idx in range(len(data[0])):
    result.append(tuple( (data[0][idx],data[1][idx],data[2][idx]) ))
  return result

training_data = list2tuple(training_data)
testing_data  = list2tuple(testing_data)
data = []
data.extend(testing_data)
data.extend(training_data)
###############################
input_width, input_height = [87,1]
#input_width, input_height = [164,1]
num_actions = 5

phi_length = 1 # phi length?  input 4 frames at once num_frames
discount = 1.
learning_rate = 0.00025
rms_decay = 0.99 # rms decay
rms_epsilon = 0.1
momentum = 0
clip_delta = 1.0
freeze_interval = 100 #???  no freeze?
batch_size = 1024
network_type = 'rl_dnn'
#network_type = 'linear'
update_rule = 'deepmind_rmsprop' # need update
batch_accumulator = 'sum'
rng = np.random.RandomState()
###############################
epsilon_start = 1.0
epsilon_min = 0.1
replay_memory_size = 100000
experiment_prefix = 'result/ret'
replay_start_size = 500
#replay_start_size = 1
update_frequency = 1
###############################
num_epoch = 80
epsilon_decay = num_epoch * 500
step_per_epoch = 1000
#step_per_epoch = 10

exp_name = 'batch_size_1024'
exp_name += '_feature_87_'

num_tr_query = len(training_data)
num_tx_query = len(testing_data)
num_query = len(data)
print "recognition type: ", rec_type
print "number of queries: ", num_query
print "number of trainig data: ", num_tr_query
print "number of testing data: ", num_tx_query
# TODO
# map -> ap  -- done
# action 2   -- done
# count num_steps -- done
# testing MAP,ans -- done (no state estimate)
# test print AP -- done (termcolor)
# test no random --  done (epsilon = 0,phi = 1,init episode no random)
# mix tr tx -- done
# test progress bar -- done
# check 4 baseline
# check dict copy?
# print action percetage -- done
# print best action seq
# overfit one query
# simulate platform
# accelerate
###############################
# logging
# TODO
# parametor -- done
# TRAIN : best actions seq, loss, epsilon
# TEST : action AP Return -- done
# filename exp name
exp_log_root = '../logs/'
try:
  os.makedirs(exp_log_root)
except:
  pass
cur_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")

exp_log_name = exp_log_root + exp_name + '_'.join(rec_type) + '_' + cur_datetime + ".log_deep_2_1024"

#exp_log_name = exp_log_root + '_'.join(rec_type) + '_' + cur_datetime + ".log"

logging.basicConfig(filename=exp_log_name,level=logging.DEBUG)

logging.info('learning_rate : %f', learning_rate)
logging.info('clip_delta : %f', clip_delta)
logging.info('freeze_interval : %f', freeze_interval)
logging.info('replay_memory_size : %d', replay_memory_size)

logging.info('num_epoch : %d', num_epoch)
logging.info('step_per_epoch : %d', step_per_epoch)
logging.info('epsilon_decay : %d', epsilon_decay)
logging.info('network_type : %s', network_type)
logging.info('input_width : %d', input_width)

print 'freeze_interval : ', freeze_interval
print 'replay_memory_size : ', replay_memory_size
print 'step_per_epoch : ', step_per_epoch
print 'network_type : ', network_type
print 'feature dimension : ', input_width
print 'exp_log_name : ', exp_log_name

###############################
class experiment():
  def __init__(self,agent,env):
    self.agent = agent
    self.env = env
    self.best_seq = {}
    self.best_return = np.zeros(num_query)

  def run(self):
    print_red( 'Init Model')
    self.agent.start_testing()
    self.run_epoch(True)
    self.agent.finish_testing(0)
    for epoch in xrange(num_epoch):
      print_red( 'Running epoch {0}'.format(epoch+1))
      logging.info('epoch {0}'.format(epoch))
      ## TRAIN ##
      self.run_epoch()
      self.agent.finish_epoch(epoch+1)
      
      ## TEST ##
      self.agent.start_testing()
      self.run_epoch(True)
      self.agent.finish_testing(epoch+1)
      random.shuffle(data)

  def run_epoch(self,test_flag=False):
    ## PROGRESS BAR SETTING
    setting = [['Training',step_per_epoch], ['Testing',num_query]]
    setting = setting[test_flag]
    widgets = [ setting[0] , Percentage(), Bar(), ETA() ]

    pbar = ProgressBar(widgets=widgets,maxval=setting[1]).start()
    APs = []
    Returns = []
    Losses = []
    self.act_stat = [0,0,0,0,0]

    steps_left = step_per_epoch
    while steps_left > 0:
      #if True:
      #  q, ans, ans_index = training_data[0]
      for idx,(q, ans, ans_index) in enumerate(data):
        logging.debug( 'ans_index {0}'.format(ans_index) )
        n_steps,AP = self.run_episode(q,ans,ans_index,steps_left,test_flag)
        steps_left -= n_steps

        if test_flag:
          pbar.update(idx)
          APs.append(AP)
          Returns.append(self.agent.episode_reward)
          logging.debug( 'Episode Reward : %f', self.agent.episode_reward )
        else:
          Losses.append(self.agent.episode_loss)
          pbar.update(step_per_epoch-steps_left)

        if self.agent.episode_reward > self.best_return[idx]:
          self.best_return[idx] = self.agent.episode_reward
          #self.best_seq[idx] = self.agent.act_seq
#          print 'query idx : ' + str(idx) + '\tbest_seq : '+ str(self.agent.act_seq) +'\treturn : ' + str(self.agent.episode_reward)

        if steps_left <= 0:
          break

      if test_flag:
        break
    pbar.finish()
    if test_flag:
      MAP,Return = [ np.mean(APs) , np.mean(Returns) ]
      print_yellow( 'MAP = '+str(MAP)+'\tReturn = '+ str(Return) )
      print_yellow( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'.format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )
      logging.info( 'MAP = %f\tReturn = %f',MAP,Return )
      
    else:
      Loss,BestReturn = [ np.mean(Losses), np.mean(self.best_return) ]
      print_blue( 'Loss = '+str(Loss)+'\tepsilon = '+str(self.agent.epsilon)+'\tBest Return = ' + str(BestReturn) )
      print_blue( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'.format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )
      logging.info( 'Loss = '+str(Loss)+'\tepsilon = '+str(self.agent.epsilon) )

  def run_episode(self,q,ans,ans_index,max_steps,test_flag = False):
    state = self.env.setSession(q,ans,ans_index,test_flag)  # Reset & First-pass
#    init_state = np.random.randn(1,89)*10000
#    print init_state
#    state = np.zeros((1,164))
#    state[0][ans_index] = 1.
#    state[0][-1] = 0

    action     = self.agent.start_episode(state)
    if test_flag and action != 4:
      logging.debug('action : -1 first pass\t\tAP : %f', self.env.dialoguemanager.MAP)

    num_steps = 0
    while True:
      reward, state = self.env.step(action)				# ENVIROMENT STEP
      terminal, AP = self.env.game_over()
      self.act_stat[action] += 1

      if test_flag :#and action != 4:
        AM = self.env.dialoguemanager.actionmanager
        logging.debug('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)
#        print('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)

      num_steps += 1
      if num_steps >= max_steps or terminal:  # STOP Retrieve
        self.agent.end_episode(reward, terminal)
        break

#      state = np.zeros((1,164))
#      state[0][ans_index] = 1.
#      state[0][-1] = num_steps
#      state = np.random.randn(1,89)
#      print state
      action = self.agent.step(reward, state)			# AGENT STEP
    return num_steps, AP

def launch():
  t = time.time()
  print 'Compiling Network...'
  network = q_network.DeepQLearner(input_width, input_height, num_actions,
                                         phi_length,
                                         discount,
                                         learning_rate,
                                         rms_decay,
                                         rms_epsilon,
                                         momentum,
                                         clip_delta,
                                         freeze_interval,
                                         batch_size,
                                         network_type,
                                         update_rule,
                                         batch_accumulator,
                                         rng)
  print 'Creating Agent and Simulator...'
  agt = agent.NeuralAgent(network,epsilon_start,epsilon_min,epsilon_decay,
                                  replay_memory_size,
                                  experiment_prefix,
                                  replay_start_size,
                                  update_frequency,
                                  rng)

  print 'Creating Environment and compiling State Estimator...'
  env = Environment(lex,background,inv_index,\
                    doclengs,docmodeldir,dir)
  print 'Initializing experiment...'
  exp = experiment(agt,env)
  print 'Done, time taken {} seconds'.format(time.time()-t)
  exp.run()

def test_action():
  env = Environment(lex,background,inv_index,\
                    doclengs,docmodeldir,dir)
  best_returns = defaultdict(float)
  best_seqs = defaultdict(list)
  for idx,(q, ans, ans_index) in enumerate(data):
    seqs = []
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

    for seq in seqs:
      print 'Running',seq
      cur_return = 0.
      init_state = env.setSession(q,ans,ans_index,True)
      for act in seq:
        reward, state = env.step(act)
        cur_return += reward
      terminal, AP = env.game_over()

      if cur_return > best_returns[idx]: 
        best_returns[idx] = cur_return
        best_seqs[idx] = seq

    print best_returns
    print best_seqs 
  
  with open('best_seq_return.pkl','w') as f:
	pickle.dump( (best_returns, best_seqs),f )    

if __name__ == "__main__":
  launch()
#  test_action()
