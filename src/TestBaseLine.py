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

train_data = 'train.fold1.pkl'
test_data  = 'test.fold1.pkl'

dir='../../ISDR-CMDP/'
#data_dir = '10fold/query/CMVN'
#answers = 'PTV.ans'

lex = 'PTV.lex'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
doclengs = 'doclength/onebest.CMVN.length'
docmodeldir = 'docmodel/onebest/CMVN/'

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
input_width, input_height = [89,1]
#input_width, input_height = [40,1]
num_actions = 5

phi_length = 1 # phi length?  input 4 frames at once num_frames
discount = 1.
learning_rate = 0.00025
rms_decay = 0.99 # rms decay
rms_epsilon = 0.1
momentum = 0
clip_delta = 1.0
freeze_interval = 750 #???  no freeze?
batch_size = 32
network_type = 'rl_dnn'
update_rule = 'deepmind_rmsprop' # need update
batch_accumulator = 'sum'
rng = np.random.RandomState()
###############################
epsilon_start = 1.0
epsilon_min = 0.1
replay_memory_size = 10000
experiment_prefix = 'result/ret'
replay_start_size = 500
#replay_start_size = 1
update_frequency = 1
###############################
num_epoch = 2
epsilon_decay = num_epoch * 500
step_per_epoch = 5000
#step_per_epoch = 10

num_tr_query = len(training_data)
num_tx_query = len(testing_data)
num_query = len(data)
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
# print action percetage
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
exp_log_name = exp_log_root + cur_datetime + ".log"
logging.basicConfig(filename=exp_log_name,level=logging.DEBUG)

logging.info('learning_rate : %f', learning_rate)
logging.info('clip_delta : %f', clip_delta)
logging.info('freeze_interval : %f', freeze_interval)
logging.info('replay_memory_size : %d', replay_memory_size)

logging.info('num_epoch : %d', num_epoch)
logging.info('step_per_epoch : %d', step_per_epoch)
logging.info('epsilon_decay : %d', epsilon_decay)

print 'freeze_interval : ', freeze_interval
print 'replay_memory_size : ', replay_memory_size
print 'step_per_epoch : ', step_per_epoch

Baseline_returns = defaultdict(list)
Baseline_APs = defaultdict(list)

###############################
class experiment():
  def __init__(self,agent,env):
    self.agent = agent
    self.env = env

  def run(self):
    
    actions = [ 0, 1, 2, 3 ]
   
    for act in actions: 
      print_red('Baseline Action {0}'.format(act))
      self.agent.start_testing() 
      self.run_epoch(True,act)
      self.agent.finish_testing(act)  
    """
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
    """

  def run_epoch(self,test_flag=False,action=0):
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
        #print 'query number {0}'.format(idx)
        logging.debug( 'ans_index {0}'.format(ans_index) )
        n_steps,AP = self.run_episode(q,ans,ans_index,steps_left,test_flag,action)
        steps_left -= n_steps

        if test_flag:
          pbar.update(idx)
          APs.append(AP)
          Returns.append(self.agent.episode_reward)
          logging.debug( 'Episode Reward : %f', self.agent.episode_reward )
        else:
          Losses.append(self.agent.episode_loss)
          pbar.update(step_per_epoch-steps_left)

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
      
      final_returns = []
      for idx in Baseline_APs.keys():
        print np.mean(Baseline_APs[idx])
        logging.info('Turn {0}, MAP {1}'.format(idx,np.mean(Baseline_APs[idx])))
        Baseline_APs[idx] = [] 
    else:
      Loss = np.mean(Losses)
      print_blue( 'Loss = '+str(Loss)+'\tepsilon = '+str(self.agent.epsilon) )
      print_blue( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'.format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )
      logging.info( 'Loss = '+str(Loss)+'\tepsilon = '+str(self.agent.epsilon) )

  def run_episode(self,q,ans,ans_index,max_steps,test_flag = False,action=0):
    init_state = self.env.setSession(q,ans,ans_index,test_flag)  # Reset & First-pass
#    init_state = np.random.rand(1,89)
#    action     = self.agent.start_episode(init_state)
#    print action, init_state
    self.agent.start_episode(init_state)
    if test_flag and action != 4:
      logging.debug('action : -1 first pass\t\tAP : %f', self.env.dialoguemanager.MAP)
    session_rewards = []
    session_APs = []
    num_steps = 0
    #while True:
    for steps in range(4):
      #print 'step number {0}'.format(self.env.dialoguemanager.curtHorizon)
      
      reward, state = self.env.step(action)				# ENVIROMENT STEP
      terminal, AP = self.env.game_over()
      #self.act_stat[action] += 1
      session_rewards.append(reward)
      session_APs.append(AP)
      """ 
      if test_flag and action != 4:
        AM = self.env.dialoguemanager.actionmanager
        logging.debug('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)
      num_steps += 1
      if num_steps >= max_steps or terminal:  # STOP Retrieve
        self.agent.end_episode(reward, terminal)
        break
      """
      #action = self.agent.step(reward, state)			# AGENT STEP
    max_idx = session_rewards.index(max(session_rewards))
    max_reward = session_rewards[max_idx]
    self.agent.end_episode(max_reward,True)
    AP = session_APs[max_idx] 
    
    for idx,r in enumerate(session_rewards):
	Baseline_APs[idx].append(session_APs[idx])	
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

if __name__ == "__main__":
  launch()
