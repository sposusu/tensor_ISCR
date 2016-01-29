import cPickle as pickle
import operator
import os
import random
import sys
import time
import pdb


import numpy as np
import progressbar
from progressbar import ProgressBar, Percentage, Bar, ETA

from IR.util import readFoldQueries,readLex,readInvIndex
from DQN import q_network
import DQN.agent as agent
from IR.environment import *

##########################
#       filename         #
##########################

train_data = 'train.fold1.pkl'
test_data  = 'test.fold1.pkl'

################################
#import argparse
#parser = argparse.ArgumentParser(description='Interactive Retrieval')
#args = parser.parse_args()
dir='../../ISDR-CMDP/'
data_dir = '10fold/query/CMVN'

lex = 'PTV.lex'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'

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
testing_data = list2tuple(testing_data)

###############################
input_width, input_height = [89,1]
num_actions = 5
phi_length = 4 # phi length?  input 4 frames at once
discount = 0.95
learning_rate = 0.00025
rms_decay = 0.99 # rms decay
rms_epsilon = 0.1
momentum = 0
clip_delta = 1.0
freeze_interval = 10000 #???  no freeze?
batch_size = 32
network_type = 'rl_dnn'
update_rule = 'deepmind_rmsprop' # need update
batch_accumulator = 'sum'
rng = np.random.RandomState()
###############################
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 1000000
replay_memory_size = 1000000
experiment_prefix = 'result/ret'
replay_start_size = 50000
update_frequency = 4  #??
###############################
num_epoch = 1
step_per_epoch = 1000
max_steps = 5
num_tr_query = len(training_data)
num_tx_query = len(testing_data)

###############################
class experiment():
  def __init__(self,agent,env):
    self.agent = agent
    self.env = env

  def run(self):
    self.training()
    self.testing()

  def training(self):
    for epoch in range(num_epoch):
      print 'Running epoch {0} out of {1} epochs'.format(epoch+1,num_epoch)
      widgets = [ 'Training', Percentage(), Bar(), ETA() ]
      pbar = ProgressBar(widgets=widgets,maxval=num_tr_query).start()
      for idx, (q, ans, ans_index) in enumerate(training_data):
          self.run_episode(q,ans,ans_index,test_flag=False)
          pbar.update(idx)
      pbar.finish()
      random.shuffle(training_data)

  def testing(self):
    widgets = [ 'Testing', Percentage(), Bar(), ETA() ]
    pbar = ProgressBar(widgets=widgets,maxval=num_tx_query).start()
    for idx,(qtest,anstest,anstest_index) in enumerate(testing_data):
      self.run_episode(qtest,anstest,anstest_index,test_flag=True)
      pbar.update(idx)
    pbar.finish()
    random.shuffle(testing_data)

  def run_episode(self,q,ans,ans_index,test_flag = False):
    init_state = self.env.setSession(q,ans,ans_index,test_flag)  # reset
    action = self.agent.start_episode(init_state)
    #print 'action {0}'.format(action)
    num_steps = 0
    while True:
      reward, state = self.env.step(action)
      terminal = self.env.game_over()
      #print 'terminal {0}'.format(terminal)
      num_steps += 1

      if num_steps >= max_steps or terminal:
        self.agent.end_episode(reward, terminal)
        break

      action = self.agent.step(reward, state)

      #print 'action {0}'.format(action)
    #print 'num_steps {0}'.format(num_steps)
    return num_steps

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
