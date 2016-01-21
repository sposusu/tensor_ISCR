import os, sys, operator, time
import numpy as np
from python.util import readFoldQueries,readLex,readInvIndex
from DQN import q_network
import DQN.agent as agent
from python.environment import *

################################
#import argparse
#parser = argparse.ArgumentParser(description='Interactive Retrieval')
#args = parser.parse_args()
dir='../../ISDR-CMDP/'
lex = 'PTV.lex'
train_data = '10fold/query/CMVN/train.fold1'
test_data = '10fold/query/CMVN/test.fold1'
background = 'background/onebest.CMVN.bg'
inv_index = 'index/onebest/PTV.onebest.CMVN.index'
#o = readInvIndex(dir+inv_index)
#print o
doclengs = 'doclength/onebest.CMVN.length'
answers = 'PTV.ans'
docmodeldir = 'docmodel/onebest/CMVN/'
train_queries,train_indexes = readFoldQueries(dir+train_data)
test_queries ,test_indexes  = readFoldQueries(dir+test_data)
###############################
input_width, input_height = [48,48]
num_actions = 10
phi_length = 4 # phi length?  input 4 frames at once
discount = 0.95
learning_rate = 0.00025
rms_decay = 0.99 # rms decay
rms_epsilon = 0.1
momentum = 0
clip_delta = 1.0
freeze_interval = 10000 #???  no freeze?
batch_size = 32
network_type = 'nature_cuda'
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
num_epoch = 100
step_per_epoch = 1000
num_tr_query = len(train_queries)
num_tx_query = len(test_queries)
###############################
class experiment():
  def __init__(self,agent,env):
    self.agent = agent
    self.env = env

  def run(self):
    epoch = 0
    it = 0
    while epoch < num_epoch:
      for q in train_queries:
        it = it + self.run_episode(q,False)

        if it % step_per_epoch == 0:
          epoch = epoch + 1
          self.testing()

  def testing(self):
    for qtest in test_queries:
      self.run_episode(qtest,True)
      print 'test','MAP = ','Total Reward = '


  def run_episode(self,queries,test_flag):
    init_state = self.env.setSession(train_queries[1],train_indexes[1])  # reset
    action = self.agent.start_episode(init_state)

    num_steps = 0

    while True:
      [reward, state] = self.env.step(action)
      terminal = self.env.game_over()
      num_steps += 1

      if num_steps >= max_steps:  # or terminal
        self.agent.end_episode(reward, terminal)
        break

      action = self.agent.step(reward, screen)
      print "Action :", action
    #if test_flag:

    return num_steps

def launch():
  t = time.time()
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
  print 'compile network .. done' , time.time()-t
  agt = agent.NeuralAgent(network,epsilon_start,epsilon_min,epsilon_decay,
                                  replay_memory_size,
                                  experiment_prefix,
                                  replay_start_size,
                                  update_frequency,
                                  rng)

  print 'create agent & simulator .. done'

  env = Environment(lex,background,inv_index,\
                    doclengs,answers,docmodeldir,dir)
  exp = experiment(agt,env)
  exp.run()

launch()
