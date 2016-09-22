import cPickle as pickle

import numpy as np

from DQN import agent
from IR.dialoguemanager import DialogueManager
from IR.searchengine import SearchEngine
from IR.human import SimulatedUser

recognitions = [ ('onebest','CMVN'),
                 ('onebest','tandem'),
                 ('lattice','CMVN'),
                 ('lattice','tandem') ]

rec_type = recognitions[0]

class AgentArgs:
  epsilon_start = 1.
  epsilon_min   = 0.1
  epsilon_decay = 100000
  experiment_prefix = 'result/ret'
  replay_memory_size = 10000
  replay_start_size = 500
  freeze_interval = 100
  update_frequency = 1
  rng = np.random.RandomState()

def get_agent():
  nn_file = '../Data/network/onebest_feature_87_epoch_50.pkl'
  handle = open(nn_file, 'r')
  network = pickle.load(handle)
  return agent.NeuralAgent(network,AgentArgs.epsilon_start,AgentArgs.epsilon_min,AgentArgs.epsilon_decay,
                                  AgentArgs.replay_memory_size,AgentArgs.experiment_prefix,AgentArgs.replay_start_size,
                                  AgentArgs.update_frequency,AgentArgs.rng)

def get_searchengine():
  dir='../../ISDR-CMDP/'
  lex = 'PTV.lex'
  background = 'background/' + '.'.join(rec_type) + '.bg'
  inv_index = 'index/' + rec_type[0] + '/PTV.' + '.'.join(rec_type) + '.index'
  doclengs = 'doclength/' + '.'.join(rec_type) + '.length'
  return SearchEngine(lex, background, inv_index, doclengs,dir)

def get_dialoguemanger():
  dir='../../ISDR-CMDP/'
  lex = 'PTV.lex'
  background = 'background/' + '.'.join(rec_type) + '.bg'
  inv_index = 'index/' + rec_type[0] + '/PTV.' + '.'.join(rec_type) + '.index'
  doclengs = 'doclength/' + '.'.join(rec_type) + '.length'
  docmodeldir = 'docmodel/' + '/'.join(rec_type) + '/'
  return DialogueManager(lex,background,inv_index,doclengs,dir,docmodeldir)

def get_simulator():
  dir='../../ISDR-CMDP/'
  docmodeldir = 'docmodel/' + '/'.join(rec_type) + '/'
  return Simulator(dir,docmodeldir)
