import os, sys, operator
from util import *
from MDP import *
from retrieval import *
from expansion import *
from action as act
from simulator import *

################################
import argparse
parser = argparse.ArgumentParser(description='Interactive Retrieval')
lex = readLex(sys.argv[1])
train_queries,train_indexes = readFoldQueries(sys.argv[2])
test_queries ,test_indexes  = readFoldQueries(sys.argv[3])

background = readBackground(sys.argv[4],lex)
inv_index = readInvIndex(sys.argv[5])
doclengs = readDocLength(sys.argv[6])
answers = readAnswer(sys.argv[7],lex)
docmodeldir = sys.argv[8]
args = parser.parse_args()
###############################
num_epoch = 100
step_per_epoch = 1000
num_tr_query = len(train_queries)
num_tx_query = len(test_queries)
###############################
def define_action_set():
  actionset = act.genActionSet()
  costTable = act.genCostTable()

class experiment():
  def __init__(self,agent,env):
    self.agent = agent
    self.env = env

  def run():
    epoch = 0
    it = 0
    while epoch < num_epoch:
      for i in xrange(train_queries):
        it = it + run_episode(q,False)

        if it % step_per_epoch == 0:
          epoch = epoch + 1
          testing()

  def testing():
    for i in xrange(test_queries):
      run_episode(q,False)
      print 'test'
    

  def run_episode(queries,test_flag):
      simulator.setSessionAnswer(answers[train_indexes[i]])

      mdp = MDP(deepcopy(simulator),actionset,dply,train_indexes[i],costTable)
      mdp.configIRMDP(train_queries[i],answers[train_indexes[i]],background,inv_index,doclengs,1000,docmodeldir,10,10,1)
      if test_flag:
        mdp.FittedValueIteration()
      return n_steps

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

  define_action_set()
  simulator = Simulator() # env = user simulator.py is modified
  env = 
  exp = experiment(agent,simulator)
  exp.run()

launch()
