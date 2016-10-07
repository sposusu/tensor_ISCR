

import cPickle as pickle
import numpy as np
import datetime,logging
import os,sys,pdb,random,time
import progressbar,argparse
from termcolor import cprint
from progressbar import ProgressBar,Percentage,Bar,ETA
from collections import defaultdict

from DQN import q_network
from DQN import agent
from IR.environment import *
from IR.dialoguemanager import DialogueManager
from IR.human import SimulatedUser
# util IO
from IR.util import readFoldQueries,readLex,readInvIndex
from sklearn.cross_validation import KFold
###############################
# online platform
# accelerate GPU?
# deep retrieval
# combine retrieval.py to search_engine
####### term color #########
def print_red(x):  # epoch
  cprint(x, 'red')
  logging.info(x)
def print_blue(x): # train info
  cprint(x, 'blue')
  logging.info(x)
def print_yellow(x): # test info
  cprint(x, 'yellow')
  logging.info(x)
def print_green(x):  # parameter
  cprint(x, 'green')
  logging.info(x)
########### argparse #########
parser = argparse.ArgumentParser(description='Interactive Retrieval')
# retrieval module
parser.add_argument("-t", "--type", type=int, help="recognitions type", default=0)
parser.add_argument("-f", "--fold", type=int, help="fold 1~10", default=-1)
parser.add_argument("--prefix", help="experiment name prefix",default="")   # store in folder
  # state machine
parser.add_argument("--feature", help="feature type (all/raw)", default="all") # TODO not implement yet
parser.add_argument("--normalize", help="normalize feature", action="store_true") # TODO not implement yet

# simulated user
parser.add_argument("--action_cost", help="action cost", default="type1") # TODO not implement yet
parser.add_argument("--topic_prob", help="topic action with probability", action="store_true") #TODO
parser.add_argument("--keyterm_thres", type=float,help="topic action with probability", default=0.5)
parser.add_argument("--cost_std", type=int, help="action cost with noise", default=1)
parser.add_argument("-nsu","--new_simulated_user", help="simulated_user according to questionnaire", action="store_true") # TODO

# experiment
parser.add_argument("--num_epoch", help="number of epoch",default=80)
parser.add_argument("--step_per_epoch", help="number of step per epoch",default=1000) # 25,0000
parser.add_argument("--test", help="testing mode",action="store_true")
parser.add_argument("--nolog", help="don't save log file",action="store_true")
parser.add_argument("-nn","--nn_file", help="pre-trained model")
parser.add_argument("--demo", help="demo mode",action="store_true")

# neural network
parser.add_argument("--model_width", type=int, help="model width", default=1024)
parser.add_argument("--model_height", type=int, help="model height", default=2)
parser.add_argument("--batch_size", type=int, help="batch size", default=256)
parser.add_argument("-lr","--learning_rate", type=float, help="learning rate", default=0.00025)
parser.add_argument("--clip_delta", type=float, help="clip delta", default=1.0)
parser.add_argument("--update_rule", help="deepmind_rmsprop/rmsprop/adagrad/adadelta/sgd", default="deepmind_rmsprop")
# reinforce
parser.add_argument("--replay_start_size", type=int, help="replay start size", default=500)   # 5,0000
parser.add_argument("--replay_memory_size", type=int, help="replay memory size", default=10000)   # 100,0000
parser.add_argument("--epsilon_decay", type=int, help="epsilon decay", default=100000)  # 100,0000
parser.add_argument("--epsilon_min", type=float, help="epsilon min", default=0.1)
parser.add_argument("--epsilon_start", type=float, help="epsilon start", default=1.0)
parser.add_argument("--freeze_interval", type=int, help="freeze interval", default=100)   # 10000
parser.add_argument("--update_frequency", type=int, help="update frequency", default=1)

args = parser.parse_args()
##########################
#       SETTING          #
##########################
# SET Logging
recognitions = [ ('onebest','CMVN'),
                 ('onebest','tandem'),
                 ('lattice','CMVN'),
                 ('lattice','tandem') ]

rec_type = recognitions[args.type]

exp_log_root = os.path.join('../result/',args.prefix)
exp_log_name = os.path.join(exp_log_root,'_'.join(rec_type) + '_fold{}'.format(str(args.fold)) + '.log')
if not os.path.exists(exp_log_root):
  os.makedirs(exp_log_root)

if args.nolog or args.demo:
  print_green('No log file')
else:
  logging.basicConfig(filename=exp_log_name,level=logging.DEBUG)
  print_green('exp_log_name : {}'.format(exp_log_name))

def setRetrievalModule():
  print 'Creating Environment with Retrieval Module and Simulated User...'
  retrievalmodule = DialogueManager(
                      lex         = 'PTV.lex',
                      background  = 'background/' + '.'.join(rec_type) + '.bg',
                      inv_index   = 'index/' + rec_type[0] + '/PTV.' + '.'.join(rec_type) + '.index',
                      doclengs    = 'doclength/' + '.'.join(rec_type) + '.length',
                      dir         = '../data/ISDR-CMDP/',
                      docmodeldir = 'docmodel/' + '/'.join(rec_type) + '/',
                      feat        = args.feature
                      )
  simulateduser = SimulatedUser(
                      dir           = '../data/ISDR-CMDP/',
                      docmodeldir   = 'docmodel/' + '/'.join(rec_type) + '/',
                      keyterm_thres = args.keyterm_thres,
                      topic_prob    = args.topic_prob,
                      survey        = args.new_simulated_user
                      )
  env = Environment(retrievalmodule,simulateduser)
  return env

"""
Update Rules:
Can combine with momentum ( default: 0.9 )
1. momentum
2. nesterov_momentum
Note: Can only set one type of momentum
"""
def setDialogueManager(env):
  print 'Creating Agent with Compiled Q Network...'
  experiment_prefix = '../result/'+args.prefix+'/model'  # TODO fix it save with log (result has logs and models)
  rng = np.random.RandomState()
  if args.nn_file is None:
    input_height = 1              # change feature
    input_width = env.retrievalmodule.statemachine.feat_len
    num_actions = 5
    phi_length = 1 # input 4 frames at once num_frames
    discount = 1.
    rms_decay = 0.99
    rms_epsilon = 0.1
    momentum = 0.
    nesterov_momentum = 0.
    network_type = 'rl_dnn'
    batch_accumulator = 'sum'
    network = q_network.DeepQLearner(input_width, input_height, args.model_width, args.model_height,
                                      num_actions,
                                         phi_length,discount,args.learning_rate,rms_decay,
                                         rms_epsilon,momentum,nesterov_momentum,
                                         args.clip_delta,args.freeze_interval,args.batch_size,
                                         network_type, args.update_rule
                                      , batch_accumulator
                                      , rng
                                    )
  else:
    print 'Loading Pre-trained Network...'
    handle = open(args.nn_file, 'r')
    network = pickle.load(handle)

  return agent.NeuralAgent(network,args.epsilon_start,args.epsilon_min,args.epsilon_decay,
                                  args.replay_memory_size,experiment_prefix,args.replay_start_size,
                                  args.update_frequency,rng)

def load_query(newdir_pickle='../data/query/data.pkl'):
  print('Loading queries from {}'.format(newdir_pickle))
  with open(newdir_pickle,'r') as f:
      data = pickle.load(f)

  if args.fold == -1:
    print_green( 'train = test = all queries')
    return data,data

  kf = KFold(163, n_folds=10)

  tr,tx = list(kf)[args.fold-1]

  training_data = [ data[i] for i in tr ]
  testing_data = [ data[i] for i in tx ]

  return training_data, testing_data

############ LOGGING ###################
def setLogging():
  print_green('learning_rate : {}'.format(args.learning_rate))
  print_green('batch_size : {}'.format(args.batch_size))
  print_green('clip_delta : {}'.format(args.clip_delta))
  print_green('freeze_interval : {}'.format(args.freeze_interval))
  print_green('replay_memory_size : {}'.format(args.replay_memory_size))
  print_green('num_epoch : {}'.format(args.num_epoch))
  print_green('step_per_epoch : {}'.format(args.step_per_epoch))
  print_green('epsilon_decay : {}'.format(args.epsilon_decay))
  #print_green('input_width(feature length) : {}'.format(input_width))
  print_green("network shape: {}".format([args.model_width,args.model_height]))

###############################
class experiment():
  def __init__(self,agent,env):
    print('Initializing experiment...')
    setLogging()
    self.agent = agent
    self.env = env
    self.best_seq = {}
    self.best_return = np.zeros(163) # to be removed
    self.training_data, self.testing_data = load_query()

  def run(self):
    print_red('Init Model')

    self.agent.start_testing()
    self.run_epoch(test_flag=True)
    self.agent.finish_testing(0)
    if args.test:
      return
    for epoch in range(1,int(args.num_epoch)+1,1):
      print_red('Running epoch {0}'.format(epoch))
      random.shuffle(self.training_data)

      ## TRAIN ##
      self.run_epoch()
      self.agent.finish_epoch(epoch)

      ## TEST ##
      self.agent.start_testing()
      self.run_epoch(True)
      self.agent.finish_testing(epoch)

  def run_epoch(self,test_flag=False):
    epoch_data = self.training_data
    if(test_flag):
      epoch_data = self.testing_data
    print('Number of queries {}'.format(len(epoch_data)))

    ## PROGRESS BAR SETTING
    setting = [['Training',args.step_per_epoch], ['Testing',len(epoch_data)]]
    setting = setting[test_flag]
    widgets = [ setting[0] , Percentage(), Bar(), ETA() ]

    pbar = ProgressBar(widgets=widgets,maxval=setting[1]).start()
    APs = []
    Returns = []
    Losses = []
    self.act_stat = [0,0,0,0,0]

    steps_left = args.step_per_epoch

    while steps_left > 0:
      #if True:
      #  q, ans, ans_index = training_data[0]
      for idx,(q, ans, ans_index) in enumerate(epoch_data):
        logging.debug( 'ans_index {0}'.format(ans_index) )
        n_steps,AP = self.run_episode(q,ans,ans_index,steps_left,test_flag)
        steps_left -= n_steps

        if test_flag:
          pbar.update(idx)
          APs.append(AP)
          Returns.append(self.agent.episode_reward)
          logging.debug('Episode Reward : %f', self.agent.episode_reward )
        else:
          Losses.append(self.agent.episode_loss)
          pbar.update(args.step_per_epoch-steps_left)

        if self.agent.episode_reward > self.best_return[ans_index]:
          self.best_return[ans_index] = self.agent.episode_reward
          self.best_seq[ans_index] = self.agent.act_seq
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

    else:
      Loss,BestReturn = [ np.mean(Losses), np.mean(self.best_return) ]
      print_blue( 'Loss = '+str(Loss)+'\tepsilon = '+str(self.agent.epsilon)+'\tBest Return = ' + str(BestReturn) )
      print_blue( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'.format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )

  def run_episode(self,q,ans,ans_index,max_steps,test_flag = False):
    state = self.env.setSession(q,ans,ans_index,test_flag)  # Reset & First-pass
    action     = self.agent.start_episode(state)
    if test_flag and action != 4:
      logging.debug('action : -1 first pass\t\tAP : %f', self.env.retrievalmodule.MAP)

    num_steps = 0
    while True:
      reward, state = self.env.step(action)				# ENVIROMENT STEP
      terminal, AP = self.env.game_over()
      self.act_stat[action] += 1
      num_steps += 1

      if test_flag :#and action != 4:
        AM = self.env.retrievalmodule.actionmanager
        logging.debug('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)

      if num_steps >= max_steps or terminal:  # STOP Retrieve
        self.agent.end_episode(reward, terminal)
        break

      action = self.agent.step(reward, state)			# AGENT STEP
    return num_steps, AP


def launch():
  if args.test:
    print_red("TESTING MODE")
  else:
    print_red( "TRAINING MODE")

  t = time.time()

  env = setRetrievalModule()
  agt = setDialogueManager(env)
  exp = experiment(agt,env)
  print('Done, time taken {} seconds'.format(time.time()-t))
  exp.run()

def demo():
  flag = False # fast generate survey example
  env = setRetrievalModule()
  training_data, testing_data = load_query()
  with open("../data/ISDR-CMDP/PTV.big5.lex","r") as f:
    big5map = f.readlines()

  with open("../PTV.query","r") as f:
    qlist = f.readlines()

  for i in xrange(4):
    if flag:
      idx = 95 #random.randint(0,162)
    else:
      idx = input(">>> Select Query Index: ")
    q,ans,ans_index = training_data[idx]
    print "=========="
    print "query: ", qlist[idx].rstrip('\n')
    print "ans: ", ans.keys()
    state = env.setSession(q,ans,ans_index) # state is nonsense

    if flag:
      action = i # 0,1,2,3
    else:
      action = input(">>> Select an Action: ")     # 0,1,2,3
    request  = env.retrievalmodule.request(action)
    feedback = env.simulateduser.feedback_demo(request,flag)


if __name__ == "__main__":
  if args.demo:
    demo()
  else:
    launch()
