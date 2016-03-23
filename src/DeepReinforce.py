import cPickle as pickle
import numpy as np
import datetime,logging
import os,sys,pdb,random,time
import progressbar
import argparse
from random import shuffle
from termcolor import cprint
from progressbar import ProgressBar,Percentage,Bar,ETA
from collections import defaultdict

from DQN import q_network
from DQN import agent
from IR.environment import *
from IR.util import readFoldQueries,readLex,readInvIndex
from sklearn.cross_validation import KFold
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
parser.add_argument("-t", "--type", type=int, help="recognitions type", default=0)
parser.add_argument("-f", "--fold", type=int, help="fold 1~10", default=-1)
parser.add_argument("--prefix", help="experiment name prefix",default="")

parser.add_argument("--model_width", type=int, help="model width", default=1024)
parser.add_argument("--model_height", type=int, help="model height", default=2)
parser.add_argument("--batchsize", type=int, help="batch size", default=256)
parser.add_argument("-lr","--learning_rate", type=int, help="learning rate", default=0.00025)

parser.add_argument("--num_epoch", help="number of epoch",default=80)
parser.add_argument("--step_per_epoch", help="number of step per epoch",default=1000)
parser.add_argument("--test", help="testing mode",action="store_true")
parser.add_argument("-nn","--nn_file", help="pre-trained model")
args = parser.parse_args()
#for arg in vars(args):
#  print_green('{} : {}'.format(arg,vars(args)[arg]))
##########################
#       filename         #
##########################
recognitions = [ ('onebest','CMVN'), 
                 ('onebest','tandem'),
                 ('lattice','CMVN'),
                 ('lattice','tandem') ]
rec_type = recognitions[args.type]
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
  if args.fold == -1:
    print_green( 'train = test = all queries')
    return data,data
  kf = KFold(163, n_folds=10)
  tr,tx = list(kf)[args.fold-1]
  training_data = [ data[i] for i in tr ]
  testing_data = [ data[i] for i in tx ]
  print '...done'
  return training_data, testing_data

training_data, testing_data = load_data()
############## NETWORK #################
#network_width, network_height = [args.model_width,args.model_height]        # change network shape
input_width = 0               # auto change
input_height = 1              # change feature
num_actions = 5

phi_length = 1 # input 4 frames at once num_frames
discount = 1.
learning_rate = args.learning_rate
rms_decay = 0.99
rms_epsilon = 0.1
momentum = 0.
nesterov_momentum = 0.
clip_delta = 1.0
freeze_interval = 100 #no freeze?
batch_size = args.batchsize
network_type = 'rl_dnn'

"""
Update Rules:
1. deepmind_rmsprop
2. rmsprop
3. adagrad
4. adadelta
5. sgd

Can combine with momentum ( default: 0.9 ) 
1. momentum
2. nesterov_momentum
Note: Can only set one type of momentum 
"""
update_rule = 'deepmind_rmsprop'

batch_accumulator = 'sum'
rng = np.random.RandomState()
############# REINFORCE ##################
epsilon_start = 1.0
epsilon_min = 0.1
replay_memory_size = 10000
experiment_prefix = 'result/ret'
replay_start_size = 500
update_frequency = 1
###############################
num_epoch = args.num_epoch
epsilon_decay = num_epoch * 500
step_per_epoch = args.step_per_epoch
# TODO
# simulate platform
# accelerate GPU?
# deep retrieval
############ LOGGING ###################

exp_log_root = '../logs/'
try:
  os.makedirs(exp_log_root)
except:
  pass
cur_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
exp_log_name = exp_log_root + args.prefix + '_'.join(rec_type) +'_fold'+str(args.fold) + ".log"
logging.basicConfig(filename=exp_log_name,level=logging.DEBUG)

def setLogging():
  print_green('learning_rate : {}'.format(learning_rate))
  print_green('batch_size : {}'.format(batch_size))
  print_green('clip_delta : {}'.format(clip_delta))
  print_green('freeze_interval : {}'.format(freeze_interval))
  print_green('replay_memory_size : {}'.format(replay_memory_size))
  print_green('batch_size : {}'.format(batch_size))
  print_green('num_epoch : {}'.format(args.num_epoch))
  print_green('step_per_epoch : {}'.format(args.step_per_epoch))
  print_green('epsilon_decay : {}'.format(epsilon_decay))
  print_green('network_type : {}'.format(network_type))
  print_green('input_width : {}'.format(input_width))
  print_green("recognition type: {}".format(rec_type))
  print_green("network shape: {}".format([args.model_width,args.model_height]))
  print_green("number of training data: {}".format(len(training_data)))
  print_green("number of testing data: {}".format(len(testing_data)))
  print_green('exp_log_name : {}'.format(exp_log_name))
###############################
class experiment():
  def __init__(self,agent,env):
    print 'Initializing experiment...'
    setLogging()
    self.agent = agent
    self.env = env
    self.best_seq = {}
    self.best_return = np.zeros(163)

  def run(self):
    print_red( 'Init Model')

    self.agent.start_testing()
    self.run_epoch(True)
    self.agent.finish_testing(0)
    if args.test:
      return
    for epoch in xrange(num_epoch):
      print_red( 'Running epoch {0}'.format(epoch+1))
      shuffle(training_data)

      ## TRAIN ##
      self.run_epoch()
      self.agent.finish_epoch(epoch+1)
      
      ## TEST ##
      self.agent.start_testing()
      self.run_epoch(True)
      self.agent.finish_testing(epoch+1)

  def run_epoch(self,test_flag=False):
    epoch_data = training_data
    if(test_flag):
      epoch_data = testing_data
    print 'number of queries',len(epoch_data)

    ## PROGRESS BAR SETTING
    setting = [['Training',step_per_epoch], ['Testing',len(epoch_data)]]
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
      for idx,(q, ans, ans_index) in enumerate(epoch_data):
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
      logging.debug('action : -1 first pass\t\tAP : %f', self.env.dialoguemanager.MAP)

    num_steps = 0
    while True:
      reward, state = self.env.step(action)				# ENVIROMENT STEP
      terminal, AP = self.env.game_over()
      self.act_stat[action] += 1
      num_steps += 1

      if test_flag :#and action != 4:
        AM = self.env.dialoguemanager.actionmanager
        logging.debug('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)

      if num_steps >= max_steps or terminal:  # STOP Retrieve
        self.agent.end_episode(reward, terminal)
        break

      action = self.agent.step(reward, state)			# AGENT STEP
    return num_steps, AP


def launch():
  t = time.time()
  env = setEnvironment()
  global input_width
  input_width = env.dialoguemanager.statemachine.feat_len
  
  if args.nn_file is None:
    print 'Compiling Network...'
    network = q_network.DeepQLearner(input_width, input_height, args.model_width, args.model_height, num_actions,
                                         phi_length,
                                         discount,
                                         learning_rate,
                                         rms_decay,
                                         rms_epsilon,
                                         momentum,
                                         nesterov_momentum,
                                         clip_delta,
                                         freeze_interval,
                                         batch_size,
                                         network_type,
                                         update_rule,
                                         batch_accumulator,
                                         rng)
  else:
    handle = open(args.nn_file, 'r')
    network = pickle.load(handle)
    
  print 'Creating Agent and Simulator...'
  agt = agent.NeuralAgent(network,epsilon_start,epsilon_min,epsilon_decay,
                                  replay_memory_size,
                                  experiment_prefix,
                                  replay_start_size,
                                  update_frequency,
                                  rng)
  exp = experiment(agt,env)
  print 'Done, time taken {} seconds'.format(time.time()-t)
  exp.run()

if __name__ == "__main__":
  if args.test:
    print_red("TESTING MODE")
    
  else:
    print_red( "TRAINING MODE")
  launch()
