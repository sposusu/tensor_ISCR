import cPickle as pickle
import numpy as np
import datetime,logging
import os,sys,pdb,random,time
import progressbar
from random import shuffle
from termcolor import cprint
from progressbar import ProgressBar,Percentage,Bar,ETA
from collections import defaultdict

from DQN import q_network
from DQN import agent
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
fold = 1
fold = sys.argv[1]
exp_name = ''
 
train_data = 'train.fold'+str(fold)+'.pkl'
test_data  = 'test.fold'+str(fold)+'.pkl'
dir='../../ISDR-CMDP/'
lex = 'PTV.lex'
background = 'background/' + '.'.join(rec_type) + '.bg'
inv_index = 'index/' + rec_type[0] + '/PTV.' + '.'.join(rec_type) + '.index'
doclengs = 'doclength/' + '.'.join(rec_type) + '.length'
docmodeldir = 'docmodel/' + '/'.join(rec_type) + '/'
newdir = '../Data/query/'

def list2tuple(data):
  result = []
  for idx in range(len(data[0])):
    result.append(tuple( (data[0][idx],data[1][idx],data[2][idx]) ))
  return result

training_data = list2tuple(pickle.load(open(newdir+train_data,'r')))
testing_data  = list2tuple(pickle.load(open(newdir+test_data,'r')))
data = testing_data + training_data
num_tr_query = len(training_data)
num_tx_query = len(testing_data)
num_query = len(data)
############## NETWORK #################
input_width, input_height = [87,1]
num_actions = 5

phi_length = 1 # input 4 frames at once num_frames
discount = 1.
learning_rate = 0.00025
rms_decay = 0.99
rms_epsilon = 0.1
momentum = 0.
nesterov_momentum = 0.
clip_delta = 1.0
freeze_interval = 100 #no freeze?
batch_size = 256
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
num_epoch = 80
epsilon_decay = num_epoch * 500
step_per_epoch = 1000
# TODO
# cross validate
# overfit one query
# simulate platform
# accelerate
############ LOGGING ###################
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

exp_log_root = '../logs/'
try:
  os.makedirs(exp_log_root)
except:
  pass
cur_datetime = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
exp_log_name = exp_log_root + exp_name + '_'.join(rec_type) +'_fold'+str(fold)+ '_' + cur_datetime + ".log"

logging.basicConfig(filename=exp_log_name,level=logging.DEBUG)

def setLogging():
  print_green('learning_rate : {}'.format(learning_rate))
  print_green('clip_delta : {}'.format(clip_delta))
  print_green('freeze_interval : {}'.format(freeze_interval))
  print_green('replay_memory_size : {}'.format(replay_memory_size))
  print_green('batch_size : {}'.format(batch_size))
  print_green('num_epoch : {}'.format(num_epoch))
  print_green('step_per_epoch : {}'.format(step_per_epoch))
  print_green('epsilon_decay : {}'.format(epsilon_decay))
  print_green('network_type : {}'.format(network_type))
  print_green('input_width : {}'.format(input_width))
  print_green('batch_size : {}'.format(batch_size))
  print_green("recognition type: {}".format(rec_type))
  print "number of trainig data: ", num_tr_query
  print "number of testing data: ", num_tx_query
  print 'exp_log_name : ', exp_log_name
###############################
class experiment():
  def __init__(self,agent,env):
    print 'Initializing experiment...'
    setLogging()
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
      shuffle(training_data)

      ## TRAIN ##
      self.run_epoch()
      self.agent.finish_epoch(epoch+1)
      
      ## TEST ##
      self.agent.start_testing()
      self.run_epoch(True)
      self.agent.finish_testing(epoch+1)
      random.shuffle(data)


  def run_epoch(self,test_flag=False):
    epoch_data = training_data
    if(test_flag):
      epoch_data = testing_data
    print 'number of queries',len(epoch_data)

    ## PROGRESS BAR SETTING
    setting = [['Training',step_per_epoch], ['Testing',num_tx_query]]
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

def setEnvironment():  
  print 'Creating Environment and compiling State Estimator...'
  return Environment(lex,background,inv_index,doclengs,docmodeldir,dir)

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
                                         nesterov_momentum,
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
  env = setEnvironment()
  exp = experiment(agt,env)
  print 'Done, time taken {} seconds'.format(time.time()-t)
  exp.run()

def get_seqs():
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
  return seqs
 

def test_action():
  env = setEnvironment()
  best_returns = - np.ones(163)
  best_seqs = defaultdict(list)
  APs = np.zeros(163)
  seqs = get_seqs()

  for idx,(q, ans, ans_index) in enumerate(data):
    print '\nQuery ',idx
    for seq in seqs:
      cur_return = 0.
      init_state = env.setSession(q,ans,ans_index,True)
      for act in seq:
        reward, state = env.step(act)
        cur_return += reward
      terminal, AP = env.game_over()
      sys.stderr.write('\rActions Sequence {}    Return = {}'.format(seq,cur_return))

      if cur_return > best_returns[idx]:
        best_returns[idx] = cur_return
        best_seqs[idx] = seq
        APs[idx] = AP
    print '\rBest seq :', best_seqs[idx],'    Best Return : ', best_returns[idx],'    AP : ', APs[idx]

  filename = 'result/' + '.'.join(rec_type) + '_best_seq_return.pkl'
  with open(filename,'w') as f:
    pickle.dump( (best_returns, best_seqs,APs),f )
  print 'MAP = ', np.mean(APs),'Return = ',np.mean(Returns)

def random_action_baseline():
  filename =  'result/' + '.'.join(rec_type) + '_random_action_baseline.log'
  f = open(filename,'w')
  f.write('Index\tMAP\tReturn\n')

  env = setEnvironment()
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
  launch()
#  test_action()
#  random_action_baseline()
