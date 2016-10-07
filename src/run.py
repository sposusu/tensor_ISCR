import argparse
from collections import defaultdict
import cPickle as pickle
import numpy as np
import datetime,logging
import os
import random
import sys
import time

import progressbar,argparse

from termcolor import cprint
from progressbar import ProgressBar,Percentage,Bar,ETA

from DQN import agent, q_network
from IR.environment import *
from IR.dialoguemanager import DialogueManager
from IR.human import SimulatedUser
# util IO
from IR.util import readFoldQueries,readLex,readInvIndex
from IR import reader
from sklearn.cross_validation import KFold

def set_environment(retrieval_args):
  print('Creating Environment with DialogueManager and Simulated User...')
  # Dialogue Manager
  data_dir = retrieval_args.get('data_dir')
  feature_type = retrieval_args.get('feature_type')

  retrievalmodule = DialogueManager(
                          data_dir            = data_dir,
                          feature_type        = feature_type
                        )

  # Simulated User
  keyterm_thres = retrieval_args.get('keyterm_thres')
  topic_prob    = retrieval_args.get('topic_prob')
  survey        = retrieval_args.get('survey')

  simulateduser = SimulatedUser(
                        data_dir      = data_dir,
                        keyterm_thres = keyterm_thres,
                        topic_prob    = topic_prob,
                        survey        = survey
                        )

  # Set Environment
  env = Environment(retrievalmodule,simulateduser)
  return env

def set_agent(training_args, reinforce_args, feature_length, result_dir):
    print("Setting up Agent...")

    ######################################
    #    Predefined Network Parameters   #
    ######################################

    input_height = 1              # change feature
    input_width  = feature_length
    num_actions  = 5
    phi_length   = 1 # input 4 frames at once num_frames
    discount     = 1.
    rms_decay    = 0.99
    rms_epsilon  = 0.1
    momentum     = 0.
    nesterov_momentum = 0.
    network_type = 'rl_dnn'
    batch_accumulator = 'sum'
    rng = np.random.RandomState()

    network = q_network.DeepQLearner(
                                    input_width       = feature_length,
                                    input_height      = 1,
                                    net_width         = training_args.get('model_width'),
                                    net_height        = training_args.get('model_height'),
                                    num_actions       = num_actions,
                                    num_frames        = phi_length,
                                    discount          = discount,
                                    learning_rate     = training_args.get('learning_rate'),
                                    rho               = rms_decay,
                                    rms_epsilon       = rms_epsilon,
                                    momentum          = momentum,
                                    nesterov_momentum = nesterov_momentum,
                                    clip_delta        = training_args.get('clip_delta'),
                                    freeze_interval   = reinforce_args.get('freeze_interval'),
                                    batch_size        = training_args.get('batch_size'),
                                    network_type      = network_type,
                                    update_rule       = training_args.get('update_rule'),
                                    batch_accumulator = batch_accumulator,
                                    rng               = rng
                                      )

    experiment_prefix = os.path.join(result_dir,'model')

    agt = agent.NeuralAgent(
                            q_network           = network,
                            epsilon_start       = reinforce_args.get('epsilon_start'),
                            epsilon_min         = reinforce_args.get('epsilon_min'),
                            epsilon_decay       = reinforce_args.get('epsilon_decay'),
                            replay_memory_size  = reinforce_args.get('replay_memory_size'),
                            exp_pref            = experiment_prefix,
                            replay_start_size   = reinforce_args.get('replay_start_size'),
                            update_frequency    = reinforce_args.get('update_frequency'),
                            rng                 = rng
                            )

    return agt


###############################
#          Experiment         #
###############################
class experiment():
    def __init__(self,retrieval_args, training_args, reinforce_args, agent,env):
        print('Initializing experiment...')
        self.set_logging(retrieval_args)

        self.training_data, self.testing_data = self.load_query(retrieval_args)

        self.agent = agent
        self.env = env

        self.num_epochs      = training_args.get('num_epochs')
        self.steps_per_epoch = reinforce_args.get('steps_per_epoch')

        self.best_seq = {}
        self.best_return = np.zeros(163) # to be removed

    def set_logging(self, retrieval_args):
        result_dir = retrieval_args.get('result_dir')
        exp_name = retrieval_args.get('exp_name')
        fold = retrieval_args.get('fold')

        exp_name = result_dir.split('/')[-1] + '_fold{}'.format(str(fold)) + '.log'
        exp_log_path = os.path.join(result_dir,exp_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        logging.basicConfig(filename=exp_log_path,level=logging.DEBUG)

    def load_query(self, retrieval_args):

      result_dir = retrieval_args.get("data_dir")
      query_pickle = os.path.join(result_dir,'query.pickle')
      data = reader.load_from_pickle(query_pickle)[:10]

      fold = retrieval_args.get('fold')

      if fold == -1:
          return data, data
      else:
          kf = KFold(len(data), n_folds=10)

          tr,tx = list(kf)[fold-1]

          training_data = [ data[i] for i in tr ]
          testing_data  = [ data[i] for i in tx ]

          return training_data, testing_data

    def run(self):
        print_red('Init Model')

        self.agent.start_testing()
        self.run_epoch(test_flag=True)
        self.agent.finish_testing(0)

        for epoch in range(1,self.num_epochs+1,1):
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
        setting = [['Training',self.steps_per_epoch], ['Testing',len(epoch_data)]]
        setting = setting[test_flag]
        widgets = [ setting[0] , Percentage(), Bar(), ETA() ]

        pbar = ProgressBar(widgets=widgets,maxval=setting[1]).start()
        APs = []
        Returns = []
        Losses = []
        self.act_stat = [0,0,0,0,0]

        steps_left = self.steps_per_epoch
        while steps_left > 0:
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
                    pbar.update(self.steps_per_epoch-steps_left)

                if self.agent.episode_reward > self.best_return[ans_index]:
                    self.best_return[ans_index] = self.agent.episode_reward
                    self.best_seq[ans_index]    = self.agent.act_seq

                if steps_left <= 0:
                    break

                if test_flag:
                    break

        pbar.finish()

        if test_flag:
            MAP,Return = [ np.mean(APs) , np.mean(Returns) ]
            print_yellow( 'MAP = {} \tReturn = {}'.format(MAP,Return) )
            print_yellow( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'\
                .format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )

        else:
            Loss,BestReturn = [ np.mean(Losses), np.mean(self.best_return) ]
            print_blue( 'Loss = {} \tepsilon = {} \tBest Return = {}'.format(Loss,self.agent.epsilon,BestReturn) )
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

            if test_flag: #and action != 4:
                AM = self.env.retrievalmodule.actionmanager
                logging.debug('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)

            if num_steps >= max_steps or terminal:  # STOP Retrieve
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, state)			# AGENT STEP

        return num_steps, AP

def run_training(retrieval_args, training_args, reinforce_args):
    tstart = time.time()

    env   = set_environment(retrieval_args)
    agent = set_agent(training_args, reinforce_args, env.retrievalmodule.statemachine.feat_len, retrieval_args.get('result_dir'))
    print("Environemt and Agent Set")

    exp   = experiment(retrieval_args, training_args, reinforce_args, agent, env)
    print('Experiment Set. Time taken: {} seconds'.format(time.time()-tstart))

    exp.run()


#################################
#        Cprint Function        #
#################################
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

if __name__ == "__main__":
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description="Interactive Spoken Content Retrieval")

    parser.add_argument("-d", "--directory", type=str, help="data directory", default='/home/ubuntu/InteractiveRetrieval/data/reference')
    parser.add_argument("-f", "--fold", type=int, help="fold 1~10", default=-1)
    parser.add_argument("--prefix",  type=str, help="experiment prefix", default=None)
    parser.add_argument("--feature", help="feature type (all/raw/wig/nqc)", default="all") # TODO not implement yet

    args = parser.parse_args()

    #################################
    #     Load Default Argument     #
    #################################
    retrieval_args = {
        'data_dir': '/home/ubuntu/InteractiveRetrieval/data/onebest_CMVN',
        'result_dir': '/home/ubuntu/InteractiveRetrieval/result/onebest_CMVN',
        'exp_name': 'onebest_CMVN',
        'fold': args.fold,
        'feature_type': args.feature
        'keyterm_thres': 0.5,
        'topic_prob': True,
        'cost_noise_std': 1
    }

    training_args = {
        'num_epochs': 100,
        'batch_size': 256,
        'model_width': 1024,
        'model_height': 2,
        'learning_rate': 0.00025,
        'clip_delta': 1.0,
        'update_rule': 'deepmind_rmsprop'
    }

    reinforce_args = {
        'steps_per_epoch': 1000,
        'replay_start_size': 500,
        'replay_memory_size': 10000,
        'epsilon_decay': 100000,
        'epsilon_min': 0.1,
        'epsilon_start': 1.0,
        'freeze_interval': 100,
        'update_frequency': 1
    }

    run_training(retrieval_args, training_args, reinforce_args)
