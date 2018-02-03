import logging
import os
import random

import numpy as np
from progressbar import ProgressBar,Percentage,Bar,ETA
from sklearn.cross_validation import KFold
from termcolor import cprint

from DQN import agent, tensor_q_network
from IR.dialoguemanager import DialogueManager
from IR.environment import Environment
from IR.human import SimulatedUser
from IR import reader

###############################
#          Experiment         #
###############################
class Experiment(object):
    def __init__(self,retrieval_args, training_args, reinforce_args):
        print('Initializing experiment...')
        self.set_logging(retrieval_args)

        self.training_data, self.testing_data = Experiment.load_query(retrieval_args)

        self.env = Experiment.set_environment(retrieval_args)

        self.agent = Experiment.set_agent(retrieval_args, training_args, reinforce_args,\
                self.env.dialoguemanager.statemachine.feat_len, retrieval_args.get('result_dir'))

        self.num_epochs      = training_args.get('num_epochs')
        self.steps_per_epoch = reinforce_args.get('steps_per_epoch')

        self.best_seq = {}
        self.best_return = np.zeros(163) # to be removed

    def __del__(self):
        if self.feature_handle is not None:
            self.feature_handle.close()

    def set_logging(self, retrieval_args):
        result_dir   = retrieval_args.get('result_dir')
        exp_name     = retrieval_args.get('exp_name')
        exp_dir      = os.path.join(result_dir,exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Log File
        fold         = retrieval_args.get('fold')
        exp_logfile  = exp_name + '_fold{}'.format(str(fold)) + '.log'
        exp_log_path = os.path.join(exp_dir,exp_logfile)

        logging.basicConfig(filename=exp_log_path,level=logging.DEBUG)

        # Feature file handler, Open is specified
        if retrieval_args.get('save_feature'):
            feature_file = os.path.join(exp_dir,exp_name + '.feature')
            self.feature_handle = open(feature_file,'wb')
        else:
            self.feature_handle = None

        # Display Parameters
        Experiment.print_green("data dir: {}".format(retrieval_args.get('data_dir')))
        Experiment.print_green("fold: {}".format(retrieval_args.get('fold')))
        Experiment.print_green("feature_type: {}".format(retrieval_args.get('feature_type')))
        Experiment.print_green("keyterm_thres: {}".format(retrieval_args.get('keyterm_thres')))
        Experiment.print_green("choose_random_topic: {}".format(retrieval_args.get('choose_random_topic')))
        Experiment.print_green("use_survey: {}".format(retrieval_args.get('use_survey')))
        if retrieval_args.get('choose_random_topic') and retrieval_args.get('use_survey'):
            Experiment.print_yellow("choose_random_topic overrided by use_survey!")
        Experiment.print_green("experiment_log_file: {}".format(exp_log_path))
        if retrieval_args.get('save_feature'):
            Experiment.print_green("feature_file: {}".format(feature_file))

    @staticmethod
    def set_environment(retrieval_args):
        print('Creating Environment with DialogueManager and Simulated User...')
        # Dialogue Manager
        data_dir = retrieval_args.get('data_dir')
        feature_type = retrieval_args.get('feature_type')
        survey = retrieval_args.get('survey')

        dialoguemanager = DialogueManager(
                              data_dir            = data_dir,
                              feature_type        = feature_type,
                              survey              = survey,
                            )

        # Simulated User
        keyterm_thres          = retrieval_args.get('keyterm_thres')
        choose_random_topic    = retrieval_args.get('choose_random_topic')
        use_survey             = retrieval_args.get('use_survey')

        simulateduser = SimulatedUser(
                            data_dir              = data_dir,
                            keyterm_thres         = keyterm_thres,
                            choose_random_topic   = choose_random_topic,
                            use_survey            = use_survey
                            )

        # Set Environment
        env = Environment(dialoguemanager,simulateduser)
        return env

    @staticmethod
    def set_agent(retrieval_args, training_args, reinforce_args, feature_length, result_dir):
        print("Setting up Agent...")

        ######################################
        #    Predefined Network Parameters   #
        ######################################

        # Network
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

        network = tensor_q_network.DeepQLearner(
                                        input_width       = feature_length,
                                        input_height      = input_height,
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
        # Agent
        experiment_prefix = os.path.join(result_dir,retrieval_args.get("exp_name"),'model')

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

    @staticmethod
    def load_query(retrieval_args):
        data_dir = retrieval_args.get("data_dir")
        query_pickle = os.path.join(data_dir,'query.pickle')
        data = reader.load_from_pickle(query_pickle)

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
        # Start Running
        # Test one epoch first
        Experiment.print_red('Init Model')
        self.agent.start_testing()
        self.run_epoch(test_flag=True)
        self.agent.finish_testing(0)

        for epoch in range(1,self.num_epochs+1,1):
            Experiment.print_red('Running epoch {0}'.format(epoch))
            random.shuffle(self.training_data)

            ## TRAIN ##
            self.run_epoch()
            self.agent.finish_epoch(epoch)

            ## TEST ##
            self.agent.start_testing()
            self.run_epoch(test_flag = True)
            self.agent.finish_testing(epoch)

    def run_epoch(self,test_flag=False):
        epoch_data = self.training_data
        if test_flag:
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
            Experiment.print_yellow( 'MAP = {}\tReturn = {}'.format(MAP,Return) )
            Experiment.print_yellow( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'\
                .format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )
        else:
            Loss,BestReturn = [ np.mean(Losses), np.mean(self.best_return) ]
            Experiment.print_blue( 'Loss = {} \tepsilon = {} \tBest Return = {}'.format(Loss,self.agent.epsilon,BestReturn) )
            Experiment.print_blue( 'act[0] = {}\tact[1] = {}\tact[2] = {}\tact[3] = {}\tact[4] = {}'\
                .format(self.act_stat[0],self.act_stat[1],self.act_stat[2],self.act_stat[3],self.act_stat[4]) )

    def run_episode(self,q,ans,ans_index,max_steps,test_flag=False):
        state  = self.env.setSession(q,ans,ans_index,test_flag)  # Reset & First-pass
        action = self.agent.start_episode(state)
        if test_flag and action != 4:
            logging.debug('action : -1 first pass\t\tAP : %f', self.env.dialoguemanager.MAP)

        # Save state
        if self.feature_handle is not None:
            self.feature_handle.write(str(state.tolist())+"\n")

        num_steps = 0
        while True:
            reward, state = self.env.step(action) # ENVIRONMENT STEP
            terminal, AP  = self.env.game_over()
            self.act_stat[ action ] += 1
            num_steps += 1

            # Save state
            if self.feature_handle is not None and state is not None:
                self.feature_handle.write(str(state.tolist())+"\n")

            # Antonie: Why do this?
            if test_flag: # and action != 4:
                AM = self.env.dialoguemanager.actionmanager
                logging.debug('action : %d %s\tcost : %s\tAP : %f\treward : %f',action,AM.actionTable[ action ],AM.costTable[ action ],AP,reward)

            if num_steps >= max_steps or terminal:  # STOP Retrieve
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, state)	# AGENT STEP

        return num_steps, AP

    @staticmethod
    def print_red(x):  # epoch
        cprint(x, 'red')
        logging.info(x)

    @staticmethod
    def print_blue(x): # train info
        cprint(x, 'blue')
        logging.info(x)

    @staticmethod
    def print_yellow(x): # test info
        cprint(x, 'yellow')
        logging.info(x)

    @staticmethod
    def print_green(x):  # parameter
        cprint(x, 'green')
        logging.info(x)
