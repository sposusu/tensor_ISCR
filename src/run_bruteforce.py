import argparse
from collections import defaultdict
import cPickle as pickle
import datetime
import logging
import itertools
from multiprocessing import Pool
import os
import random
import sys
import time

import numpy as np

from experiment import Experiment

if __name__ == "__main__":
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description="ISCR Brute Force")

    parser.add_argument("-d", "--directory", type=str, help="data directory", default="")
    parser.add_argument("--result", type=str, help="result directory", default=None)
    parser.add_argument("--name", type=str, help="experiment name", default=None)
    parser.add_argument("--feature", help="feature type (all/raw/wig/nqc)", default="all")

    args = parser.parse_args()

    # Check parser arguments
    assert os.path.isdir(args.directory)
    assert isinstance(args.result,str),"Specify result directory location!"
    assert isinstance(args.name,str),"Specify experiment name!"

    #################################
    #     Load Default Argument     #
    #################################
    retrieval_args = {
        'data_dir': args.directory,
        'result_dir': args.result,
        'exp_name': args.name,
        'fold': -1,
        'feature_type': args.feature,
        'keyterm_thres': 0.5,
        'topic_prob': True,
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

    #############################
    #       Run Brute Force     #
    #############################

    print("Running brute force")

    tstart = time.time()

    env   = Experiment.set_environment(retrieval_args)

    assert retrieval_args.get('fold') == -1, "Fold should be -1 in brue force"

    # Brute Force Queries
    queries, test_queries = Experiment.load_query(retrieval_args)

    # Brute Force Logging File
    result_dir = retrieval_args.get("result_dir")
    exp_name   = retrieval_args.get("exp_name")
    exp_dir    = os.path.join(result_dir,exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp_logpath = os.path.join(exp_dir,exp_name + '_bruteforce.log')

    lf = open(exp_logpath,'w')
    lf.write('Index\tBest Sequence\tBest Return\tAP\n')
    lf.flush()

    best_returns = - np.ones(163)
    best_seqs = defaultdict(list)
    APs = np.zeros(163)

    def brute_force_job(idx):
        tstart = time.time()
        print("Brute forcing answer index {}".format(idx))
        q, ans, ans_index = queries[idx]
        for seq in itertools.product(range(5),repeat=4):
            cur_return = 0.
            init_state = env.setSession(q,ans,ans_index,True)
            seq = list(seq) + [ 4 ] # Final action is show
            for act in seq:
                reward, state = env.step(act)
                cur_return += reward
                if act == 4:
                    break

            terminal, AP = env.game_over()
            sys.stderr.write('Query {}, Actions Sequence {}, Return = {}\n'.format(idx,seq,cur_return))

            if cur_return > best_returns[idx]:
                best_returns[idx] = cur_return
                best_seqs[idx] = seq
                APs[idx] = AP

        lf.write( '{}\t{}\t{}\t{}\n'.format(idx, best_seqs[idx], best_returns[idx], APs[idx]))
        lf.flush()

        print("Brute forcing answer index {}, Best return {}. Time taken {}".format(idx, best_returns[idx], time.time() - tstart))
        return best_seqs[idx],best_returns[idx],APs[idx]

    # Start multiprocessing
    pool = Pool(8)
    results = pool.map(brute_force_job, tuple(range(163)))

    lf.close()
    # Bruce Force Pickle
    bruteforce_pickle = os.path.join(exp_dir,exp_name+'_bruteforce.pickle')
    with open(bruteforce,'w') as f:
        pickle.dump( (best_returns, best_seqs,APs),f )
    print("MAP = {}, Return = {}".format(np.means(APs),np.mean(best_returns)))
