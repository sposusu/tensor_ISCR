import argparse
import multiprocessing
import os
import time

import numpy as np
from tqdm import tqdm

from experiment import Experiment


if __name__ == "__main__":
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description="Interactive Spoken Content Retrieval")

    parser.add_argument("-d", "--directory", type=str, help="data directory", default="")
    parser.add_argument("--result", type=str, help="result directory", default=None)
    parser.add_argument("--name", type=str, help="experiment name", default=None)
    parser.add_argument("--feature", help="feature type (all/raw/wig/nqc)", default="raw")
    parser.add_argument("--num_cores", help="number of cores to use", default=8)

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

    #################################
    #  Run random action baseline   #
    #################################

    queries, _ = Experiment.load_query(retrieval_args)
    env = Experiment.set_environment(retrieval_args)

    exp_name = retrieval_args.get('exp_name')
    result_dir = retrieval_args.get('result_dir')
    exp_dir = os.path.join(result_dir,exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    logfile = os.path.join(exp_dir,'repeat_action_baseline.log')
    logfile_handle = open(logfile,'w')


    def randon_action_job(idx):
        print("Query idx {} started.".format(idx))
        tstart = time.time()
        q, ans, ans_index = queries[idx]

        idx_ap_list = []
        idx_return_list = []

        for action in range(4):
            action_ap_list = []
            action_return_list = []

            cur_return = 0.

            # Start interaction
            init_state = env.setSession(q,ans,ans_index,True)
            ret = env.dialoguemanager.ret
            AP  = env.dialoguemanager.evalAP(ret,ans)

            action_ap_list.append(AP)
            action_return_list.append(cur_return)

            print("Query idx {}, action {}, turn {}, AP {}, Return {}.".format(idx,action,0,AP,cur_return))
            logfile_handle.write("Query idx {}, action {}, turn {}, AP {}, Return {}.\n".format(idx,action,0,AP,cur_return))
            for turn in range(1,5,1): # For turns
                reward, state = env.step(action)
                cur_return += reward
                terminal, AP = env.game_over()

                action_ap_list.append(AP)
                action_return_list.append(cur_return)

                print("Query idx {}, action {}, turn {}, AP {}, Return {}.".format(idx,action,turn,AP,cur_return))
                logfile_handle.write("Query idx {}, action {}, turn {}, AP {}, Return {}.\n".format(idx,action,turn,AP,cur_return))
            idx_ap_list.append(action_ap_list)
            idx_return_list.append(action_return_list)
        print("Query idx {}, Time taken {} seconds".format(idx, time.time() - tstart))
        return idx, idx_ap_list, idx_return_list

    total_time_start = time.time()
    p = multiprocessing.Pool(args.num_cores)
    results = p.map(randon_action_job,tuple(range(163)))

    # Dimension: Query idx, Action, Turn
    EAPs     = np.zeros((163,5,5))
    EReturns = np.zeros((163,5,5))

    for query_idx, idx_ap_list, idx_return_list in results:
        for action_idx, (action_ap_list, action_reward_list) in enumerate(zip(idx_ap_list, idx_return_list)):
            EAPs[query_idx][action_idx] = action_ap_list
            EReturns[query_idx][action_idx] = action_reward_list

    print("Total time taken {}\n".format(time.time()-total_time_start))
    print("Repeat action baseline:")
    for action_idx, action in enumerate(['doc','keyterm','request','topic']):
        print("Repeat {}".format(action))
        print("MAP: {}".format(EAPs.mean(0)[action_idx]))
        print("Return: {}\n".format(EReturns.mean(0)[action_idx]))
        logfile_handle.write("Repeat {}\n".format(action))
        logfile_handle.write("MAP: {}\n".format(EAPs.mean(0)[action_idx]))
        logfile_handle.write("Return: {}\n".format(EReturns.mean(0)[action_idx]))
    logfile_handle.flush()
    logfile_handle.close()
