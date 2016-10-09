import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from experiment import Experiment

import multiprocessing

if __name__ == "__main__":
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description="Interactive Spoken Content Retrieval")

    parser.add_argument("-d", "--directory", type=str, help="data directory", default="")
    parser.add_argument("-r", "--repeat", type=int, help="repeat iterations for random action", default=10)
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

    #################################
    #  Run random action baseline   #
    #################################
    repeat = args.repeat

    queries, _ = Experiment.load_query(retrieval_args)
    env = Experiment.set_environment(retrieval_args)

    exp_name = retrieval_args.get('exp_name')
    result_dir = retrieval_args.get('result_dir')
    exp_dir = os.path.join(result_dir,exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    logfile = os.path.join(exp_dir,'random_action_baseline.log')
    logfile_handle = open(logfile,'w')
    logfile_handle.write('Index\tMAP\tReturn\n')

    EAPs = np.zeros(163)
    EReturns = np.zeros(163)

    def randon_action_job(idx):
        print("Random action idx {} started.".format(idx))
        tstart = time.time()
        q, ans, ans_index = queries[idx]

        APs = np.zeros(repeat)
        Returns = np.zeros(repeat)
        for i in range(repeat):
            cur_return = 0.
            terminal = False

            init_state = env.setSession(q,ans,ans_index,True)

            while not terminal:
                act = np.random.randint(5)
                reward, state = env.step(act)
                cur_return += reward
                terminal, AP = env.game_over()

            APs[i] = AP
            Returns[i] = cur_return

        EAPs[idx] = np.mean(APs)
        EReturns[idx] = np.mean(Returns)

        logfile_handle.write( '{}\t{}\t{}\n'.format(idx,EAPs[idx],EReturns[idx]) )
        logfile_handle.flush()

        EAP_avg = np.mean(EAPs)
        EReturns_avg = np.mean(EReturns)

        print("Random action idx {} ended. AP {}, Return {}, Time taken {} seconds.".format(idx,EAP_avg, EReturns_avg, time.time()-tstart))

        return EAP_avg, EReturns_avg

    p = multiprocessing.Pool(8)
    results = p.map(randon_action_job,tuple(range(163)))

    aps, rets = zip(*results)
    total_ap_avg = np.mean(aps)
    total_ret_avg = np.mean(rets)

    print("Total AP {}, Total Return {}".format(total_ap_avg,total_ap_avg))
    logfile_handle.write("Total AP {}, Total Return {}".format(total_ap_avg,total_ap_avg))
    logfile_handle.flush()
    logfile_handle.close()
