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
    logfile = os.path.join(exp_dir,'firstpass_map.log')
    logfile_handle = open(logfile,'w')


    def run_firstpass_job(idx):
        print("Query idx {} started.".format(idx))
        tstart = time.time()
        q, ans, ans_index = queries[idx]

        idx_ap_list = []
        idx_return_list = []

        # Start interaction
        init_state = env.setSession(q,ans,ans_index,True)
        ret = env.dialoguemanager.ret
        AP  = env.dialoguemanager.evalAP(ret,ans)
        
  
        msg = "Query {}, Average Precision {}".format(idx, AP)
        print(msg)
        logfile_handle.write(msg)

        return idx, AP

    total_time_start = time.time()
    p = multiprocessing.Pool(args.num_cores)
    results = p.map(run_firstpass_job,tuple(range(163)))
    
    MAP = np.mean([ x[1] for x in results ])
    print("Mean Average Precision: {}".format(MAP))
    logfile_handle.write("Mean Average Precision: {}".format(MAP))
    logfile_handle.close()


