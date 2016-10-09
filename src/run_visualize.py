import argparse
import os
import pickle
import random

import h5py
import numpy as np
from tqdm import tqdm
from tsne import bh_sne

from DQN import q_network

def run_visualize(network, feature, num_of_features, save_path):
    print("Loading network...")
    with open(network,'rb') as f:
        network = pickle.load(f)

    # Count number of lines
    print("Loading features...")
    with open(feature,'r') as f:
        count = 0
        for _ in f.readlines():
            count += 1

    # Select features
    num_of_features = ( num_of_features if count > num_of_features else count )
    sample_indices = set(random.sample(range(count),num_of_features))

    features = []
    with open(feature,'r') as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            if idx in sample_indices:
                tokens = line.strip('[]\n').split(', ')

                feature = map(float,tokens)
                if len(feature) == network.input_width:# Work around for unfinished feature files
                    features.append(feature)

    print("Predicting actions...")
    # Predict actions
    epsilon = 0.
    actions = []
    for feat in tqdm(features):
        act = network.choose_action(feat,epsilon)
        actions.append(act)

    # Save to h5_handle
    print("Saving to h5py file at {}".format(save_path))
    features = np.array(features)
    tsne_features = bh_sne(features)
    actions = np.array(actions)

    with h5py.File(save_path,'w') as h5_handle:
        h5_handle.create_dataset('features',data=features)
        h5_handle.create_dataset('tsne_features',data=tsne_features)
        h5_handle.create_dataset('actions',data=actions)

    # Information
    print("Number of features: {}".format(num_of_features))


if __name__ == "__main__":
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description="ISCR visualize")

    parser.add_argument("-n","--network",type=str,help="q_network pickle file",default = None)
    parser.add_argument("-f","--feature",type=str,help="feature file",default = None)
    parser.add_argument("--num_of_features",type=int,help="number of features",default = 10000)
    parser.add_argument("--save_path",type=str,help="save_path",default = None)

    args = parser.parse_args()

    assert args.network is not None,"Specify network pickle!"
    assert args.feature is not None,"Specify feature pickle!"
    assert args.save_path is not None,"Specify h5py save path!"

    ##################################
    #            Visualize           #
    ##################################
    run_visualize(args.network, args.feature, args.num_of_features, args.save_path)
