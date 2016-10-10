import argparse
import os

from experiment import Experiment

if __name__ == "__main__":
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description="Interactive Spoken Content Retrieval")

    # Retrieval Arguments
    parser.add_argument("-f", "--fold", type=int, help="fold 1~10", default=-1)
    parser.add_argument("-d", "--directory", type=str, help="data directory", default="")
    parser.add_argument("--feature", help="feature type (all/raw/wig/nqc)", default="all")
    parser.add_argument("--use_survey", action = "store_true", help="use survey prob distributions", default = False)

    # Training Arguments
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=100)


    # Saving Path Arguments
    parser.add_argument("--save_feature", action="store_true", help="save encountered features to file", default = False)
    parser.add_argument("--name", type=str, help="experiment name", default=None)
    parser.add_argument("--result", type=str, help="result directory", default=None)

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
        'fold': args.fold,
        'feature_type': args.feature,
        'survey': args.use_survey,
        'keyterm_thres': 0.5,
        'topic_prob': True,
        'save_feature': args.save_feature
    }

    training_args = {
        'num_epochs': args.num_epochs,
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
        'freeze_interval': 500,
        'update_frequency': 1
    }

    ###############################
    #        Run Experiment       #
    ###############################
    exp = Experiment(retrieval_args, training_args, reinforce_args)
    exp.run()
