# exp1 : train = test
python DeepReinforce.py --prefix toy_example 

# exp2 : validation set
python DeepReinforce.py --prefix valid_0 -f0

# exp3 : cross validation
./cross_validation.sh --prefix cx_valid

# exp4 : model complexity & feature
python DeepReinforce.py --prefix linear_model --model_height 0

# exp5 : run pre-trained model
python DeepReinforce.py -nn ../Data/network/onebest_feature_87_epoch_50.pkl --test

# exp6 : simulated user


