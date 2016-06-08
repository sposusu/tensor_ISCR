exec:
	python DeepReinforce.py
env:
	python python/environment.py
se:
	python train_stateestimation.py
test: 
	nosetests

# brute force oracle
bf:
	python bf_oracle.py
# random action
random:
	python random_action_baseline.py
# exp1 : train = test
exp1:
	python DeepReinforce.py --prefix toy_example
# exp2 : validation set
exp2:
	python DeepReinforce.py --prefix valid_0 -f0
# exp3 : cross validation
exp3:
	./cross_validation.sh --prefix cx_valid
# exp4 : model complexity & feature
exp4:
	python DeepReinforce.py --prefix linear_model --model_height 0
# exp5 : run pre-trained model
exp5:
	python DeepReinforce.py ---prefix test -nn ../Data/network/onebest_feature_87_epoch_50.pkl --test
# exp6 : simulated user 
