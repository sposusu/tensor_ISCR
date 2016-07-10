exec:
	python DeepReinforce.py
env:
	python python/environment.py
se:
	python train_stateestimation.py
test: 
	nosetests
clean:
	rm -r *.pyc

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
test_model:
	python DeepReinforce.py -nn ../Data/network/onebest_feature_87_epoch_50.pkl --test --nolog
# exp_cost : simulated user 
exp_cost1:
	python DeepReinforce.py --prefix cost_var_1 
exp_cost10:
	python DeepReinforce.py --prefix cost_var_10   --cost_std 10
exp_cost100:
	python DeepReinforce.py --prefix cost_var_100  --cost_std 100
exp_cost1000:
	python DeepReinforce.py --prefix cost_var_1000 --cost_std 1000

demo:
	python DeepReinforce.py --demo        
