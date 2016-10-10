#!/bin/bash

# -f : fold
# --feature: all/raw/wig/nqc
# --directory: data directory $ISCR_HOME/data/target_directory/
# --result: result directory $ISCR_HOME/result/
# --name: experiment_name

for i in {1..10}
do
    python run_training.py -f ${i} --feature ${1} --directory ${2} --result ${3} --name ${4}
done
