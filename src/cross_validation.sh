#!/bin/bash

for i in {1..10}
do
  python DeepReinforce.py -t $1 -f ${i} -nsu --feature $2 --prefix new_simulated_user_$1_$2 &
done
