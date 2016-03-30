#!/bin/bash

for i in {1..10}
do
  python DeepReinforce.py -t 0 -f ${i} --model_height 0 --prefix linear_   &
done
