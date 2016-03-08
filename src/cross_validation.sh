#!/bin/bash

for i in {1..10}
do
  python old_DeepReinforce.py ${i} &
done
