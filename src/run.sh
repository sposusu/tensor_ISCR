#!/bin/bash

for i in {1..10}
do
  python DeepReinforce.py -t $1 -f ${i} --feature $2 --prefix $3 &
done
