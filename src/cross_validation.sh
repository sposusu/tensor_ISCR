#!/bin/bash

for i in {1..10}
do
  python DeepReinforce.py ${i} &
done
