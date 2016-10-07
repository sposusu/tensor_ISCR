#!/bin/bash

for i in {1..10}
do
  ./get_csv.sh $1${i}.log > $1${i}.csv
done

