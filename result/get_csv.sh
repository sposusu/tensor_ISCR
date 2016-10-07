#!/bin/bash 

cat $1 |grep INFO | sed 's/INFO:root://g' | grep MAP | sed 's/MAP = //g' | sed 's/Return = //g' | sed 's/\t/,/g'
