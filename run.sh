#!/bin/bash

cd ..
# wget speech.ee.ntu.edu.tw/~tlkagk/interactive.zip
wget speech.ee.ntu.edu.tw/~tlkagk/ISDR-CMDP.zip
unzip ISDR-CMDP.zip
rm ISDR-CMDP.zip

cd InteractiveRetrieval
pip install -r requirements.txt

echo 'Now you can "cd src && ./cross_validation.sh" to run the experiment'
