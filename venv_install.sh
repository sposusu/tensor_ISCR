#!/bin/bash

# Todo: Migrate to current repo
# Download Language Model and Files
cd .. 
wget speech.ee.ntu.edu.tw/~tlkagk/ISDR-CMDP.zip
unzip ISDR-CMDP.zip
rm ISDR-CMDP.zip

cd InteractiveRetrieval

# Install virtual environment
sudo apt-get install python-virtualenv

# Create virtualenvironment
virtualenv venv
source venv/bin/activate

# hdf5 prerequisites
sudo apt-get install libhdf5-dev

# Install lapack and blas for numpy & scipy
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install Lasagne
pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
pip install Lasagne==0.1

# Install Other Packages
pip install -r requirements.txt

echo 'Now you can type "cd src" and type "make" to run the experiment!'
echo 'Type "deactivate" to exit the virtual environment'
echo 'Type "source venv/bin/activate to enter the virtual environment'

