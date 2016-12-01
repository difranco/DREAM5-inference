#!/usr/bin/env bash

apt-get update
apt-get dist-upgrade -y
apt-get install -y gcc g++ git build-essential python python-pip python3 python3-pip octave
apt-get autoclean

pip install --upgrade pip

pip install --upgrade jupyter scikit-learn cython setuptools numpy scipy
pip install --upgrade crosscat

pip3 install --upgrade pip

pip3 install --upgrade jupyter scikit-learn cython setuptools numpy scipy

#cd /vagrant
#git clone https://github.com/probcomp/crosscat.git
#cd crosscat
#find . -iname \*.py -execdir 2to3 -f all -f idioms -Wn {} \;
#python3 setup.py build
#python3 setup.py install  # or python setup.py develop
