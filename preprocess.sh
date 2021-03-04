#!/bin/bash
# $1: dev-size: 0-1
python3 scripts/preprocess.py -c SemEval2016 -s $1 -d data/
python3 scripts/preprocess.py -c IBMCS -s $1 -d data/
python3 scripts/preprocess.py -c arc -s $1 -d data/
python3 scripts/preprocess.py -c ArgMin -s $1 -d data/
python3 scripts/preprocess.py -c FNC1 -s $1 -d data/
python3 scripts/preprocess.py -c IAC -s $1 -d data/
python3 scripts/preprocess.py -c PERSPECTRUM -s $1 -d data/
python3 scripts/preprocess.py -c SCD -s $1 -d data/
python3 scripts/preprocess.py -c SemEval2019 -s $1 -d data/
python3 scripts/preprocess.py -c Snopes -s $1 -d data/