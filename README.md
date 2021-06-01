# Multi-Task Learning for Stance Detection: Testing Auxiliary Datasets and Tasks

## Installation

The scripts are tested with python 3.8.6.

To install the necessary packages into a virtual environment, run:
```
python3 -m venv mtl_sd_venv
pip3 install -r requirements.txt
```

## Obtaining the Data
To download the corpora of the stance detection benchmark, use the script provided by [Schiller et al 2021](https://link.springer.com/article/10.1007/s13218-021-00714-w): [https://github.com/UKPLab/mdl-stance-robustness/blob/master/download.sh](https://github.com/UKPLab/mdl-stance-robustness/blob/master/download.sh)

To download the corpora taken from the GLUE Benchmark, go to: [https://gluebenchmark.com/tasks](https://gluebenchmark.com/tasks)

The IMDB corpus can be obtained at
[https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)

To download the iSarcasm corpus a Twitter developer account is necessary. The tweed ids and labels can be downloaded at: 
[https://anonymous.4open.science/r/24639225-ac0e-4057-b2d4-16e7e50570d0/README.md](https://anonymous.4open.science/r/24639225-ac0e-4057-b2d4-16e7e50570d0/README.md)

After obtaining a Twitter developer account the tweets themselves can be downloaded with `scripts/download\_tweets.py`.

## Preprocessing
To preprocess the data, first move the downloaded corpora into the folder `data`. Then run 
```
    python3 scripts/preprocess.py -c <corpus-name> -s 0.1 -d data/
```
for each corpus that you will use for training or evaluation. 

## Training and Evaluation
The scripts `train_eval.sh` executes training and evaluation. 

The script takes the following arguments:
```
bash train_eval.sh <name-of-configuration> <name-of-test-file> <gpu-num> label_orig <server-name> <warmup-ratio> <batch-size>  <num-epochs> <space-separated-list-of-corpora-to-use-for-evaluation>
```

An example call looks as follows:
```
bash train_eval.sh arc_ArgMin_850 test.jsonl 1 label_orig rattle  0.2 8 5 arc
```