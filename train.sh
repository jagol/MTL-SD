#!/bin/bash
# $1: config-name
# $2: cuda-device

path_config="configs/${1}.jsonnet"
path_results="results/${1}"
allennlp train $path_config --include-package mtl_sd -s $path_results -o "{'trainer.cuda_device': $2}"