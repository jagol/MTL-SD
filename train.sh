#!/bin/bash
# $1: config-name

path_config="configs/${1}.jsonnet"
path_results="results/${1}"
allennlp train $path_config --include-package mtl_sd -s $path_results