#!/bin/bash
# $1: config-name
# $2: cuda-device
# $3: location

if [[ "$3" == "local" ]]
then
  results_dir="results"
elif [[ "$3" == "rattle" ]]
then
  results_dir="results"
else
  echo "Unknown location: ${5}"
fipath_config="configs/${1}.jsonnet"
path_results="${results_dir}results/${1}"
allennlp train $path_config --include-package mtl_sd -s $path_results -o "{'trainer.cuda_device': $2}"
