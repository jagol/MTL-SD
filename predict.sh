#!/bin/bash
# $1: model name
# $2: test data
# $4: cuda-device

model_path = "results/${1}/model.tar.gz"
path_output_file = "results/${1}/predictions.jsonl"

allennlp predict model_path $2 --include-package mtl_sd --predictor multitask --cuda-device $4 --output-file path_output_file
python3 extract_results.py --path $3