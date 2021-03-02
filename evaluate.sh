#!/bin/bash
# $1: model name
# $2: test data
# $3: cuda-device
# $4: label-type, "label_orig" or "label_uni" (check that it's same as specified in StanceDetectionReader)

model_path="results/${1}/model.tar.gz"
path_output_file="results/${1}/predictions.jsonl"
allennlp predict $model_path $2 --include-package mtl_sd --predictor multitask --cuda-device $3 --output-file $path_output_file --silent
python3 scripts/extract_results.py --path "results/${1}/predictions.jsonl"
python3 scripts/evaluate.py --predictions "results/${1}/predictions_extracted.csv" --labels $2 --evaluation "results/${1}/evaluation.json" --vocab "results/${1}/vocabulary" --label_type $4