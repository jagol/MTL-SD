#!/bin/bash
# $1: model name
# $2 name of test-files e.g. "test.jsonl"
# $3: cuda-device
# $4: label-type, "label_orig" or "label_uni" (check that it's same as specified in StanceDetectionReader)
# $5: local or rattle
# $6: "all" or list of corpora to test

# example: bash evaluate.sh base_conf test.jsonl -1 label_orig local SemEval2016Task6

model_path="results/${1}/model.tar.gz"
mkdir "results/${1}/predictions/"

if [[ "$6" == "all" ]]
then
  tasks=("arc" "ArgMin" "fnc-1" "IAC" "IBM_CLAIM_STANCE" "multi-target-sd" "PERSPECTRUM" "SCD" "SemEval2016Task6" "SemEval2019Task7" "Snopes")
else
  tasks=${@:6};
fi

if [[ "$5" == "local" ]]
then
  data_dir="data"
elif [[ "$5" == "rattle" ]]
then
  data_dir="/srv/scratch0/jgoldz/mthesis/data"
else
  echo "Unknown location: ${5}"
fi

for dataset in ${tasks[@]}; do
  data_path="${data_dir}/${dataset}/${2}"
  path_output_file="results/${1}/predictions/${dataset}.jsonl"
  allennlp predict $model_path $data_path --include-package mtl_sd --predictor multitask --cuda-device $3 --output-file $path_output_file --silent
  python3 scripts/extract_results.py --path $path_output_file
done

python3 scripts/evaluate.py --predictions "results/${1}/predictions/" --labels $2 --evaluation "results/${1}/evaluation.json" --vocab "results/${1}/vocabulary" --label_type $4 --data_dir $data_dir
