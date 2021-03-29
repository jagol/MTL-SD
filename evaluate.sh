#!/bin/bash
# $1: model name
# $2 name of test-files e.g. "test.jsonl"
# $3: cuda-device
# $4: label-type, "label_orig" or "label_uni" (check that it's same as specified in StanceDetectionReader)
# $5: local or rattle
# $6: "all" or list of corpora to test, if no corpus arg is given predicting is skipped.

# example: bash evaluate.sh base_conf test.jsonl -1 label_orig local SemEval2016Task6

if [[ "$6" == "all" ]]
then
  tasks=("arc" "ArgMin" "FNC1" "IAC" "IBMCS" "PERSPECTRUM" "SCD" "SemEval2016Task6" "SemEval2019Task7" "Snopes")
elif [[ "$6" == "" ]]
then
  tasks=()
else
  tasks=${@:6};
fi

if [[ "$5" == "local" ]]
then
  data_dir="data"
  results_dir="results"
elif [[ "$5" == "rattle" ]]
then
  data_dir="/srv/scratch0/jgoldz/mthesis/data"
  results_dir="results"
else
  echo "Unknown location: ${5}"
fi

model_path="${results_dir}/${1}/model.tar.gz"
mkdir "${results_dir}/${1}/predictions/"

for dataset in ${tasks[@]}; do
  data_path="${data_dir}/${dataset}/${2}"
  path_output_file="${results_dir}/${1}/predictions/${dataset}.jsonl"
  echo "Now processing dataset: $dataset"
  echo "Data path set to: $data_path"
  echo "Path to output file set to: $path_output_file"
  echo "Predicting..."
  allennlp predict $model_path $data_path --include-package mtl_sd --predictor multitask --cuda-device $3 --output-file $path_output_file --silent --predictor multitask_stance
done

echo "Compute metrics for predictions..."
python3 scripts/evaluate.py --predictions "${results_dir}/${1}/predictions/" --labels $2 --evaluation "${results_dir}/${1}/evaluation.json" --vocab "${results_dir}/${1}/vocabulary" --label_type $4 --data_dir $data_dir
echo "Main metrics as csv:"
python3 scripts/evaluation_to_csv.py -c $1 -d $data_dir
