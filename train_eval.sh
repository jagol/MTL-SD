#!/bin/bash
# $1: config-name
# $2: name of test-files e.g. "test.jsonl"
# $3: cuda-device
# $4: label-type, "label_orig" or "label_uni" (check that it's same as specified in StanceDetectionReader)
# $5: local or rattle
# $6: warmup ratio
# $7: batch-size
# $8: num epochs
# $9: "all" or list of corpora to test, if no corpus arg is given predicting is skipped.
set -e

path_config="configs/${1}.jsonnet"

if [[ "$9" == "all" ]]
then
  tasks=("arc" "ArgMin" "FNC1" "IAC" "IBMCS" "PERSPECTRUM" "SCD" "SemEval2016Task6" "SemEval2019Task7" "Snopes")
elif [[ "$9" == "" ]]
then
  tasks=()
else
  tasks=${@:9};
fi

if [[ "$5" == "local" ]]
then
  data_dir="./data"
  results_dir="./results"
elif [[ "$5" == "rattle" ]]
then
  data_dir="/srv/scratch0/jgoldz/mthesis/data"
  results_dir="/srv/scratch0/jgoldz/mthesis/results"
else
  echo "Unknown location: ${5}"
  exit 1
fi

echo "Compute warmup steps in case they are needed..."
python3 scripts/compute_warmup_steps.py -c $9 -d $data_dir -o "configs/warmup_steps.txt" -r $6 -b $7 -e $8
echo "Start training..."
allennlp train $path_config --include-package mtl_sd -o "{'trainer.cuda_device': $3}" -s "${results_dir}/$1"
echo "Training finished."

echo "Setup evaluation."
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
python3 scripts/evaluation_to_csv.py -e "${results_dir}/${1}/evaluation.json" -b

rm "${results_dir}/*.th"