import argparse
import os
import json
from typing import *

import allennlp.data.vocabulary
import numpy as np
from allennlp.data.vocabulary import Vocabulary


"""
Usage: 
python3 scripts/collect_test_instances_given_predictions.py -d /srv/scratch0/jgoldz/mthesis/data/ -t /srv/scratch0/jgoldz/mthesis/results_archive/test_aux_task_850 -c PERPSECTRUM -o all_wrong.jsonl
"""


corpora_sent = ['IMDB', 'SemEval2016Task4A', 'SemEval2016Task4B', 'SemEval2016Task4C', 'SST']
corpora_infer = ['MSRPara', 'MultiNLI', 'QQP', 'RTE', 'STSB', 'WNLI']


def parse_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Path to data directory.')
    parser.add_argument('-t', '--test_dir', help='Name of directory in data-dir containing tests.')
    parser.add_argument('-c', '--corpus', help='Name of corpus to analyze predictions for.')
    parser.add_argument('-o', '--output', help='Path to output file.')
    args = parser.parse_args()
    return args


def load_test_instances_for_corpus(data_dir: str, corpus: str) -> List[Dict[str, str]]:
    instances = []
    with open(os.path.join(data_dir, corpus, 'test.jsonl')) as fin:
        for line in fin:
            instances.append(json.loads(line))
    return instances


# def load_test_instances(data_dir: str, corpora: str) -> List[]:


def load_predictions_for_test(test_dir: str) -> Dict[str, List[str]]:
    out = {}
    configs = os.listdir(test_dir)
    for config in configs:
        preds = load_prediction_for_config(test_dir=test_dir, config=config)
        out[config] = preds
    return out


def load_prediction_for_config(test_dir: str, config: str) -> List[str]:
    predictions = []
    pred_dir = os.path.join(test_dir, config, 'predictions')
    pred_files = os.listdir(pred_dir)
    assert len(pred_files) == 1
    pred_file = pred_files[0]
    corpus = pred_file.split('.')[0]
    voc_dir = os.path.join(test_dir, config, 'vocabulary')
    voc = Vocabulary.from_files(voc_dir)
    with open(os.path.join(pred_dir, pred_file)) as fin:
        for line in fin:
            pred = json.loads(line)[f'{corpus}_probs']
            label_int = int(np.argmax(pred))
            label_str = voc.get_token_from_index(label_int, namespace=corpus + '_labels')
            predictions.append(label_str)
    return predictions


def write_to_outfile(test_instances: List[Dict[str, str]], path_out: str) -> None:
    with open(path_out, 'w') as fout:
        for instance in test_instances:
            fout.write(json.dumps(instance) + '\n')


def filter_instances(test_instances: List[Dict[str, str]],
                     predictions: Dict[str, List[str]],
                     pred_filter='all_false') -> List[Dict[str, str]]:
    filtered_instances = []
    for i, instance in enumerate(test_instances):
        instance_preds = []
        for model in predictions:
            for j, pred in enumerate(predictions[model]):
                if i == j:
                    instance_preds.append(pred)
                    break
        if pred_filter == 'all_false':
            preds_true_false = [pred == instance['label_orig'] for pred in instance_preds]
            if not all(preds_true_false):
                filtered_instances.append(instance)
    return filtered_instances


def main() -> None:
    args = parse_cmd_args()
    print('Load test instances for corpus')
    test_instances_perspectrum = load_test_instances_for_corpus(data_dir=args.data,
                                                                corpus='PERSPECTRUM')
    # test_instances_arc = load_test_instances(data_dir=args.data, corpus='arc')
    print('Load predictions for test')
    predictions = load_predictions_for_test(test_dir=args.test_dir)
    print('Filter instances')
    filtered_instances = filter_instances(test_instances_perspectrum, predictions)
    print('Write to outfile')
    write_to_outfile(filtered_instances, args.output)


# 1. get instances predicted wrong by all models with aux tasks


if __name__ == '__main__':
    main()
