import csv
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from allennlp.data.vocabulary import Vocabulary


def load_predictions(fpath: str) -> List[int]:
    predictions = []
    with open(fpath) as fin:
        reader = csv.reader(fin)
        for row in reader:
            class_probs = [float(num) for num in row]
            class_id = int(np.argmax(class_probs))
            predictions.append(class_id)
    return predictions


def load_labels(fpath_labels: str, dir_path_vocab: str, label_type: str
                ) -> Tuple[List[int], Dict[int, str]]:
    labels = []
    voc = Vocabulary.from_files(dir_path_vocab)
    label_mapping = {}  # {label_int: label_str}
    with open(fpath_labels) as fin:
        for line in fin:
            test_instance = json.loads(line)
            label_str = test_instance[label_type]
            task = test_instance['task']
            label_int = voc.get_token_index(label_str, namespace=task+'_labels')
            labels.append(label_int)
            if label_str not in label_mapping:
                label_mapping[label_int] = label_str
    return labels, label_mapping


def compute_metrics(predictions: List[int], labels: List[int], label_mapping: Dict[int, str]):
    labels = labels[:len(predictions)]
    f1_per_class = f1_score(y_true=labels, y_pred=predictions, average=None)
    precision_per_class = precision_score(y_true=labels, y_pred=predictions, average=None)
    recall_per_class = recall_score(y_true=labels, y_pred=predictions, average=None)
    precision_by_class = {label_mapping[i]: precision_per_class[i] for i in
                          range(len(precision_per_class))}
    recall_by_class = {label_mapping[i]: recall_per_class[i] for i in range(len(recall_per_class))}
    f1_by_class = {label_mapping[i]: f1_per_class[i] for i in range(len(f1_per_class))}
    return {
        'accuracy': accuracy_score(y_true=labels, y_pred=predictions),
        'f1_macro': f1_score(y_true=labels, y_pred=predictions, average='macro'),
        'precision_macro': precision_score(y_true=labels, y_pred=predictions, average='macro'),
        'recall_macro': recall_score(y_true=labels, y_pred=predictions, average='macro'),
        'f1_by_class': f1_by_class,
        'precision_by_class': precision_by_class,
        'recall_by_class': recall_by_class
    }


def write_to_file(metrics: Dict[str, float], fpath: str):
    metrics = {key: round(value, 3) for key, value in metrics.items()}
    with open(fpath, 'w') as fout:
        json.dump(metrics, fout, indent=4)


def main(cmd_args):
    predictions = load_predictions(cmd_args.predictions)
    labels, label_mapping = load_labels(cmd_args.labels, args.vocab, args.label_type)
    metrics = compute_metrics(predictions, labels, label_mapping)
    write_to_file(metrics, cmd_args.evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to extracted predictions, csv-file '
                             'produced by extract_results.py.')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to testdata in jsonl-format containing labels.')
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to evaluation file where metrics are written into.')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary dir.')
    parser.add_argument('--label_type', choices=['label_orig', 'label_uni'], default='label_orig',
                        help='Type of label to use.')
    args = parser.parse_args()
    main(args)
