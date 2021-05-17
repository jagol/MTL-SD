import csv
import json
import argparse
import os
from typing import List, Dict, Tuple, Union

import numpy as np
from sklearn.metrics import (f1_score, accuracy_score, recall_score, precision_score,
                             confusion_matrix)
from allennlp.data.vocabulary import Vocabulary


label_to_range = {
    'arc': {"disagree": [0, 0.33], "unrelated": [0.33, 0.67], "discuss": [0.33, 0.67], "agree": [0.67, 1.0]},
    'FNC1': {"disagree": [0, 0.33], "unrelated": [0.33, 0.67], "discuss": [0.33, 0.67], "agree": [0.67, 1.0]},
    'SemEval2019Task7': {"deny": [0, 0.33], "comment": [0.33, 0.67], "query": [0.33, 0.67], "support": [0.67, 1.0]}
}


def load_predictions(fpath: str, fname: str = None) -> List[int]:
    if fname:
        fpath = os.path.join(fpath, fname)
    predictions = []
    with open(fpath) as fin:
        for line in fin:
            d = json.loads(line)
            key = [k for k in d.keys() if k.endswith('probs')][0]
            class_id = int(np.argmax(d[key]))
            predictions.append(class_id)
    return predictions


def load_labels(corpora: List[str], fname_labels: str, dir_path_vocab: str, label_type: str,
                data_dir: str) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, str]]]:
    labels = {}
    label_mappings = {}
    for corpus in corpora:
        fpath_labels = os.path.join(data_dir, corpus, fname_labels)
        corpus_labels, label_mapping = load_corpus_labels(fpath_labels, dir_path_vocab, label_type)
        labels[corpus] = corpus_labels
        label_mappings[corpus] = label_mapping
    return labels, label_mappings


def load_corpus_labels(fpath_labels: str, dir_path_vocab: str, label_type: str
                       ) -> Tuple[List[int], Dict[int, str]]:
    labels = []
    voc = Vocabulary.from_files(dir_path_vocab)
    label_mapping = {}  # {label_int: label_str}
    with open(fpath_labels) as fin:
        for line in fin:
            test_instance = json.loads(line)
            label_str = test_instance[label_type]
            if label_type == 'label_orig':
                task = test_instance['task']
                label_int = voc.get_token_index(label_str, namespace=task + '_labels')
            elif label_type == 'label_uni':
                task = 'UNIFIED'
                # if 'regr' in dir_path_vocab:
                #     task += '_regr'
                # else:
                #     task += '_class'
                label_int = voc.get_token_index(label_str, namespace=task + '_labels')
            labels.append(label_int)
            if label_str not in label_mapping:
                label_mapping[label_int] = label_str
    return labels, label_mapping


def compute_metrics(predictions: Dict[str, List[int]],
                    labels: Dict[str, List[int]],
                    label_mappings: Dict[str, Dict[int, str]]
                    ) -> Tuple[Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                               Dict[str, List[str]], Dict[str, List[List[int]]]]:
    """
    Args:
        predictions: {corpus: list of predicted labels}
        labels: {corpus: list of gold-labels}
        label_mappings: {corpus: {label-int: label-str}}
    Returns:
        {corpus: {metric1: value, metric2: {submetric: value}}}
    """
    metrics = {}
    headers = {}
    confms = {}
    assert list(predictions.keys()) == list(labels.keys())
    for corpus in predictions.keys():
        corpus_preds = predictions[corpus]
        corpus_labels = labels[corpus]
        corpus_label_mapping = label_mappings[corpus]
        corpus_metrics, corpus_header, corpus_confm = compute_corpus_metrics(
            predictions=corpus_preds,
            labels=corpus_labels,
            label_mapping=corpus_label_mapping
        )
        metrics[corpus] = corpus_metrics
        headers[corpus] = corpus_header
        confms[corpus] = corpus_confm
    return metrics, headers, confms


def compute_corpus_metrics(predictions: List[int],
                           labels: List[int],
                           label_mapping: Dict[int, str]
                           ) -> Tuple[Dict[str, Union[float, Dict[str, float]]],
                                      List[str], List[List[int]]]:
    """
    Args:
        predictions: list of predicted labels
        labels: list of gold-labels
        label_mapping: {label-int: label-str}}
    Returns:
        {metric1: value, metric2: {submetric: value}}}
    """
    labels = labels[:len(predictions)]
    f1_per_class = f1_score(y_true=labels, y_pred=predictions, average=None)
    precision_per_class = precision_score(y_true=labels, y_pred=predictions, average=None)
    recall_per_class = recall_score(y_true=labels, y_pred=predictions, average=None)
    precision_by_class = {label_mapping[i]: precision_per_class[i] for i in
                          range(len(precision_per_class))}
    recall_by_class = {label_mapping[i]: recall_per_class[i] for i in range(len(recall_per_class))}
    f1_by_class = {label_mapping[i]: f1_per_class[i] for i in range(len(f1_per_class))}
    num_true_per_class = {label_mapping[i]: labels.count(i) for i in set(labels)}
    num_pred_per_class = {label_mapping[i]: predictions.count(i) for i in set(labels)}
    conf_header = [label_mapping[i] for i in sorted(set(labels))]
    conf_matrix = confusion_matrix(y_true=labels, y_pred=predictions,
                                   labels=[i for i in sorted(set(labels))])
    metrics = {
        'accuracy': accuracy_score(y_true=labels, y_pred=predictions),
        'f1_macro': f1_score(y_true=labels, y_pred=predictions, average='macro'),
        'precision_macro': precision_score(y_true=labels, y_pred=predictions, average='macro'),
        'recall_macro': recall_score(y_true=labels, y_pred=predictions, average='macro'),
        'f1_by_class': f1_by_class,
        'precision_by_class': precision_by_class,
        'recall_by_class': recall_by_class,
        'true_labels_per_class': num_true_per_class,
        'predicted_labels_per_class': num_pred_per_class
    }
    return metrics, conf_header, conf_matrix


def write_to_file(metrics: Dict[str, Dict[str, Union[float, Dict[str, float]]]], fpath: str):
    # before writing, round metric-values to 3 decimals
    metrics_rounded = {}
    for corpus in metrics:
        corpus_metrics = metrics[corpus]
        if corpus not in metrics_rounded:
            metrics_rounded[corpus] = {}
        for metric in corpus_metrics:
            if isinstance(corpus_metrics[metric], dict):
                submetrics = corpus_metrics[metric]
                if metric not in metrics_rounded[corpus]:
                    metrics_rounded[corpus][metric] = {}
                for submetric in submetrics:
                    metrics_rounded[corpus][metric][submetric] = round(submetrics[submetric], 3)
            else:
                metrics_rounded[corpus][metric] = round(corpus_metrics[metric], 3)

    with open(fpath, 'w') as fout:
        json.dump(metrics_rounded, fout, indent=4)


def write_confms_to_file(conf_matrices: Dict[str, List[List[int]]],
                         conf_headers: Dict[str, List[str]],
                         eval_path: str):
    dir_path = '/'.join(eval_path.split('/')[:-1])
    for corpus in conf_matrices:
        matrix = conf_matrices[corpus]
        header = conf_headers[corpus]
        out_path = os.path.join(dir_path, f'{corpus}_confusion_matrix.csv')
        with open(out_path, 'w') as fout:
            writer = csv.writer(fout)
            writer.writerow(['label'] + header)
            for i, row in enumerate(matrix):
                writer.writerow(['true_' + header[i]] + [str(num) for num in row])


def infer_pred_paths(pred_path: str, fname: str) -> Tuple[str, str]:
    if 'regr' in fname:
        fname_regr = fname
        fname_class = fname.replace('regr', 'class')
    elif 'class' in fname:
        fname_regr = fname.replace('class', 'regr')
        fname_class = fname
    else:
        raise Exception('Either "regr" or "class" must be in fname.')
    return os.path.join(pred_path, fname_class), os.path.join(pred_path, fname_regr)


def load_regr_predictions(regr_pred_path: str) -> List[int]:
    predictions = []
    with open(regr_pred_path) as fin:
        for line in fin:
            d = json.loads(line)
            idx = int(np.argmax(d['IBMCS_regr_probs']))
            predictions.append(idx)
    return predictions


def combine_preds(preds_class: List[int], preds_regr: List[int]) -> List[int]:
    """
    Only combine when corpus has more than 3 classes (and ignore class-file in other cases)
    Or always combine?
    """
    predictions = []
    for pred_cls, pred_regr in zip(preds_class, preds_regr):
        if pred_regr == 1:
            predictions.append(pred_cls)
        else:
            predictions.append(pred_regr)
    return predictions


def load_regr_class_predictions(pred_path: str, fname: str) -> List[int]:
    path_preds_class, path_preds_regr = infer_pred_paths(pred_path, fname)
    preds_class = load_predictions(path_preds_class)
    preds_regr = load_regr_predictions(path_preds_regr)
    return combine_preds(preds_class, preds_regr)


def main(cmd_args):
    predictions = {}
    if os.path.isdir(cmd_args.predictions):
        list_to_skip = []
        for fname in os.listdir(cmd_args.predictions):
            if fname in list_to_skip:
                continue
            if 'regr' in fname or 'class' in fname:
                if 'regr' in fname:
                    list_to_skip.append(fname.replace('regr', 'class'))
                elif 'class' in fname:
                    list_to_skip.append(fname.replace('class', 'regr'))
                corpus = fname.split('.')[0].split('_')[0]
                if corpus in label_to_range:
                    predictions[corpus] = load_regr_class_predictions(
                        cmd_args.predictions, fname=fname)
                else:
                    if 'regr' in fname:
                        predictions[corpus] = load_regr_predictions(
                            regr_pred_path=os.path.join(cmd_args.predictions, fname))
                    else:
                        continue
            else:
                corpus = fname.split('.')[0]
                predictions[corpus] = load_predictions(cmd_args.predictions, fname=fname)
    elif os.path.isfile(cmd_args.predictions):
        if 'regr' in cmd_args.predictions or 'class' in cmd_args.predictions:
            fname = cmd_args.predictions.split('/')[-1]
            # fname_regr = cmd_args.predictions
            corpus = fname.split('.')[0].split('_')[0]
            # class_preds = load_predictions(cmd_args.predictions)
            predictions[corpus] = load_regr_class_predictions(cmd_args.predictions, fname=fname)
        else:
            corpus = '_'.join(cmd_args.path.split('_')[:-1])
            predictions[corpus] = load_predictions(cmd_args.predictions)
    # corpora = [c.replace('_regr', '').replace('class', '') for c in predictions.keys()]
    labels, label_mappings = load_labels(
        corpora=list(predictions.keys()),
        fname_labels=cmd_args.labels,
        dir_path_vocab=args.vocab,
        label_type=args.label_type,
        data_dir=cmd_args.data_dir
    )
    metrics, conf_headers, conf_matrices = compute_metrics(predictions, labels, label_mappings)
    write_to_file(metrics, cmd_args.evaluation)
    write_confms_to_file(conf_matrices, conf_headers, cmd_args.evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions. Jsonl-file')
    parser.add_argument('--labels', type=str, required=True,
                        help='Name of test-files in jsonl format.')
    parser.add_argument('--evaluation', type=str, required=True,
                        help='Path to evaluation file where metrics are written into.')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary dir.')
    parser.add_argument('--label_type', choices=['label_orig', 'label_uni'], default='label_orig',
                        help='Type of label to use.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory.')
    args = parser.parse_args()
    main(args)
