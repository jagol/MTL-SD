import collections
import os
import random
import sys
import json
import pickle
import argparse
import string
import logging
from collections import defaultdict
from typing import Optional, Dict, List, Any

import numpy
import numpy as np
from nltk import ngrams as get_ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score


"""
Example call to precompute:
python3 scripts/baseline.py -M cache -c arc -d data/ -s results/baseline/ -t label_orig
Example call to train: 
python3 scripts/baseline.py -M train -c arc -d data/ -s results/baseline/ -t label_orig
Example call to evaluate on devset:
python3 scripts/baseline.py -M eval-dev -c arc -d data/ -s results/baseline/ -t label_orig
Example call to evaluate on testset:
python3 scripts/baseline.py -M eval-test -c arc -d data/ -s results/baseline/ -t label_orig
"""


STOPWORDS = set(stopwords.words('english'))


def get_logger(args: argparse.Namespace) -> logging.Logger:
    logFormatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    rootLogger = logging.getLogger('main')
    fname = os.path.join(args.serialization_dir, 'logs.txt')
    fileHandler = logging.FileHandler(os.path.join('.', fname))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel('INFO')
    return rootLogger


class Instance:
    """Class representing a single example/instance."""

    def __init__(self, idx: str, text1: str, text2: str, label_orig: str, label_uni: str,
                 kwargs: Dict[Any, Any]) -> None:
        self.idx = idx
        self.text1 = text1
        self.text2 = text2
        self.text1_tokens_clean = self.remove_punctuation(self.remove_stop_words(
            word_tokenize(self.text1)))
        self.text2_tokens_clean = self.remove_punctuation(self.remove_stop_words(
            word_tokenize(self.text2)))
        self.label_orig = label_orig
        self.label_uni = label_uni
        self.kwargs = kwargs
        self.text1_ngrams = defaultdict(list)  # {length: list of ngrams}
        self.text2_ngrams = defaultdict(list)  # {length: list of ngrams}

    def as_one_string(self):
        return f'{self.text1} {self.text2}'

    def set_ngrams(self, length: int):
        for ng in get_ngrams(self.text1_tokens_clean, length):
            self.text1_ngrams[length].append(':'.join(ng))
        for ng in get_ngrams(self.text2_tokens_clean, length):
            self.text2_ngrams[length].append(':'.join(ng))

    @staticmethod
    def remove_punctuation(tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in string.punctuation]

    @staticmethod
    def remove_stop_words(tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in STOPWORDS]


def load_jsonl(fpath: str, max_instances: Optional[int] = None) -> List[Instance]:
    logger.info(f'Load jsonl file: {fpath}')
    instances = []
    with open(fpath) as fin:
        for i, line in enumerate(fin, start=1):
            if max_instances and i > max_instances:
                break
            d = json.loads(line)
            inst = Instance(idx=d['id'], text1=d['text1'], text2=d['text2'],
                            label_orig=d['label_orig'], label_uni=d['label_uni'], kwargs=d)
            instances.append(inst)
    return instances


def load_pickled_data(fpath: str, max_instances: Optional[int] = None) -> List[Instance]:
    logger.info('Load train set with precomputed ngrams')
    with open(fpath, 'rb') as fin:
        instances = pickle.load(fin)
    if max_instances and max_instances < len(instances):
        return instances[:max_instances]
    return instances


def precompute_ngrams(train_instances: List[Instance]) -> None:
    for length in range(1, 4):
        logger.info(f'Precompute ngrams of length {length}')
        for instance in train_instances:
            instance.set_ngrams(length)


def compute_most_freq_ngrams(train_instances: List[Instance], max_number: int, length: int,
                             target: bool) -> List[str]:
    logger.info(f'Compute most frequent {length}grams.')
    ngrams = defaultdict(int)
    for instance in train_instances:
        if target:
            for ngram in instance.text1_ngrams[length]:
                ngrams[ngram] += 1
        else:
            for ngram in instance.text2_ngrams[length]:
                ngrams[ngram] += 1
    freq_sorted_ngrams = sorted(ngrams.items(), key=lambda pair: pair[1], reverse=True)
    return [item[0] for item in freq_sorted_ngrams[:max_number]]


# def map_to_dimsngrams: List[str]) -> Dict[str, int]:
#     mapping = {}
#     i = 0
#     for ng in ngram:
#         mapping[gram] = i
#         i += 1
#     return mapping


def map_labels_to_dims(train_instances: List[Instance], label_type: str = 'label_orig'
                       ) -> Dict[str, int]:
    logger.info('Map labels to dimensions')
    labels = set()
    for inst in train_instances:
        labels.add(getattr(inst, label_type))
    return {label: i for i, label in enumerate(labels)}


def save_to_pickle(data: Any, fpath_out: str) -> None:
    logger.info(f'Save to {fpath_out}')
    with open(fpath_out, 'wb') as fout:
        pickle.dump(data, fout)


def save_dict(data: Dict[Any, Any], fpath_out: str) -> None:
    logger.info(f'Saving to {fpath_out}')
    with open(fpath_out, 'w') as fout:
        json.dump(data, fout)


def get_all_text1_ngrams(train_instances: List[Instance], length: int) -> List[str]:
    all_ngrams = []
    for instance in train_instances:
        all_ngrams.extend(instance.text1_ngrams[length])
    return list(set(all_ngrams))


def preprocess(args: argparse.Namespace) -> None:
    """Precompute and save ngrams in serialization directory.

    Args:
        args: command-line arguments.
    """
    data_dir = os.path.join(args.data_dir, args.corpus)
    train_file = os.path.join(data_dir, 'train.jsonl')
    train_instances = load_jsonl(train_file, max_instances=args.max_instances)
    precompute_ngrams(train_instances)
    text1_gram1 = compute_most_freq_ngrams(train_instances, max_number=args.max_1gram,
                                           length=1, target=True)
    text1_gram2 = compute_most_freq_ngrams(train_instances, max_number=args.max_2gram,
                                           length=2, target=True)
    text1_gram3 = compute_most_freq_ngrams(train_instances, max_number=args.max_3gram,
                                           length=3, target=True)
    text2_gram1 = compute_most_freq_ngrams(train_instances, max_number=args.max_1gram,
                                           length=1, target=False)
    text2_gram2 = compute_most_freq_ngrams(train_instances, max_number=args.max_2gram,
                                           length=2, target=False)
    text2_gram3 = compute_most_freq_ngrams(train_instances, max_number=args.max_3gram,
                                           length=3, target=False)
    all_ngrams = list(set(text1_gram1 + text1_gram2 + text1_gram3 + text2_gram1 + text2_gram2 +
                          text2_gram3))
    gram_to_dim_mapping = {ng: i for i, ng in enumerate(all_ngrams)}
    label_to_dim_mapping = map_labels_to_dims(train_instances)
    save_to_pickle(data=train_instances, fpath_out=os.path.join(
        args.serialization_dir, 'train_instances.pickle'))
    save_dict(data=gram_to_dim_mapping, fpath_out=os.path.join(args.serialization_dir,
                                                               'gram_mapping.json'))
    save_dict(data=label_to_dim_mapping, fpath_out=os.path.join(args.serialization_dir,
                                                                'label_mapping.json'))
    # save_dict(data=gram1, fpath_out=os.path.join(args.serialization_dir, '1grams.json'))
    # save_dict(data=gram2, fpath_out=os.path.join(args.serialization_dir, '2grams.json'))
    # save_dict(data=gram3, fpath_out=os.path.join(args.serialization_dir, '3grams.json'))


def vectorize_features(train_instances: List[Instance], serialization_dir: str) -> numpy.ndarray:
    logger.info('Vectorize features')
    gram_mapping = json.load(open(os.path.join(serialization_dir, 'gram_mapping.json')))
    num_feats = len(gram_mapping)
    feat_array = np.zeros((len(train_instances), num_feats))
    for i, inst in enumerate(train_instances):
        for length in inst.text1_ngrams:
            for ng in inst.text1_ngrams[length] + inst.text2_ngrams[length]:
                if ng in gram_mapping:
                    j = gram_mapping[ng]
                    feat_array[i][j] = 1
    return feat_array


def vectorize_labels(train_instances: List[Instance], serialization_dir: str, label_type: str
                     ) -> numpy.ndarray:
    logger.info('Vectorize labels')
    label_mapping = json.load(open(os.path.join(serialization_dir, 'label_mapping.json')))
    labels = []
    for i, inst in enumerate(train_instances):
        label = label_mapping[getattr(inst, label_type)]
        labels.append(label)
    return numpy.array(labels)


def train_svm(args: argparse.Namespace, svm_kwargs: Optional[Dict] = None) -> SVC:
    if svm_kwargs is None:
        svm_kwargs = {}
    train_file = os.path.join(args.serialization_dir, 'train_instances.pickle')
    train_instances = load_pickled_data(train_file, max_instances=args.max_instances)
    feat_vecs = vectorize_features(train_instances, args.serialization_dir)
    labels = vectorize_labels(train_instances, args.serialization_dir, args.label_type)
    logger.info('Fit SVM')
    svc = SVC(**svm_kwargs)
    svc.fit(feat_vecs, labels)
    logger.info('Save SVM')
    fpath = os.path.join(args.serialization_dir, 'svm_model.pickle')
    with open(fpath, 'wb') as fout:
        pickle.dump(svc, fout)
    return svc


def load_svm(serialization_dir: str) -> SVC:
    with open(os.path.join(serialization_dir, 'svm_model.pickle'), 'rb') as fin:
        return pickle.load(fin)


def compute_scores(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> Dict[str, float]:
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'recall': float(recall_score(y_true, y_pred, average='macro')),
        'precision': float(precision_score(y_true, y_pred, average='macro'))
    }


def compute_per_class_scores(y_true: numpy.ndarray, y_pred: numpy.ndarray
                             ) -> Dict[str, List[float]]:
    return {
        'f1_macro': [float(f) for f in f1_score(y_true, y_pred, average=None)],
        'f1_micro': [float(f) for f in f1_score(y_true, y_pred, average=None)],
        'recall': [float(f) for f in recall_score(y_true, y_pred, average=None)],
        'precision': [float(f) for f in precision_score(y_true, y_pred, average=None)]
    }


def evaluate(args: argparse.Namespace) -> None:
    svm = load_svm(args.serialization_dir)
    if args.mode == 'eval-dev':
        test_file = os.path.join(args.data_dir, args.corpus, 'dev.jsonl')
    elif args.mode == 'eval-test':
        test_file = os.path.join(args.data_dir, args.corpus, 'test.jsonl')
    else:
        raise Exception(f'Unexpected mode: {args.mode}')
    test_instances = load_jsonl(test_file, max_instances=args.max_instances)
    precompute_ngrams(test_instances)
    # test_instances = load_pickled_data(test_file, max_instances=args.max_instances)
    # save_to_pickle(data=test_instances, fpath_out=os.path.join(
    #     args.serialization_dir, f'{args.mode}_instances.pickle'))
    feat_vecs = vectorize_features(test_instances, args.serialization_dir)
    labels = vectorize_labels(test_instances, args.serialization_dir, args.label_type)
    logger.info('Predict')
    y_pred = svm.predict(feat_vecs)
    logger.info('Compute scores')
    scores = compute_scores(labels, y_pred)
    per_class_scores = compute_per_class_scores(labels, y_pred)
    fpath_scores = os.path.join(args.serialization_dir, f'{args.mode}_scores.json')
    fpath_per_class_scores = os.path.join(args.serialization_dir,
                                          f'{args.mode}_per_class_scores.json')
    with open(fpath_scores, 'w') as fout:
        json.dump(scores, fout, indent=4)
    with open(fpath_per_class_scores, 'w') as fout:
        json.dump(per_class_scores, fout, indent=4)
    logger.info(f'f1-macro: {scores["f1_macro"]}')
    logger.info(f'f1-micro: {scores["f1_micro"]}')
    logger.info(f'recall-macro: {scores["recall"]}')
    logger.info(f'precision-macro: {scores["precision"]}')


def load_true_test_labels(corpus: str, args: argparse.Namespace) -> List[str]:
    fpath_test = os.path.join(args.data_dir, corpus, 'test.jsonl')
    labels = []
    with open(fpath_test) as fin:
        for line in fin:
            d = json.loads(line)
            labels.append(d[args.label_type])
    return labels


def generate_random_labels(true_labels: List[str]) -> List[str]:
    distinct_labels = list(set(true_labels))
    random_labels = []
    for _ in true_labels:
        random_labels.append(random.choice(distinct_labels))
    return random_labels


def generate_majority_labels(corpus: str, args: argparse.Namespace) -> List[str]:
    fpath_train = os.path.join(args.data_dir, corpus, 'train.jsonl')
    fpath_test = os.path.join(args.data_dir, corpus, 'test.jsonl')
    labels = []
    with open(fpath_train) as fin:
        for line in fin:
            d = json.loads(line)
            labels.append(d[args.label_type])
    label_counts = collections.Counter(labels)
    majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
    test_count = 0
    with open(fpath_test) as fin:
        for _ in fin:
            test_count += 1
    pred_labels = test_count * [majority_label]
    return pred_labels


def main(args: argparse.Namespace) -> None:
    benchmark_corpora = ['arc', 'ArgMin', 'FNC1', 'IAC', 'IBMCS', 'PERSPECTRUM', 'SCD',
                         'SemEval2016Task6', 'SemEval2019Task7', 'Snopes']
    args.serialization_dir = args.serialization_dir.strip('/')
    args.max_instances = args.m__max_instances

    logger.info(f'Mode: {args.mode}')
    if args.mode == 'prepro':
        logger.info('Setting up serialization directory')
        if os.path.exists(args.serialization_dir) and len(os.listdir(args.serialization_dir)) > 1:
            msg = 'Serialzation directory "{}" exists already and is not empty.'
            raise Exception(msg.format(args.serialization_dir))
        up_dir = '/'.join(args.serialization_dir.split('/')[:-1])
        if not os.path.exists(args.serialization_dir) and os.path.exists(up_dir):
            os.mkdir(args.serialization_dir)
        logger.info(f'Serialization directory: {args.serialization_dir}')
        logger.info('prepro ngrams')
        preprocess(args)
    elif args.mode == 'train':
        if os.path.exists(os.path.join(args.serialization_dir, 'svm_model.pickle')):
            raise Exception(f'Model already exists in serialization directory.')
        logger.info('Train SVM')
        train_svm(args)
    elif args.mode == 'eval-dev' or args.mode == 'eval-test':
        logger.info('Evaluate SVM.')
        evaluate(args)
    elif args.mode == 'random':
        for corpus in benchmark_corpora:
            y_true = load_true_test_labels(corpus, args)
            y_pred = generate_random_labels(y_true)
            metrics = compute_scores(y_true, y_pred)
            fpath_out = os.path.join(args.serialization_dir, f'{corpus}_random_baseline.json')
            save_dict(metrics, fpath_out)
    elif args.mode == 'majority':
        for corpus in benchmark_corpora:
            y_true = load_true_test_labels(corpus, args)
            y_pred = generate_majority_labels(corpus, args)
            metrics = compute_scores(y_true, y_pred)
            fpath_out = os.path.join(args.serialization_dir, f'{corpus}_majority_baseline.json')
            save_dict(metrics, fpath_out)
    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--mode', choices=['prepro', 'train', 'eval-dev', 'eval-test',
                                                 'random', 'majority'])
    parser.add_argument('-c', '--corpus', help='Corpus to process.')
    parser.add_argument('-d', '--data_dir', help='Path to data directory.')
    parser.add_argument('-s', '--serialization_dir',
                        help='Path to directory where cached ngrams and model are saved.')
    parser.add_argument('-m' '--max_instances', default=None, type=int,
                        help='Max instances to load from train-file.')
    parser.add_argument('--max_1gram', type=int, default=5000)
    parser.add_argument('--max_2gram', type=int, default=5000)
    parser.add_argument('--max_3gram', type=int, default=5000)
    parser.add_argument('-t', '--label_type', choices=['label_orig', 'label_uni'])
    cmd_args = parser.parse_args()
    logger = get_logger(cmd_args)
    main(cmd_args)
