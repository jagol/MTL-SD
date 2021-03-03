import os
import json
import argparse
import random
from collections import defaultdict
from typing import Set, Dict, Union


"""
Script to change/rebalance class distributions for one or many given 
datasets. The dataset is assumed to be in the data-dir of this 
repository and to be in allennlp-compatible jsonl format.

Usage:
python3 change_class_distr.py -c <corpus1> <corpus2> -b -m 10000 -D <data-dir> -l <label-type>
"""


def get_corpus_path(data_dir: str, corpus: str) -> Union[str, None]:
    corpus_path = None
    for lang in os.listdir(data_dir):
        for dir_name in os.listdir(os.path.join(data_dir, lang)):
            if dir_name == corpus:
                corpus_path = os.path.join(data_dir, lang, dir_name)
    return corpus_path


def get_labels(corpus_path: str, label_type: str) -> Set[str]:
    train_path = os.path.join(corpus_path, 'train.jsonl')
    labels = set()
    with open(train_path) as fin:
        for line in fin:
            labels.add(json.loads(line)[label_type])
    return labels


def change_class_distr(corpus_path: str, target_distr: Dict[str, float], max_instances: int,
                       label_type: str, seed: int = 4) -> None:
    random.seed(seed)
    instances_by_label = defaultdict(list)
    num_inst_by_label_before = {}
    num_inst_by_label_after = {}
    with open(os.path.join(corpus_path, 'train.jsonl')) as fin:
        for line in fin:
            instance_dict = json.loads(line)
            instances_by_label[instance_dict[label_type]].append(instance_dict)
    for label in instances_by_label:
        num_inst_by_label_before[label] = len(instances_by_label[label])
        random.shuffle(instances_by_label[label])
        # compute maximum number of instances for given label
        num_inst_by_label_after[label] = int(max_instances * target_distr[label])
        if num_inst_by_label_before[label] < num_inst_by_label_after[label]:
            raise Exception(f'For label {label} there are {num_inst_by_label_after[label]} '
                            f'instances needed but only {len(instances_by_label[label])} '
                            f'instances exist.')
        instances_by_label[label] = instances_by_label[label][:num_inst_by_label_after[label]]
    distr_repr = '_'.join([label + ':' + str(round(ratio, 1))
                           for label, ratio in target_distr.items()])
    distr_repr += '_' + str(max_instances)
    out_path = os.path.join(corpus_path, f'train_{distr_repr}.jsonl')
    instances_out = []
    for label, instances in instances_by_label.items():
        instances_out.extend(instances)
    random.shuffle(instances_out)
    print(f"Writing to file {out_path.split('/')[-1]}.")
    print(f'New Instances per class:')
    for label in num_inst_by_label_after:
        print(f'  {label}: {num_inst_by_label_after[label]}')
    print('Original number of instances per class:')
    for label in num_inst_by_label_before:
        print(f'  {label}: {num_inst_by_label_before[label]}')
    with open(out_path, 'w') as fout:
        for instance in instances_out:
            fout.write(json.dumps(instance)+'\n')


def main(cmd_args: argparse.Namespace) -> None:
    if cmd_args.distribution:
        target_distr = json.loads(args.distribution)
        corpora = target_distr.keys()
    elif cmd_args.dfile:
        target_distr = json.load(open(args.dfile))
        corpora = target_distr.keys()
    else:
        target_distr = {}
        if cmd_args.balanced:
            for corpus in cmd_args.corpora:
                corpus_path = get_corpus_path(data_dir=args.data_dir, corpus=corpus)
                labels = get_labels(corpus_path=corpus_path, label_type=args.label_type)
                class_freq = 1 / len(labels)
                target_distr[corpus] = {label: class_freq for label in labels}
        else:
            raise Exception('Neither a distribution nor --balanced was specified.')
    for corpus in cmd_args.corpora:
        corpus_path = get_corpus_path(data_dir=args.data_dir, corpus=corpus)
        if not corpus_path:
            raise Exception(f'Corpus {corpus} not found in data-dir {args.data_dir}.')
        change_class_distr(corpus_path=corpus_path, target_distr=target_distr[corpus],
                           max_instances=args.max, label_type=cmd_args.label_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', nargs='+')
    parser.add_argument('-b', '--balanced', action='store_true',
                        help='Completely balance the classes.')
    parser.add_argument('-m', '--max', type=int, help='Maximum number of instances per corpus.')
    parser.add_argument('-d', '--distribution',
                        help='json-dict in the form: {corpus-name: {label-name: ratio}}. '
                             'If this option is used, -c and -b are ignored.')
    parser.add_argument('-f', '--dfile', help='The same argument as -d, but read from a file.')
    parser.add_argument('-D', '--data_dir', help='Path to data/ directory.')
    parser.add_argument('-l', '--label_type', choices=['label_orig', 'label_uni'])
    args = parser.parse_args()
    main(args)
