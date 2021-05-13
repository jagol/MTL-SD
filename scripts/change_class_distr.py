import os
import json
import argparse
import random
from collections import defaultdict
from typing import Set, Dict, Union, List

from compute_target_distr_given_kurtosis_entropy import compute_target_distr


"""
Script to change/rebalance class distributions for one or many given 
datasets. The dataset is assumed to be in the data-dir of this 
repository and to be in allennlp-compatible jsonl format.

If no maximum number of instances for the balanced corpus is given, 
creates a balanced dataset.

Usage:
python3 change_class_distr.py -c <corpus1> <corpus2> -b -m 10000 -D <data-dir> -l <label-type>
"""


def get_corpus_path(data_dir: str, corpus: str) -> Union[str, None]:
    corpus_path = None
    for dir_name in os.listdir(data_dir):
        if dir_name == corpus:
            corpus_path = os.path.join(data_dir, dir_name)
    return corpus_path


def get_labels(corpus_path: str, label_type: str) -> Set[str]:
    train_path = os.path.join(corpus_path, 'train.jsonl')
    labels = set()
    with open(train_path) as fin:
        for line in fin:
            labels.add(json.loads(line)[label_type])
    return labels


def load_labels(corpus_path: str, label_type: str) -> List[str]:
    labels = []
    with open(os.path.join(corpus_path, 'train.jsonl')) as fin:
        for line in fin:
            labels.append(json.loads(line)[label_type])
    return labels


def change_class_distr(corpus_path: str, target_distr: Dict[str, float], max_ds_size: int,
                       label_type: str, out_id: str, seed: int = 4) -> None:
    """Change class distribution by upsampling classes until target distribution is reached."""
    random.seed(seed)
    instances_by_label = defaultdict(list)  # {label: list of instances}
    num_inst_by_label_before = {}  # {label: num instances before balancing}
    num_inst_by_label_after = {}  # {label: num instances after balancing}
    # categorize instances by label
    print('Categorize instances by label')
    with open(os.path.join(corpus_path, 'train.jsonl')) as fin:
        for line in fin:
            instance_dict = json.loads(line)
            instances_by_label[instance_dict[label_type]].append(instance_dict)
    print('Add instances by category')
    for label in instances_by_label:
        num_inst_by_label_before[label] = len(instances_by_label[label])
        # compute maximum number of instances for given label
        if max_ds_size > 0:
            num_inst_by_label_after[label] = int(round(max_ds_size * target_distr[label]))
        else:
            # then target distr must be balanced
            num_inst_by_label_after[label] = max(num_inst_by_label_before.values())
        if num_inst_by_label_before[label] < num_inst_by_label_after[label]:
            while len(instances_by_label[label]) < num_inst_by_label_after[label]:
                instances_by_label[label].extend(instances_by_label[label])
        random.shuffle(instances_by_label[label])
        instances_by_label[label] = instances_by_label[label][:num_inst_by_label_after[label]]
    # distr_repr = '_'.join([label + ':' + str(round(ratio, 2))
    #                        for label, ratio in target_distr.items()])
    # distr_repr += '_' + str(max_ds_size)
    out_path = os.path.join(corpus_path, f'train_{out_id}.jsonl')
    instances_out = []
    print('Merge instances')
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
        corpora = cmd_args.corpora
    elif cmd_args.kurtosis or cmd_args.entropy:
        corpora = cmd_args.corpora
        target_distr = {}
        for corpus in corpora:
            corpus_path = get_corpus_path(data_dir=args.data_dir, corpus=corpus)
            if not corpus_path:
                raise Exception(f'Corpus {corpus} not found in data-dir {args.data_dir}.')
            labels = load_labels(corpus_path=corpus_path, label_type=args.label_type)
            num_to_label = {i: label for i, label in enumerate(set(labels))}
            label_to_num = {label: i for i, label in num_to_label.items()}
            labels_num = [label_to_num[label] for label in labels]
            target_freqs_num = compute_target_distr(labels_num, cmd_args.kurtosis,
                                                    cmd_args.entropy)
            target_freqs = {num_to_label[num]: freq for num, freq in target_freqs_num.items()}
            total = sum(target_freqs.values())
            target_distr_cur_corpus = {label: freq / total for label, freq in target_freqs.items()}
            target_distr[corpus] = target_distr_cur_corpus
        print('Target distributions:')
        print(target_distr)
    else:
        target_distr = {}
        corpora = cmd_args.corpora
        if cmd_args.balanced:
            for corpus in cmd_args.corpora:
                corpus_path = get_corpus_path(data_dir=args.data_dir, corpus=corpus)
                labels = get_labels(corpus_path=corpus_path, label_type=args.label_type)
                class_freq = 1 / len(labels)
                target_distr[corpus] = {label: class_freq for label in labels}
        else:
            raise Exception('No distribution, kurtosis, entropy or "--balanced" was specified.')
    for corpus in corpora:
        corpus_path = get_corpus_path(data_dir=args.data_dir, corpus=corpus)
        print(f'Corpus path: {corpus_path}')
        if not corpus_path:
            raise Exception(f'Corpus {corpus} not found in data-dir {args.data_dir}.')
        if cmd_args.kurtosis:
            out_id = f'kurtosis_{cmd_args.kurtosis}_{cmd_args.max_ds_size}'
        elif cmd_args.entropy:
            out_id = f'entropy_{cmd_args.entropy}_{cmd_args.max_ds_size}'
        else:
            out_id = f'distr_{target_distr}_{cmd_args.max_ds_size}'
        change_class_distr(corpus_path=corpus_path, target_distr=target_distr[corpus],
                           max_ds_size=cmd_args.max_ds_size, label_type=cmd_args.label_type,
                           out_id=out_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', nargs='+')
    parser.add_argument('-b', '--balanced', action='store_true',
                        help='Completely balance the classes.')
    parser.add_argument('-m', '--max_ds_size', type=int, default=-1,
                        help='Maximum number of instances per corpus. '
                             '-1 to not define any upper bound.')
    parser.add_argument('-d', '--distribution',
                        help='json-dict in the form: {corpus-name: {label-name: ratio}}. '
                             'If this option is used, -c and -b are ignored.')
    parser.add_argument('-k', '--kurtosis', type=float,
                        help='If given, a class distribution is created that leads to the given '
                             'kurtosis.')
    parser.add_argument('-e', '--entropy', type=float,
                        help='If given, a class distribution is created that leads to the given '
                             'entropy.')
    parser.add_argument('-f', '--dfile', help='The same argument as -d, but read from a file.')
    parser.add_argument('-D', '--data_dir', help='Path to data/ directory.')
    parser.add_argument('-l', '--label_type', choices=['label_orig', 'label_uni'])
    args = parser.parse_args()
    main(args)
