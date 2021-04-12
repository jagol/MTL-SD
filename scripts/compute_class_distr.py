import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, DefaultDict

"""Compute the class distribution of a given corpus.

Usage: python3 compute_class_distr.py -p <path_to_corpus>

<path_to_corpus> should be a path to a directory containing 'train.jsonl', 'dev.jsonl' and 
'test.jsonl' and produced by 'create_train_dev_test_files.py'.
"""


def compute_stats(file_path: str) -> Dict[str, DefaultDict[str, int]]:
    stats = {
        'LABELS_ORIG': defaultdict(int),
        'LABELS_UNIFIED': defaultdict(int)
    }
    with open(file_path) as fin:
        for line in fin:
            instance = json.loads(line)
            stats['LABELS_ORIG'][instance['label_orig']] += 1
            stats['LABELS_UNIFIED'][instance['label_uni']] += 1
    return stats


def aggregate_stats(list_of_stats: List[Dict[str, DefaultDict[str, int]]]
                    ) -> Dict[str, DefaultDict[str, int]]:
    total_stats = {'LABELS_ORIG': defaultdict(int), 'LABELS_UNIFIED': defaultdict(int)}
    for stats in list_of_stats:
        for label, freq in stats['LABELS_ORIG'].items():
            total_stats['LABELS_ORIG'][label] += freq
        for label, freq in stats['LABELS_UNIFIED'].items():
            total_stats['LABELS_UNIFIED'][label] += freq
    return total_stats


def add_percentages(stats: Dict[str, DefaultDict[str, int]]
                    ) -> Dict[str, Dict[str, Tuple[int, float]]]:
    with_percentages = {'LABELS_ORIG': {}, 'LABELS_UNIFIED': {}}
    total_labels_orig = sum([stats['LABELS_ORIG'][label] for label in stats['LABELS_ORIG']])
    total_labels_uni = sum([stats['LABELS_UNIFIED'][label] for label in stats['LABELS_UNIFIED']])
    for label, freq in stats['LABELS_ORIG'].items():
        with_percentages['LABELS_ORIG'][label] = (freq, freq / total_labels_orig)
    for label, freq in stats['LABELS_UNIFIED'].items():
        with_percentages['LABELS_UNIFIED'][label] = (freq, freq / total_labels_uni)
    return with_percentages


def add_total(stats: Dict[str, Dict[str, Tuple[int, float]]]
              ) -> Dict[str, Dict[str, Tuple[int, float]]]:
    """
    Add the total number of instances over all datasplits

    Args:
        stats: {label_type: {label: (abs-freq, rel-freq)}}
    """
    key = 'LABELS_ORIG'
    num_instances = 0
    for label in stats[key]:
        num_instances += stats[key][label][0]
    stats[key]['total_instances'] = (num_instances, 1.0)
    return stats


def print_stats(train_stats: Dict[str, Dict[str, Tuple[int, float]]],
                dev_stats: Dict[str, Dict[str, Tuple[int, float]]],
                test_stats: Dict[str, Dict[str, Tuple[int, float]]],
                total_stats: Dict[str, Dict[str, Tuple[int, float]]],
                per_set: bool = False,
                unified: bool = False
                ) -> None:
    if per_set:
        print('*** TRAINING SET STATISTICS ***')
        print_set_stats(train_stats, unified)
        print('*** DEVELOPMENT SET STATISTICS ***')
        print_set_stats(dev_stats, unified)
        print('*** TEST SET STATISTICS ***')
        print_set_stats(test_stats, unified)
    print('*** TOTAL STATISTICS ***')
    print_set_stats(total_stats, unified)


def print_set_stats(subset: Dict[str, Dict[str, Tuple[int, float]]], unified: bool = False) -> None:
    print('LABELS ORIGINAL:')
    for label, (freq, perc) in subset['LABELS_ORIG'].items():
        print(f'  {label}: {freq} / {100*perc:.1f}%')
    if unified:
        print('LABELS UNIFIED:')
        for label, (freq, perc) in subset['LABELS_UNIFIED'].items():
            print(f'  {label}: {freq} / {100*perc:.1f}%')


def print_all_labels(train_stats: Dict[str, Dict[str, Tuple[int, float]]],
                     dev_stats: Dict[str, Dict[str, Tuple[int, float]]],
                     test_stats: Dict[str, Dict[str, Tuple[int, float]]],
                     total_stats: Dict[str, Dict[str, Tuple[int, float]]],
                     per_set=False,
                     unified=False
                     ) -> None:
    labels_train = set(label for label in train_stats['LABELS_ORIG'])
    labels_dev = set(label for label in dev_stats['LABELS_ORIG'])
    labels_test = set(label for label in test_stats['LABELS_ORIG'])
    labels_total = set(label for label in total_stats['LABELS_ORIG'])
    if per_set:
        print(f"Labels train: {', '.join(labels_train)}")
        print(f"Labels dev: {', '.join(labels_dev)}")
        print(f"Labels test: {', '.join(labels_test)}")
    print(f"Labels in train-dev-test: {', '.join(labels_total)}")
    if unified:
        labels_train = set(label for label in train_stats['LABELS_UNIFIED'])
        labels_dev = set(label for label in train_stats['LABELS_UNIFIED'])
        labels_test = set(label for label in train_stats['LABELS_UNIFIED'])
        labels_total = set(label for label in train_stats['LABELS_UNIFIED'])
        if per_set:
            print(f"Labels train: {', '.join(labels_train)}")
            print(f"Labels dev: {', '.join(labels_dev)}")
            print(f"Labels test: {', '.join(labels_test)}")
        print(f"Labels in train, dev, test: {', '.join(labels_total)}")


def main(args: argparse.Namespace) -> None:
    train_stats = compute_stats(os.path.join(args.path, 'train.jsonl'))
    dev_stats = compute_stats(os.path.join(args.path, 'dev.jsonl'))
    test_stats = compute_stats(os.path.join(args.path, 'test.jsonl'))
    total_stats_perc = add_total(add_percentages(aggregate_stats(
        [train_stats, dev_stats, test_stats])))
    train_stats_perc = add_total(add_percentages(train_stats))
    dev_stats_perc = add_total(add_percentages(dev_stats))
    test_stats_perc = add_total(add_percentages(test_stats))
    print_stats(train_stats_perc, dev_stats_perc, test_stats_perc, total_stats_perc,
                per_set=args.per_set, unified=args.unified)
    print_all_labels(train_stats_perc, dev_stats_perc, test_stats_perc, total_stats_perc,
                     per_set=args.per_set)
    path_name = args.path.replace('/', '-')
    json.dump(
        {
            'train': train_stats_perc,
            'dev': dev_stats_perc,
            'test': test_stats_perc,
            'total': total_stats_perc
        },
        open(os.path.join(cmd_args.output_dir, f'stats_{path_name}.json'), 'w'),
        indent=4
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='Path to directory containing the jsonl-files.')
    parser.add_argument('-u', '--unified', action='store_true')
    parser.add_argument('-s', '--per_set', action='store_true')
    parser.add_argument('-o', '--output_dir', help='Path to output directory.')
    cmd_args = parser.parse_args()
    main(cmd_args)
