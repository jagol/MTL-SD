import json
import collections
import operator
import argparse
from collections import Counter
from typing import List, Optional, Dict

from compute_class_distr import compute_kurtosis
from compute_class_distr import compute_entropy


"""
Given a labels of a training set and a target kurtosis/entropy, compute a label 
distribution that has the given kurtosis/entropy.
"""


def load_labels(fpath: str) -> List[str]:
    labels = []
    with open(fpath) as fin:
        for line in fin:
            d = json.loads(line)
            labels.append(d['label_orig'])
    return labels


# *** KURTOSIS FUNCTIONS ***

def adjust_class_distr_to_kurtosis(labels: List[int], kurtosis: float,
                                   print_threshold: int = 1000) -> List[int]:
    cur_kurt = compute_kurtosis(labels)
    cur_labels = [lb for lb in labels]
    count = 0
    print(f'Starting kurtosis: {cur_kurt}')
    if cur_kurt < kurtosis:
        while cur_kurt < kurtosis:
            cur_labels = increase_kurtosis(cur_labels, step=1)
            cur_kurt = compute_kurtosis(cur_labels)
            count += 1
            if count % print_threshold == 0:
                print(f'Iterations: {count}, kurtosis: {cur_kurt:.3f}')
        else:
            print(f'Iterations: {count}, kurtosis: {cur_kurt:.3f}')
    elif cur_kurt > kurtosis:
        while cur_kurt > kurtosis:
            cur_labels = decrease_kurtosis(cur_labels, step=1)
            cur_kurt = compute_kurtosis(cur_labels)
            count += 1
            if count % print_threshold == 0:
                print(f'Iterations: {count}, kurtosis: {cur_kurt:.3f}')
        else:
            print(f'Iterations: {count}, kurtosis: {cur_kurt:.3f}')
    return cur_labels


def increase_kurtosis(labels: List[int], step: int = 1):
    class_distr = Counter(labels)
    maj_class = max(class_distr.items(), key=operator.itemgetter(1))[0]
    return labels + step * [maj_class]


def decrease_kurtosis(labels: List[int], step: int = 1):
    class_distr = Counter(labels)
    min_class = min(class_distr.items(), key=operator.itemgetter(1))[0]
    return labels + step * [min_class]


# *** ENTROPY FUNCTIONS ***

def adjust_class_distr_to_entropy(labels: List[int], entropy: float,
                                  print_threshold: int = 1000) -> List[int]:
    cur_entropy = compute_entropy(labels)
    cur_labels = [lb for lb in labels]
    count = 0
    print(f'Starting entropy: {cur_entropy}')
    if cur_entropy < entropy:
        while cur_entropy < entropy:
            cur_labels = increase_kurtosis(cur_labels, step=1)
            cur_entropy = compute_entropy(cur_labels)
            count += 1
            if count % print_threshold == 0:
                print(f'Iterations: {count}, entropy: {cur_entropy:.3f}')
        else:
            print(f'Iterations: {count}, entropy: {cur_entropy:.3f}')
    elif cur_entropy > entropy:
        while cur_entropy > entropy:
            cur_labels = decrease_kurtosis(cur_labels, step=1)
            cur_entropy = compute_entropy(cur_labels)
            count += 1
            if count % print_threshold == 0:
                print(f'Iterations: {count}, entropy: {cur_entropy:.3f}')
        else:
            print(f'Iterations: {count}, entropy: {cur_entropy:.3f}')
    return cur_labels


def increase_entropy(labels: List[str], step: int = 1):
    """Increase entropy by adding an instance to the smallest class."""
    class_distr = Counter(labels)
    min_class = min(class_distr.items(), key=operator.itemgetter(1))[0]
    return labels + step * [min_class]


def decrease_entropy(labels: List[str], step: int = 1):
    """Decrease entropy by adding an instance to the largest class."""
    class_distr = Counter(labels)
    maj_class = max(class_distr.items(), key=operator.itemgetter(1))[0]
    return labels + step * [maj_class]


# *** main ***


def compute_target_distr(labels: List[int], kurtosis: Optional[float], entropy: Optional[float]
                         ) -> Dict[int, float]:
    if kurtosis:
        target_labels = adjust_class_distr_to_kurtosis(labels, kurtosis)
    elif entropy:
        target_labels = adjust_class_distr_to_entropy(labels, entropy)
    else:
        raise Exception('Error! Either kurtosis or entropy must be specified.')
    target_distr = collections.Counter(target_labels)
    return target_distr


def main(args: argparse.Namespace) -> None:
    labels = load_labels(args.labels)
    num_to_label = {i: label for i, label in enumerate(set(labels))}
    label_to_num = {label: i for i, label in num_to_label.items()}
    labels_num = [label_to_num[label] for label in labels]
    target_distr_num = compute_target_distr(labels_num, args.kurtosis, args.entropy)
    target_distr_str = {num_to_label[num]: freq for num, freq in target_distr_num.items()}
    print(f'Writing output dict to: {args.output}')
    with open(args.output, 'w') as fin:
        json.dump(target_distr_str, fin)
    print(target_distr_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kurtosis', required=False, type=float, help='Target kurtosis.')
    parser.add_argument('-e', '--entropy', required=False, type=float, help='Target entropy.')
    parser.add_argument('-l', '--labels', help='Path to data file, from which labels are loaded.')
    parser.add_argument('-o', '--output', help='Path output file (json).')
    cmd_args = parser.parse_args()
    main(cmd_args)
