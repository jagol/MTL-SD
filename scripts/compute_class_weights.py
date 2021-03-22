import json
import os
import argparse
from collections import defaultdict
from typing import Dict, DefaultDict


"""Script to compute class weights that can be used in loss functions."""


def get_class_counts(fpath: str, label_type: str) -> DefaultDict[str, int]:
    class_counts = defaultdict(int)
    with open(fpath) as fin:
        for line in fin:
            fields = json.loads(line)
            class_counts[fields[label_type]] += 1
    return class_counts


def compute_class_weights(class_counts: DefaultDict[str, int]
                          ) -> Dict[str, float]:
    class_weights = {}
    num_instances = sum(class_counts.values())
    for class_name in class_counts:
        class_weights[class_name] = 1 - class_counts[class_name] / num_instances
    return class_weights


def write_to_file(corpora_clsws: Dict[str, Dict[str, float]],
                  fpath_out: str) -> None:
    """Output format: json file of form
    {
        <corpus_name>: {<class_name>: weight, ...},
        ...
    }
    """
    for corpus in corpora_clsws:
        for cls_name in corpora_clsws[corpus]:
            corpora_clsws[corpus][cls_name] = round(corpora_clsws[corpus][cls_name], 3)
    with open(fpath_out, 'w') as fout:
        json.dump(corpora_clsws, fout, indent=4)


def main(cmd_args: argparse.Namespace) -> None:
    all_class_weights = {}
    for corpus_dir in os.listdir(cmd_args.data_dir):
        train_file = os.path.join(cmd_args.data_dir, corpus_dir, 'train.jsonl')
        if not os.path.exists(train_file):
            print(f'Warning: No train file for {corpus_dir} exists. Skipping.')
            continue
        class_counts = get_class_counts(train_file, cmd_args.label_type)
        class_weights = compute_class_weights(class_counts)
        all_class_weights[corpus_dir] = class_weights
    write_to_file(all_class_weights, cmd_args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='Path to data directory.')
    parser.add_argument('-o', '--output', help='Path to output file.')
    parser.add_argument('-t', '--label_type', choices=['label_orig', 'label_uni'],
                        help='Label type to use.')
    args = parser.parse_args()
    main(args)
