import os
import json
import argparse
from typing import Set


def load_targets(fpath: str) -> Set[str]:
    targets = []
    with open(fpath) as fin:
        for line in fin:
            target = json.loads(line)['text1']
            targets.append(target)
    return set(targets)


def main(args: argparse.Namespace) -> None:
    benchmark_corpora = ['arc', 'ArgMin', 'FNC1', 'IAC', 'IBMCS', 'PERSPECTRUM', 'SCD',
                         'SemEval2016Task6', 'SemEval2019Task7', 'Snopes']
    for corpus in benchmark_corpora:
        train_path = os.path.join(args.data, corpus, 'train.jsonl')
        dev_path = os.path.join(args.data, corpus, 'dev.jsonl')
        test_path = os.path.join(args.data, corpus, 'test.jsonl')
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            continue
        train_targets = load_targets(train_path)
        dev_targets = load_targets(dev_path)
        test_targets = load_targets(test_path)
        num_train_in_test = len([1 for t in train_targets if t in test_targets])
        num_train_targets = len(train_targets)
        num_dev_targets = len(dev_targets)
        num_test_targets = len(test_targets)
        num_total_targets = len(train_targets | dev_targets | test_targets)
        print(10 * '*')
        print(f'{corpus}:')
        print(f'num train targets: {num_train_targets}')
        print(f'num dev targets: {num_dev_targets}')
        print(f'num test targets: {num_test_targets}')
        print(f'total number of targets: {num_total_targets}')
        print(f'num train targets in test: {num_train_in_test}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Path to data dir.')
    cmd_args = parser.parse_args()
    main(cmd_args)