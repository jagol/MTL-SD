import argparse
import os
import json
from typing import List


benchmark_corpora = ['arc', 'ArgMin', 'FNC1', 'IAC', 'IBMCS', 'PERSPECTRUM', 'SCD',
                     'SemEval2016Task6', 'SemEval2019Task7', 'Snopes']


def as_percentages(scores: List[float]) -> List[float]:
    return [round(100 * s, 1) for s in scores]


def main(args: argparse.Namespace) -> None:
    scores = []
    for corpus in benchmark_corpora:
        fpath = os.path.join(args.results_dir, f'baseline_{corpus}', f'{args.mode}_scores.json')
        with open(fpath) as fin:
            d = json.load(fin)
        scores.append(d['f1_micro'])
        scores.append(d['f1_macro'])
    print(benchmark_corpora)
    print(as_percentages(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dir')
    parser.add_argument('-M', '--mode', choices=['eval-dev', 'eval-test'])
    cmd_args = parser.parse_args()
    main(cmd_args)
