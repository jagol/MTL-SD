import json
import os
import argparse
from collections import OrderedDict


"""Extract evaluation results and relevant config and print as csv-line.

Output columns:
config-name, f1-macro-avg-over-all-tasks, acc-avg-over-all-tasks,
f1-macro-task1, acc-task1, f1-macro-taskn, acc-taskn

Task columns are in alphabetic order by task name. 
Tasks that were not predicted on lead to empty columns.
"""


DATASETS = ['arc', 'ArgMin', 'FNC1', 'IAC', 'IBMCS', 'PERSPECTRUM', 'SCD', 'SemEval2016Task6',
            'SemEval2019Task7', 'Snopes']


def main(cmd_args: argparse.Namespace) -> None:
    eval_path = os.path.join(cmd_args.evaluation_file)
    with open(eval_path) as feval:
        eval_dict = json.load(feval)
    if cmd_args.benchmark:
        results = OrderedDict([(d, None) for d in DATASETS])
    else:
        results = OrderedDict([(d, None) for d in eval_dict])

    for dataset in results:
        if dataset in eval_dict:
            results[dataset] = {
                'f1_macro': eval_dict[dataset]['f1_macro'],
                'accuracy': eval_dict[dataset]['accuracy']
            }
        else:
            print(f'Warning: There are no results for dataset {dataset}')

    for dataset in eval_dict:
        if dataset not in results:
            print(f'Warning: Results for dataset {dataset} not used in output.')

    accs = [results[dataset]['accuracy'] for dataset in results]
    f1_macros = [results[dataset]['f1_macro'] for dataset in results]

    avg_acc = sum([n for n in accs if n]) / len([n for n in accs if n])
    avg_f1_macro = sum([n for n in f1_macros]) / len([n for n in f1_macros])

    header = ['avg_acc', 'avg_f1_macro']
    row = [avg_acc, avg_f1_macro]
    for dataset in results:
        header.append(dataset + '_acc')
        header.append(dataset + '_f1_macro')
        row.append(results[dataset]['accuracy'])
        row.append(results[dataset]['f1_macro'])

    print(','.join(header))
    print(','.join([str(r) for r in row]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation_file', help='Path to evaluation file.')
    parser.add_argument('-b', '--benchmark', action='store_true',
                        help='Use the benchmark datasets from "How robust is your stance '
                             'detection?"')
    args = parser.parse_args()
    main(args)
