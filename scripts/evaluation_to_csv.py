import json
import os
import argparse


"""Extract evaluation results and relevant config and print as csv-line.

Output columns:
config-name, f1-macro-avg-over-all-tasks, acc-avg-over-all-tasks,
f1-macro-task1, acc-task1, f1-macro-taskn, acc-taskn

Task columns are in alphabetic order by task name. 
Tasks that were not predicted on lead to empty columns.
"""


def main(cmd_args: argparse.Namespace) -> None:
    header = ['avg_f1_macro', 'avg_accuracy'] + os.listdir(cmd_args.data_dir)
    eval_path = os.path.join('results/', cmd_args.config, 'evaluation.json')
    with open(eval_path) as feval:
        eval_dict = json.load(feval)
    f1_macros = []
    accs = []
    for task in header:
        if task in eval_dict:
            f1_macros.append(eval_dict[task]['f1_macro'])
            accs.append(eval_dict[task]['accuracy'])
        else:
            f1_macros.append('')
            accs.append('')
    avg_f1_macro = sum([n for n in f1_macros if n]) / len([n for n in f1_macros if n])
    avg_acc = sum([n for n in accs if n]) / len([n for n in accs if n])
    results = []
    for f1, acc in zip(f1_macros, accs):
        results.append(f1)
        results.append(acc)
    print(','.join(header))
    print(','.join([str(avg_f1_macro), str(avg_acc)] + [str(r) for r in results]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Name of config.')
    parser.add_argument('-d', '--data_dir', help='Path to data folder.')
    args = parser.parse_args()
    main(args)
