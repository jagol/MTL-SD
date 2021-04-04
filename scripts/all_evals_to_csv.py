import os
import json
import csv
import argparse
from typing import Dict, Union


scores_type = Dict[str, Dict[str, Dict[str, float]]]
# {config-name: {dataset-name: {metric: score}}}


def extract_scores(results_dir: str, must_end_with: Union[str, None],
                   must_begin_with: Union[str, None], single: bool = False) -> scores_type:
    scores = {}
    eval_file = 'evaluation.json'
    print(f'Must begin with: {must_begin_with}')
    print(f'Must end with: {must_end_with}')
    for config_name in [fn for fn in os.listdir(results_dir) if fn not in ['optuna', 'trial.db']]:
        if single and not config_name.startswith('sT_'):
            continue
        if must_end_with and not config_name.endswith(must_end_with):
            continue
        if must_begin_with and not config_name.startswith(must_begin_with):
            continue
        print(f'processing config: {config_name}')
        eval_fpath = os.path.join(results_dir, config_name, eval_file)
        if not os.path.exists(eval_fpath):
            print(f'Warning: evaluation file does not exist for config {config_name}.')
            print(f'Skipping {config_name}.')
            continue
        evaluation = json.load(open(eval_fpath))
        config_scores = {}
        for dataset in evaluation:
            config_scores[dataset] = {}
            config_scores[dataset]['acc'] = evaluation[dataset]['accuracy']
            config_scores[dataset]['f1_macro'] = evaluation[dataset]['f1_macro']
        scores[config_name] = config_scores
    return scores


def write_to_csv(scores: scores_type, fpath_out: str) -> None:
    datasets = []
    for config_name in scores:
        for dsname in scores[config_name]:
            datasets.append(dsname + '_acc')
            datasets.append(dsname + '_f1_macro')
    header = ['config_name'] + sorted(set(datasets))
    rows = []
    for config_name, config_scores in scores.items():
        row = [config_name]
        for column_name in header[1:]:
            dataset = column_name.split('_')[0]
            if dataset in config_scores:
                if column_name.endswith('acc'):
                    row.append(round(config_scores[dataset]['acc'], 4))
                elif column_name.endswith('f1_macro'):
                    row.append(round(config_scores[dataset]['f1_macro'], 4))
                else:
                    raise Exception(f'Invalid column-name: {column_name}')
            else:
                row.append(0.0)
        rows.append(row)
    with open(fpath_out, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_single_scores_to_csv(scores: scores_type, fpath_out: str) -> None:
    with open(fpath_out, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['config-name', 'dataset', 'accuracy', 'f1_macro'])
        for config in scores:
            assert len(scores[config]) == 1
            dataset = list(scores[config].keys())[0]
            accuracy = scores[config][dataset]['acc']
            f1_macro = scores[config][dataset]['f1_macro']
            writer.writerow([config, dataset, accuracy, f1_macro])


def main(cmd_args: argparse.Namespace) -> None:
    # if cmd_args.single:
    #     scores = extract_scores(cmd_args.results, cmd_args.must_end_with,
    #                             cmd_args.must_begin_with, single=True)
    #     write_single_scores_to_csv(scores, cmd_args.output)
    # else:
    scores = extract_scores(cmd_args.results, cmd_args.must_end_with, cmd_args.must_begin_with,
                            cmd_args.single)
    write_to_csv(scores, cmd_args.output)
    print(f'Results written to: {cmd_args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results', help='Path to results-directory.')
    parser.add_argument('-o', '--output', help='Path to output file.')
    parser.add_argument('-s', '--single', action='store_true',
                        help='If true, only extract results of singel task models. Output format: '
                             'config-name KOMMA corpus-name KOMMA accuarcy KOMMA f1-macro')
    parser.add_argument('-m', '--must_end_with', required=False, default=None,
                        help='If given, only configs that end with given string are processed.')
    parser.add_argument('-b', '--must_begin_with', required=False, default=None,
                        help='If given, only configs that start with given string are processed.')
    args = parser.parse_args()
    main(args)
