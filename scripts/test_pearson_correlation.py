import os
import csv
import argparse
from typing import Optional, Tuple, List, Dict, Any

from scipy import stats


def load_all_results() -> Dict[str, List[Any]]:
    results = {}  # {config-name: results-row}
    fnames_evaluation = os.listdir('evaluations')
    for fn in fnames_evaluation:
        if not fn.endswith('.csv'):
            continue
        if 'kurtosis' in fn:
            continue
        if 'test_aux_dataset' not in fn:
            continue
        with open(os.path.join('evaluations', fn)) as fin:
            reader = csv.reader(fin)
            for row in reader:
                results[row[0]] = row
    return results


def search_for_k1_values(config_name: str, results: Dict[str, List[Any]]
                         ) -> List[Any]:
    first_corpus, second_corpus = config_name.split('_')[:2]
    shtl = 'shtl' in config_name
    length = None
    if '850' in config_name:
        length = '850'
    elif '2800' in config_name:
        length = '2800'
    for cn in results:
        if not cn.startswith(first_corpus):
            continue
        if second_corpus not in cn:
            continue
        if shtl:
            if 'shtl' not in cn:
                continue
        if length:
            if length not in cn:
                continue
        return results[cn]


def load_data(path_in: str, begin: Optional[str], end: Optional[str], results: Dict[str, List[Any]]
              ) -> Tuple[List[float], List[float], List[float]]:
    kurtosis_values = []
    acc_values = []
    f1_values = []
    with open(path_in) as fin:
        reader = csv.reader(fin)
        for config_name, acc1, f11, acc2, f12 in reader:
            if begin and not config_name.startswith(begin):
                continue
            if end and not config_name.endswith(end):
                continue
            if not acc_values and config_name.startswith('PERSPECTRUM'):
                # PERSPECTRUM results with kurtosis = 1
                cn, acc1, f11, acc2, f12 = search_for_k1_values(config_name, results)
                acc = float(acc1) if float(acc1) else float(acc2)
                f1 = float(f11) if float(f11) else float(f12)
                acc_values.append(acc)
                f1_values.append(f1)
                kurtosis_values.append(1.0)
                print(f'Added results value: {cn}')
            kurtosis = float(config_name.split('_')[2].strip('K'))
            acc = float(acc1) if float(acc1) else float(acc2)
            f1 = float(f11) if float(f11) else float(f12)
            kurtosis_values.append(kurtosis)
            acc_values.append(acc)
            f1_values.append(f1)
            print(config_name)
    return kurtosis_values, acc_values, f1_values


def parse_cmd_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_in', help='Path to input file in csv-format as ouputted by '
                                                'all_evals_to_csv.py')
    parser.add_argument('-b', '--begin', help='If given, all config_names that do not begin with '
                                              'given string are ignored.')
    parser.add_argument('-e', '--end', help='If given, all config_names that do not end with '
                                            'given string are ignored.')
    return parser.parse_args()


def calc_pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    pearson_corr, p_val = stats.pearsonr(x, y)
    print(f'Pearson correlation coefficient: {pearson_corr:.3f}')
    print(f'P value: {p_val:.3f}')
    return pearson_corr, p_val


def main() -> None:
    args = parse_cmd_args()
    results = load_all_results()
    kurtosis_values, acc_values, f1_values = load_data(args.path_in, args.begin, args.end, results)
    print('Correlate kurtosis with accuarcy:')
    calc_pearson(kurtosis_values, acc_values)
    print('Correlate kurtosis with f1-score:')
    calc_pearson(kurtosis_values, f1_values)


if __name__ == '__main__':
    main()
