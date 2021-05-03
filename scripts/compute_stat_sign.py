import csv
import argparse
import random
from typing import Dict, List, Union, Tuple


result_type = Dict[str, Union[float, str]]
# dict of form: {'value': value, 'name': name, 'label': label}


population_type = List[result_type]
# A population is a list of results.


label_mapping = {
    'arc': 'h',
    'ArgMin': 'l',
    'FNC1': 'h',
    'IAC': 'l',
    'IBMCS': 'l',
    'PERSPECTRUM': 'l',
    'SCD': 'l',
    'SemEval2016Task6': 'l',
    'SemEval2019Task7': 'h',
    'Snopes': 'l'
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
                        help='Path to input file contianing test set results in csv.')
    parser.add_argument('-i', '--iterations', type=int,
                        help='Number of bootstrapping samples to compute.')
    parser.add_argument('-v', '--value', type=int, help='Column index for result value.')
    parser.add_argument('-n', '--name', type=int, help='Column index for result name.')
    parser.add_argument('-b', '--begin',
                        help='Only consider rows/results whose name begins with given string.')
    parser.add_argument('-e', '--end',
                        help='Only consider rows/results whose name ends with given string.')
    return parser.parse_args()


def bootstrap_test_statistics(population: population_type,
                              args: argparse.Namespace
                              ) -> List[float]:
    labels = list(set([p['label'] for p in population]))
    assert len(labels) == 2
    freq_label_0 = sum([1 for p in population if p['label'] == labels[0]])
    freq_label_1 = sum([1 for p in population if p['label'] == labels[1]])
    test_statistics = []
    for i in range(args.iterations):
        chosen_0 = []
        for _ in range(freq_label_0):
            chosen_0.append(random.choice(population))
        chosen_1 = []
        for _ in range(freq_label_1):
            chosen_1.append(random.choice(population))
        test_statistics.append(compute_test_statistic(chosen_0, chosen_1))
    return test_statistics


def compute_test_statistic(group_0: population_type,
                           group_1: population_type) -> float:
    values_0 = [float(i['value']) for i in group_0]
    values_1 = [float(i['value']) for i in group_1]
    mean_0 = sum(values_0) / len(values_0)
    mean_1 = sum(values_1) / len(values_1)
    diff = abs(mean_0 - mean_1)
    return diff


def load_population(args: argparse.Namespace) -> population_type:
    population = []
    with open(args.path) as fin:
        reader = csv.reader(fin)
        for row in reader:
            name = row[args.name]
            if args.begin and not name.startswith(args.begin):
                continue
            if args.end and not name.endswith(args.end):
                continue
            if 'shtl' in name:
                continue
            if '_wo_' in name:
                continue
            population.append(
                {
                    'name': name,
                    'value': row[args.value],
                    'label': get_label_from_name(name)
                }
            )
    return population


def get_label_from_name(name: str) -> str:
    aux_corpus = name.split('_')[1]
    return label_mapping[aux_corpus]


def compute_p_value(bootstrapped_test_stat: List[float], observerd_test_stat: float
                    ) -> Tuple[float, float]:
    over_observed = len([t for t in bootstrapped_test_stat if t >= observerd_test_stat])
    return over_observed / len(bootstrapped_test_stat), over_observed


def main():
    args = parse_args()
    population = load_population(args)
    # compute bootstrapped test statistics
    bs_test_statistics = bootstrap_test_statistics(population, args)

    # compute observed test statistics
    label_0, label_1 = list(set([p['label'] for p in population]))
    observed_test_statistic = compute_test_statistic(
        [p for p in population if p['label'] == label_0],
        [p for p in population if p['label'] == label_1]
    )
    p_val, over_observed = compute_p_value(bs_test_statistics, observed_test_statistic)
    print(f'Number of bootstrapped samples: {len(bs_test_statistics)}')
    print(f'Number of bootstrapped samples over observed sample: {over_observed}')
    print(f'P value: {p_val}')


if __name__ == '__main__':
    main()
