import os
import csv
import argparse
import random
from typing import Dict, List, Union

# import numpy as np
# import bootstrapped.bootstrap as bs
# import bootstrapped.stats_functions as bs_stats


result_type = Dict[str, Union[float, str]]
# dict of form: {'value': value, 'name': name, 'label': label}


population_type = List[result_type]
# A population is a list of results.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', 'Path to input file contianing test set results in csv.')
    parser.add_argument('-i', '--iterations', help='Number of bootstrapping samples to compute.')
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
    mean_0 = sum(group_0) / len(group_0)
    mean_1 = sum(group_1) / len(group_1)
    diff = abs(mean_0 - mean_1)
    return diff


def load_population(args: argparse.Namespace) -> population_type:
    population = []
    with open(args.path) as fin:
        reader = csv.reader(fin)
        for name, value, label in reader:
            population.append(
                {
                    'name': name,
                    'value': value,
                    'label': label
                }
            )
        return population


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
