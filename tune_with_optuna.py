import argparse

import optuna


def objective(trial: optuna.Trial) -> float:
    trial.suggest_float('lr', 3e-6, 3e-3, log=True)
    trial.suggest_float('weight_decay', 0.005, 0.05)
    trial.suggest_int('num_epochs', 1, 5)
    trial.suggest_int('warmup_steps', 0, 1000)
    # trial.suggest_categorical('optimizer', ['huggingface_adamw', 'adamw', 'adamax'])

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,
        config_file=args.config,
        serialization_dir=f'./results/optuna/{trial.number}_{args.config.split("/")[-1]}',
        metrics=f'best_validation_{args.dataset_name}_f1_macro'
    )
    return executor.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to configuration file to use.')
    parser.add_argument('-s', '--study_name', help='Name of study in results-db.')
    # parser.add_argument('-m', '--metric', help='Validation metric to be used by optuna.')
    parser.add_argument('-t', '--timeout', type=int, default=60*60*20,  # 20hours
                        help='Max time to run experiments.')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of jobs/processes to run in parallel.')
    parser.add_argument('-n', '--n_trials', type=int, default=50, help='Number of trials to run.')
    parser.add_argument('-d', '--database', default='sqlite:///results/trial.db',
                        help='Path to (sqlite3) db where results are saved.')
    parser.add_argument('-N', '--dataset_name',
                        help='Name of dataset that is used. (Name used to construct name of '
                             'metric to use for allennlp.)')
    args = parser.parse_args()
    study = optuna.create_study(
        storage=args.database,  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name=args.study_name,
        direction='maximize',
    )

    study.optimize(
        objective,
        n_jobs=args.jobs,  # number of processes in parallel execution
        n_trials=args.n_trials,  # number of trials to train a model
        timeout=args.timeout,  # threshold for executing time (sec)
    )
