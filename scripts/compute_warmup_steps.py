import os
import argparse


def get_line_count(fpath: str) -> int:
    line_count = 0
    with open(fpath) as fin:
        for _ in fin:
            line_count += 1
    return line_count


def calc_warmup_steps(num_instances: int, num_epochs: int, batch_size: int,
                      warmup_ratio: float) -> int:
    return ()


def main(cmd_args: argparse.Namespace) -> None:
    if not cmd_args.corpus and not cmd_args.sum:
        total_steps = {}
        for dir_name in os.listdir(cmd_args.data):
            train_path = os.path.join(cmd_args.data, dir_name, 'train.jsonl')
            if os.path.exists(train_path):
                num_inst = get_line_count(train_path)
                total_steps[dir_name] = int(cmd_args.num_epochs * num_inst / cmd_args.batch_size)
        warmup_steps = {dir_name: int(cmd_args.ratio * steps) for dir_name, steps in total_steps.items()}
        for dir_name in sorted(total_steps):
            print(f'{dir_name}: [{total_steps[dir_name]}] / [{warmup_steps[dir_name]}]')
    elif cmd_args.corpus:
        train_path = os.path.join(cmd_args.data, cmd_args.corpus, 'train.jsonl')
        if os.path.exists(train_path):
            num_instances = get_line_count(train_path)
            total_steps = int(cmd_args.num_epochs * num_instances / cmd_args.batch_size)
            warmup_steps = int(cmd_args.ratio * total_steps)
            print(f'{cmd_args.corpus}: [{total_steps}] / [{warmup_steps}]')
            if cmd_args.output:
                with open(cmd_args.output, 'w') as f:
                    f.write(str(warmup_steps))
    elif cmd_args.sum:
        total_steps = {}
        for dir_name in os.listdir(cmd_args.data):
            train_path = os.path.join(cmd_args.data, dir_name, 'train.jsonl')
            if os.path.exists(train_path):
                num_inst = get_line_count(train_path)
                total_steps[dir_name] = int(cmd_args.num_epochs * num_inst / cmd_args.batch_size)
        sum_total_steps = sum(total_steps.values())
        sum_warmup_steps = int(cmd_args.ratio * sum_total_steps)
        print(f'Sum: [{sum_total_steps}] / [{sum_warmup_steps}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ratio', type=float, help='Ratio of warmup steps in comparison to '
                                                          'entire train set.')
    parser.add_argument('-d', '--data', help='Path to data-dir.')
    parser.add_argument('-c', '--corpus', default=None,
                        help='Compute only warmup steps for the given corpus.')
    parser.add_argument('-s', '--sum', nargs='+', default=None,
                        help='Compute summed warmup steps for all given corpora.')
    parser.add_argument('-o', '--output',
                        help='Path to output file. Only used in combination with -c.')
    parser.add_argument('-e', '--num_epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(cmd_args=args)
