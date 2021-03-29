import os
import argparse


def get_line_count(fpath: str) -> int:
    line_count = 0
    with open(fpath) as fin:
        for _ in fin:
            line_count += 1
    return line_count


def main(cmd_args: argparse.Namespace) -> None:
    if not cmd_args.corpus and not cmd_args.sum:
        line_counts = {}
        for dir_name in os.listdir(cmd_args.data):
            train_path = os.path.join(cmd_args.data, dir_name, 'train.jsonl')
            if os.path.exists(train_path):
                line_counts[dir_name] = get_line_count(train_path)
        warmup_steps = {dir_name: int(cmd_args.ratio*line_count) for dir_name, line_count in
                        line_counts.items()}
        for dir_name in sorted(line_counts):
            print(f'{dir_name}: [{line_counts[dir_name]}] / [{warmup_steps[dir_name]}]')
    elif cmd_args.corpus:
        train_path = os.path.join(cmd_args.data, cmd_args.corpus, 'train.jsonl')
        if os.path.exists(train_path):
            line_count = get_line_count(train_path)
            warmup_steps = int(cmd_args.ratio + line_count)
            print(f'{cmd_args.corpus}: [{line_count}] / [{warmup_steps}]')
            if cmd_args.output:
                with open(cmd_args.output, 'w') as f:
                    f.write(str(warmup_steps))
    elif cmd_args.sum:
        line_counts = {}
        for dir_name in os.listdir(cmd_args.data):
            train_path = os.path.join(cmd_args.data, dir_name, 'train.jsonl')
            if os.path.exists(train_path):
                line_counts[dir_name] = get_line_count(train_path)
        warmup_steps = {dir_name: int(cmd_args.ratio * line_count) for dir_name, line_count in
                        line_counts.items()}
        print(f'Sum: [{sum(line_counts.values())}] / [{sum(warmup_steps.values())}]')


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
    args = parser.parse_args()
    main(cmd_args=args)
