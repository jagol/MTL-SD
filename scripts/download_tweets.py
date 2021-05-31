import argparse
import csv
import json
import time
from typing import Dict

from twitter import Api, TwitterError


"""
https://github.zhaw.ch/vode/gswid2020.git served as a basis for this script.
"""


def get_keys(path: str) -> Dict[str, str]:
    with open(path) as fin:
        keys = json.load(fin)
    return {
        'consumer_key': keys['api_key'],
        'consumer_secret': keys['api_secret'],
        'access_token_key': keys['access_token_key'],
        'access_token_secret': keys['access_token_secret'],
    }


def main(cmd_args: argparse.Namespace):
    api = Api(**get_keys(cmd_args.keys))
    with open(cmd_args.inpath) as fin, open(cmd_args.outpath, 'w') as fout:
        reader = csv.reader(fin, delimiter=cmd_args.separator)  # find better sol for delimiter
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            tweet_id = row[cmd_args.column]
            time.sleep(1.1)
            print(f'fetching tweet: {i + 1}')
            try:
                status = api.GetStatus(tweet_id)
                row_out = [tweet_id, status.full_text or status.text]
                writer.writerow(row_out)
            except TwitterError as e:
                print(f'failed to retrieve tweet with id: {tweet_id}')
                print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keys',
                        help='Path to file with consumer key on the first line, consumer_secret '
                             'on the second, access_token_key on the third and access_token_secret'
                             ' on the fourth.')
    parser.add_argument('-o', '--outpath', help='Path to output csv-file of with format: '
                                                'id KOMMA tweet')
    parser.add_argument('-i', '--inpath', help='Path to csv-file containing ids')
    parser.add_argument('-c', '--column', type=int, help='Column index of the input ids.')
    parser.add_argument('-s', '--separator', default=',',
                        help='Separator/Delimiter of input file.')
    args = parser.parse_args()
    main(args)
