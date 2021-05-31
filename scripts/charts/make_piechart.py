import argparse
import json

import matplotlib.pyplot as plt
# plt.style.use('ggplot')


"""
Usage:
python3 scripts/make_piechart.py -i corpus_stats/stats_data-arc.json -o charts/arc_labels.png
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_in', help='Path to json file with corpus statistics.')
    parser.add_argument('-o', '--path_out', help='Path to where output png is written to.')
    args = parser.parse_args()
    return args


def draw_piechart(path_in: str, path_out: str) -> None:
    labels = []
    sizes = []
    with open(path_in) as fin:
        corpus_stats = json.load(fin)['total']['LABELS_ORIG']
        for label in corpus_stats:
            labels.append(label)
            sizes.append(corpus_stats[label][1])

    # colors
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    fig1, ax1 = plt.subplots()
    patches, texts, autotexts = ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%',
                                        startangle=0)
    for text in texts:
        text.set_color('grey')
        # text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('grey')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.setp(autotexts, fontsize=12, fontweight='bold')
    plt.setp(texts, fontsize=12, fontweight='bold')
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.savefig(path_out)


def main():
    args = parse_args()
    draw_piechart(args.path_in, args.path_out)


if __name__ == '__main__':
    main()
