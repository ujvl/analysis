from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(args):
    data = {}
    if len(args.fname) > 1:
        raise NotImplementedError("Need to implement multiple-file plot.")
    data = pd.read_csv(args.fname[0], sep=args.delim)
    sns.set()
    sns.relplot(x=args.x_col, y=args.y_col, hue=args.label_col, kind="line", data=data)
    decorate_plot(args)
    plt.show()

def decorate_plot(args):
    """
    Requests user input to add plot labels, title etc.
    """
    title = args.title or input("Title: ")
    plt.title(title)

    x_label = args.xlabel or input("x-axis label: ")
    y_label = args.ylabel or input("y-axis label: ")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if args.x_max:
        plt.xlim(None, args.x_max)
    if args.y_max:
        plt.ylim(None, args.y_max)


def parse():
    parser = argparse.ArgumentParser(description='plot some dataz')
    # Data processing
    parser.add_argument('--fname', nargs="*", type=str)
    parser.add_argument('--x-col', type=str)
    parser.add_argument('--y-col', type=str)
    parser.add_argument('--label-col', type=str)
    parser.add_argument('--delim', type=str, default=',')
    # Decorate plot.
    parser.add_argument('--no-scatter', action='store_true', default=False)
    parser.add_argument('--no-line', action='store_true', default=False)
    parser.add_argument('--x-max', type=int, default=None)
    parser.add_argument('--y-max', type=int, default=None)
    parser.add_argument('--xlabel', type=str, default=None)
    parser.add_argument('--ylabel', type=str, default=None)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    if args.label_col is not None and len(args.fname) > 1:
        raise Exception('Label data by column data or by file')
    return args


if __name__ == '__main__':
    args = parse()
    print(args)
    main(args)
