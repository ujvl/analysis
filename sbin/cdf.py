import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(args):
    data = np.loadtxt(fname=args.fname, delimiter=args.delim, usecols=(args.col,))
    bin_edges, cdf = plot(data)
    decorate_plot(bin_edges, cdf, args)
    plt.show()

def plot(data):
    counts, bin_edges = np.histogram(data, bins=50, normed=True)
    cdf = np.cumsum(counts)
    cdf /= cdf[-1]
    return bin_edges, cdf

def decorate_plot(bin_edges, cdf, args):
    title = args.title or input("Title: ")
    x_label = args.xlabel or input("x-axis label: ")
    y_label = "F(x)"

    xlow = np.min(bin_edges[1:])
    xhi = np.max(bin_edges[1:])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(bin_edges[1:], cdf)

def build_arg_parser():
    parser = argparse.ArgumentParser(description='plot CDF')
    parser.add_argument('--fname', type=str)
    parser.add_argument('--delim', type=str)
    parser.add_argument('--col', type=int, default=0)
    parser.add_argument('--title', type=str)
    parser.add_argument('--xlabel', type=str)
    return parser

if __name__ == '__main__':
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    main(args)
