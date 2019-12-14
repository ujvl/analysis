from __future__ import print_function
from scipy.optimize import curve_fit
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import sys

def main(args):
    if args.group_col is not None:
        # TODO use pandas to group data or consider seaborn
        return 
    for fname in args.fname:
        label = file_path_to_label(fname)
        x_data, y_data = load(fname, args.x_col, args.y_col, args.delim)
        plot_data(x_data, y_data, args.no_line, args.no_scatter, label)
    decorate_plot(args)
    plt.show()

def file_path_to_label(path):
    return path.split('/')[-1].split('.')[0]

def plot_data(x_data, y_data, no_line, no_scatter, label):
    scatter = not args.no_scatter
    line = not no_line
    # plt.xticks(x_data)
    if scatter:
        plt.scatter(x_data, y_data, marker='.')
    if line:
        plt.plot(x_data, y_data, linewidth=0.5, label=label)

def load(fname, x_col, y_col, delim):
    x_data = np.loadtxt(fname=fname, delimiter=delim, usecols=((x_col,)))
    y_data = np.loadtxt(fname=fname, delimiter=delim, usecols=((y_col,)))
    print("[Sanity Check]")
    print("X len, min, max: ", len(x_data), np.amin(x_data), np.amax(x_data))
    print("Y len, min, max: ", len(y_data), np.amin(y_data), np.amax(y_data))
    assert len(x_data) == len(y_data)
    return x_data, y_data

def decorate_plot(args):
    """
    Requests user input to add plot labels, title etc.
    """
    title = args.title or input("Title: ")
    x_label = args.xlabel or input("x-axis label: ")
    y_label = args.ylabel or input("y-axis label: ")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().legend()

    if args.x_max:
        plt.xlim(None, args.x_max)
    if args.y_max:
        plt.ylim(None, args.y_max)

class Parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='plot some dataz')
        self.parser.add_argument('--fname', nargs="*", type=str)
        self.parser.add_argument('--no-scatter', action='store_true', default=False)
        self.parser.add_argument('--no-line', action='store_true', default=False)
        self.parser.add_argument('--delim', type=str, default=',')
        self.parser.add_argument('--x-col', type=int, default=0)
        self.parser.add_argument('--y-col', type=int, default=1)
        self.parser.add_argument('--x-max', type=int, default=None)
        self.parser.add_argument('--y-max', type=int, default=None)
        self.parser.add_argument('--group-col', type=int, default=None)
        self.parser.add_argument('--xlabel', type=str, default=None)
        self.parser.add_argument('--ylabel', type=str, default=None)
        self.parser.add_argument('--title', type=str, default=None)

    def parse(self):
        args = self.parser.parse_args()
        if args.group_col is not None and len(args.fname) > 1:
            raise Exception('Group data by column data or by file')
        return args

if __name__ == '__main__':
    args = Parser().parse()
    print(args)
    main(args)
