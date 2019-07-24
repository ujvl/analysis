from __future__ import print_function
from scipy.optimize import curve_fit
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    
    x_data, y_data = load(args)
    plot_data(x_data, y_data, scatter=not args.no_scatter) 
    decorate_plot(x_data, y_data)

def build_arg_parser():
    parser = argparse.ArgumentParser(description='plot some dataz')
    parser.add_argument('--fname', type=str)
    parser.add_argument('--no-scatter', action='store_true', default=False)
    parser.add_argument('--delim', type=str, default=None)
    parser.add_argument('--x-col', type=int, default=0)
    parser.add_argument('--y-col', type=int, default=1)
    return parser

def plot_data(x_data, y_data, scatter=True):
    if scatter:
        plt.scatter(x_data, y_data)
    plt.plot(x_data, y_data)

def load(args):
    x_data = np.loadtxt(fname=args.fname, delimiter=args.delim, usecols=((args.x_col,)))
    y_data = np.loadtxt(fname=args.fname, delimiter=args.delim, usecols=((args.y_col,)))
    print("[Sanity Check]")
    print("X length, min, max: ", len(x_data), np.amin(x_data), np.amax(x_data))
    print("Y length, min, max: ", len(y_data), np.amin(y_data), np.amax(y_data))
    assert len(x_data) ==len(y_data) 
    return x_data, y_data

def decorate_plot(x_data, y_data):
    """
    Requests user input to add plot labels, title etc.
    """
    title = input("Title: ")
    x_label = input("x-axis label: ")
    y_label = input("y-axis label: ")

    x_min = 0
    x_max = 1.1 * np.amax(x_data) # float(input("x-axis max: "))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_min, x_max)
    plt.show()

if __name__ == '__main__':
    main()
