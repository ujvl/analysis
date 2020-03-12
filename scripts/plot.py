from __future__ import print_function
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def main(args):
    data = pd.read_csv(args.fname, sep=args.delim)
    plotter_cls(args.plt)().plot(data, args)
    decorate_plot(args)
    logger.info("Showing plot...")
    plt.show()


def plotter_cls(name):
    plotters = {
        "scatter": ScatterPlot,
        "line": ScatterPlot,
        "reg": RegressionPlot,
        "box": BoxPlot,
    }
    return plotters[name]


def decorate_plot(args):
    """
    Requests user input to add plot labels, title etc.
    """
    title = args.title or input("Title: ")
    plt.title(title)

    x_label = args.xlabel or input("x-axis label: ") or args.x_col
    y_label = args.ylabel or input("y-axis label: ") or args.y_col

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if args.x_max:
        plt.xlim(None, args.x_max)
    if args.y_max:
        plt.ylim(None, args.y_max)

    ax = plt.gca()
    if args.x_log:
        ax.set_xscale("symlog")
    if args.y_log:
        ax.set_yscale("symlog")


class ScatterPlot:
    """Scatter plot"""

    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.y_col, args.x_col]):
            cols = data.columns
            args.x_col, args.y_col, args.z_col = cols[0], cols[1], cols[2]
            logger.info("Using x=%s, y=%s, z=%s by default.", args.x_col, args.y_col, args.z_col)

        # Workaround since hue uses duck-typing for numerics.
        if args.z_col:
            data[args.z_col] = data[args.z_col].astype(str)
            data[args.z_col] = "$" + data[args.z_col] + "$"

        sns.set()
        plot = sns.relplot(
            x=args.x_col,
            y=args.y_col,
            hue=args.z_col,
            kind=args.plt,
            data=data,
        )


class RegressionPlot:
    """Scatter plot with fitted function"""

    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.y_col, args.x_col]):
            cols = data.columns
            args.x_col, args.y_col, args.z_col = cols[0], cols[1], cols[2]
            logger.info("Using x=%s, y=%s, z=%s by default.", args.x_col, args.y_col, args.z_col)

        sns.set()
        plot = sns.lmplot(x=args.x_col, y=args.y_col, hue=args.z_col, data=data)


class BoxPlot:
    """Box plot"""

    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.y_col]):
            cols = data.columns
            args.x_col, args.y_col = cols[0], cols[1]
            logger.info("Using x=%s, y=%s by default.", args.x_col, args.y_col)

        p = self.p
        data = data[[args.x_col, args.y_col]]
        aggs = data.groupby(args.x_col).agg(
            ["mean", "std", p(25), p(50), p(75), p(90), p(95), p(99)]
        )
        logger.info(aggs)

        sns.set()
        boxplt = sns.boxplot(x=args.x_col, y=args.y_col, data=data, palette="Blues", width=0.35);

    def p(self, n):
        def p_(x):
            return np.percentile(x, n)
        p_.__name__ = 'p_%s' % n
        return p_


def parse():
    parser = argparse.ArgumentParser(description='plot some dataz')
    # Data processing
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--plt', type=str, choices=["scatter", "line", "reg", "box"], default="scatter")
    parser.add_argument('--x-col', type=str)
    parser.add_argument('--y-col', type=str)
    parser.add_argument('--z-col', type=str)
    parser.add_argument('--delim', type=str, default=',')
    # Decorate plot.
    parser.add_argument('--x-max', type=int)
    parser.add_argument('--y-max', type=int)
    parser.add_argument('--x-log', action="store_true")
    parser.add_argument('--y-log', action="store_true")
    parser.add_argument('--xlabel', type=str)
    parser.add_argument('--ylabel', type=str)
    parser.add_argument('--title', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    logger.info("Args: %s", args)
    main(args)
