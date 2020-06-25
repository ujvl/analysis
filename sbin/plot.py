#!/usr/bin/env python
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
    logger.debug("Args: %s", args)
    data = pd.read_csv(args.fname, sep=args.delim)
    data = process_data(data)
    PLOTTERS[args.plt]().plot(data, args)
    decorate_plot(args)
    logger.info("Showing plot...")
    plt.show()


def process_data(data):
    return data


def decorate_plot(args):
    """Requests user input to add plot labels, title etc."""
    title = args.title or input("Title: ")
    plt.title(title)

    x_label = args.xlabel or input("x-axis label: ") or args.x_col
    y_label = args.ylabel or input("y-axis label: ") or args.y_col

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if args.x_max:
        plt.xlim(0, args.x_max)
    if args.y_max:
        plt.ylim(0, args.y_max)

    ax = plt.gca()
    if args.x_scale:
        ax.set_xscale(args.x_scale)
    if args.y_scale:
        ax.set_yscale(args.y_scale)


def p(n):
    def p_(x):
        return np.percentile(x, n)

    p_.__name__ = "p_%s" % n
    return p_


class ScatterPlot:
    """Scatter plot"""

    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.y_col, args.z_col]):
            cols = data.columns
            args.x_col, args.y_col = cols[0], cols[1]
            args.z_col = cols[2] if len(cols) > 2 else None
            logger.info("Using x=%s, y=%s, z=%s by default.", args.x_col,
                        args.y_col, args.z_col)

        # Workaround since hue uses duck-typing for numerics.
        if args.z_col and data.dtypes[args.z_col] == int:
            data[args.z_col] = data[args.z_col].astype(str)
            data[args.z_col] = "$" + data[args.z_col] + "$"

        sns.set()
        sns.relplot(
            x=args.x_col,
            y=args.y_col,
            hue=args.z_col,
            kind=args.plt,
            data=data,
        )


class RegressionPlot:
    """Scatter plot with fitted function"""

    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.y_col]):
            cols = data.columns
            args.x_col, args.y_col = cols[0], cols[1]
            args.z_col = cols[2] if len(cols) > 2 else None
            logger.info("Using x=%s, y=%s, z=%s by default.", args.x_col,
                        args.y_col, args.z_col)

        sns.set()
        sns.lmplot(x=args.x_col, y=args.y_col, hue=args.z_col, data=data)


class BoxPlot:
    """Box plot"""

    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.y_col]):
            cols = data.columns
            args.x_col, args.y_col = cols[0], cols[1]
            logger.info("Using x=%s, y=%s by default.", args.x_col, args.y_col)

        data = data[[args.x_col, args.y_col]]
        aggs = data.groupby(args.x_col).agg(
            ["mean", "std", p(25), p(50), p(75), p(90), p(95), p(99)])
        logger.info(aggs)

        sns.set()
        sns.boxplot(
            x=args.x_col, y=args.y_col, data=data, palette="Blues", width=0.35)


class CDFPlot:
    def plot(self, data, args):
        if all(col is None for col in [args.x_col, args.z_col]):
            cols = data.columns
            args.x_col, args.z_col = [cols[0]], cols[1]
            logger.info("Using x=%s, z=%s by default.", args.x_col, args.z_col)

        # Workaround since hue uses duck-typing for numerics.
        if args.z_col and data.dtypes[args.z_col] == int:
            data[args.z_col] = data[args.z_col].astype(str)
            data[args.z_col] = "$" + data[args.z_col] + "$"

        kwargs = {"cumulative": True}
        aggs = data.groupby(args.z_col)

        sns.set()
        for agg in aggs:
            for x_col in args.x_col:
                samples = agg[1][x_col]
                sns.distplot(
                    samples,
                    hist=False,
                    hist_kws=kwargs,
                    kde_kws=kwargs,
                    bins=50,
                    label=f"{x_col} - {agg[0]}")


PLOTTERS = {
    "scatter": ScatterPlot,
    "line": ScatterPlot,
    "reg": RegressionPlot,
    "box": BoxPlot,
    "cdf": CDFPlot,
}


def parse():
    parser = argparse.ArgumentParser(description="plot some dataz")
    # Data processing
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument(
        "--plt", type=str, choices=list(PLOTTERS.keys()), default="scatter")
    parser.add_argument(
        "--x-col", nargs="+", help="note: this can only be a list for CDF")
    parser.add_argument("--y-col", nargs="+")
    parser.add_argument("--z-col")
    parser.add_argument("--delim", default=",")
    # Decorate plot. Omitting some of these may trigger input request.
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--x-scale")
    parser.add_argument("--y-scale")
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    parser.add_argument("--title")
    args = parser.parse_args()
    return args


def clean_validate(args):
    err = None
    if args.plt != "cdf":
        if len(args.x_col) > 1:
            err = f"--x-col must be len 1 for {args.plt}"
        else:
            # set as str since other plotters don't expect list
            args.x_col = args.x_col[0]
    elif args.plt == "cdf" and args.y_col:
        err = f"--y-col must not be set for {args.plt}"

    args.y_col = args.y_col[0] if args.y_col else None

    if err:
        raise ValueError(err)
    return args


if __name__ == "__main__":
    main(clean_validate(parse()))
