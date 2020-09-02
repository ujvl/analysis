#!/usr/bin/env python
from __future__ import print_function
import argparse
from inspect import getmembers, isfunction
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def main(args):
    logger.debug("Args: %s", args)
    processor = PROCESSORS[args.process]
    data = processor(pd.read_csv(args.csv, sep=args.delim), args)
    PLOTTERS[args.plt](data, args)
    decorate(args)
    logger.info("Showing plot...")
    plt.show()


def decorate(args):
    """Requests user input to add plot labels, title etc."""
    title = args.title or input("Title: ")
    x_label = args.xlabel or input("x-axis label: ") or args.x
    y_label = args.ylabel or input("y-axis label: ") or args.y

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if args.x_max:
        plt.xlim(args.x_min, args.x_max)
    if args.y_max:
        plt.ylim(args.y_min, args.y_max)

    if args.x_scale:
        plt.gca().set_xscale(args.x_scale)
    if args.y_scale:
        plt.gca().set_yscale(args.y_scale)


class DataProcessor:
    """Data processing namespace."""

    @staticmethod
    def identity(data: DataFrame, args) -> DataFrame:
        """Identity."""
        return data


class Plot:
    """Data plotting namespace."""

    @staticmethod
    def line(data, args):
        """Scatter plot"""
        Plot.scatter(data, args)

    @staticmethod
    def scatter(data, args):
        """Scatter plot"""
        if all(col is None for col in [args.x, args.y, args.z]):
            cols = data.columns
            args.x, args.y = cols[0], cols[1]
            args.z = cols[2] if len(cols) > 2 else None
            logger.info("Using x=%s, y=%s, z=%s by default.", args.x,
                        args.y, args.z)

        # Workaround since hue uses duck-typing for numerics.
        if args.z and data.dtypes[args.z] == int:
            data[args.z] = data[args.z].astype(str)
            data[args.z] = "$" + data[args.z] + "$"

        sns.set()

        sns.relplot(
            x=args.x,
            y=args.y,
            hue=args.z,
            kind=args.plt,
            data=data,
        )

    @staticmethod
    def reg(data, args):
        """Scatter plot with fitted function"""
        if all(col is None for col in [args.x, args.y]):
            cols = data.columns
            args.x, args.y = cols[0], cols[1]
            args.z = cols[2] if len(cols) > 2 else None
            logger.info("Using x=%s, y=%s, z=%s by default.", args.x,
                        args.y, args.z)

        sns.set()
        sns.lmplot(x=args.x, y=args.y, hue=args.z, data=data)

    @staticmethod
    def box(data, args):
        """Box plot"""
        if all(col is None for col in [args.x, args.y]):
            cols = data.columns
            args.x, args.y = cols[0], cols[1]
            logger.info("Using x=%s, y=%s by default.", args.x, args.y)

        data = data[[args.x, args.y]]
        aggs = data.groupby(args.x).agg(
            ["mean", "std", p(25), p(50), p(75), p(90), p(95), p(99)])
        logger.info(aggs)

        sns.set()
        sns.boxplot(
            x=args.x, y=args.y, data=data, palette="Blues", width=0.35)

    @staticmethod
    def bar(data, args):
        """Bar plot"""
        if all(col is None for col in [args.x, args.y]):
            cols = data.columns
            args.x, args.y = cols[0], cols[1]
            logger.info("Using x=%s, y=%s by default.", args.x, args.y)

        sns.barplot(x=args.x, y=args.y, hue=args.z, data=data)

    @staticmethod
    def cdf(data, args):
        """CDF plot"""
        if all(col is None for col in [args.x, args.z]):
            cols = data.columns
            args.x, args.z = [cols[0]], cols[1]
            logger.info("Using x=%s, z=%s by default.", args.x, args.z)

        # Workaround since hue uses duck-typing for numerics.
        if args.z and data.dtypes[args.z] == int:
            data[args.z] = data[args.z].astype(str)
            data[args.z] = "$" + data[args.z] + "$"

        kwargs = {"cumulative": True}
        aggs = data.groupby(args.z)

        sns.set()
        for agg in aggs:
            for x_col in args.x:
                samples = agg[1][x_col]
                sns.distplot(
                    samples,
                    hist=False,
                    hist_kws=kwargs,
                    kde_kws=kwargs,
                    bins=50,
                    label=f"{x_col} - {agg[0]}")


def p(n):
    """Gets nth percentile func"""
    def p_(x):
        return np.percentile(x, n)

    p_.__name__ = "p_%s" % n
    return p_


PLOTTERS = dict(getmembers(Plot, predicate=isfunction))
PROCESSORS = dict(getmembers(DataProcessor, predicate=isfunction))


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(description="plot some dataz")
    # Data processing
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--plt", choices=list(PLOTTERS.keys()), default="scatter")
    parser.add_argument(
        "--process", choices=list(PROCESSORS.keys()), default="identity")
    parser.add_argument(
        "--x",
        nargs="+",
        help="x column name; note: this may only be a list for CDF")
    parser.add_argument("--y", nargs="+", help="y column name")
    parser.add_argument("--z", help="z column name (used for legend)")
    parser.add_argument("--delim", default=",", help="input csv delimeter")
    # Decorate plot. Omitting some of these may trigger input request.
    parser.add_argument("--x-min", type=float, help="min on x-axis")
    parser.add_argument("--y-min", type=float, help="min on y-axis")
    parser.add_argument("--x-max", type=float, help="max on x-axis")
    parser.add_argument("--y-max", type=float, help="max on y-axis")
    parser.add_argument("--x-scale", help="x-scale type; eg log or linear")
    parser.add_argument("--y-scale", help="y-scale type; eg log or linear")
    parser.add_argument("--xlabel", help="x-axis label; default = column name")
    parser.add_argument("--ylabel", help="y-axis label; default = column name")
    parser.add_argument("--title", help="title of plot")
    return parser.parse_args()


def clean_and_validate(args):
    """Clean args, validate."""
    err = None
    if args.plt != "cdf":
        if len(args.x) > 1:
            err = f"--x-col must be len 1 for {args.plt}"
        else:
            # set as str since other plotters don't expect list
            args.x = args.x[0]
    elif args.plt == "cdf" and args.y:
        err = f"--y-col must not be set for {args.plt}"

    args.y = args.y[0] if args.y else None

    if err:
        raise ValueError(err)
    return args


if __name__ == "__main__":
    main(clean_and_validate(parse()))
