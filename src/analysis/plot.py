#!/usr/bin/env python
from __future__ import print_function
import argparse
from inspect import getmembers, isfunction, ismethod
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
    """main"""
    logger.debug(f"Args: {args}")
    processor = PROCESSORS[args.process]
    data = processor(pd.read_csv(args.csv, sep=args.delim), args)
    if args.filter:
        data = filter_data(data, args)
    plot(data, args)


class Args:
    def __init__(self, adict: dict = None):
        if adict:
            self.__dict__.update(adict)

    def __getattr__(self, name):
        return None


def plot(data, plot_args=None, **plot_kwargs):
    """Plot data.

    Args:
        plot_args: This is just the Namespace or Args object.
        plot_kwargs: Instead of passing in plot_args, the individual
            arguments can be passed in by keyword (see `parse`).
    """
    args = plot_args or Args(plot_kwargs)
    if plot_args and plot_kwargs:
        raise ValueError("Plot args/kwargs are mutually exclusive.")

    facet_grid = PLOTTERS[args.plt](data, args)
    style(args, facet_grid)
    logger.info("Showing plot...")
    if args.out:
        plt.savefig(args.out)
    plt.show()


def style(args, facet_grid):
    """Requests user input to add plot labels, title etc."""
    if args.style != "paper":
        title = args.title or input("Title: ")
        plt.title(title)

    x_label = args.xlabel or input("x-axis label: ") or args.x
    y_label = args.ylabel or input("y-axis label: ") or args.y
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

    if isinstance(args.style, str):
        STYLES[args.style](facet_grid)
    elif callable(args.style):
        args.style(facet_grid)
    else:
        logger.warning("No styling applied.")


def filter_data(data: DataFrame, args) -> DataFrame:
    """Filter data via query string."""
    return data.query(args.filter)


class DataProcessor:
    """Data processing namespace."""

    @staticmethod
    def identity(data: DataFrame, args) -> DataFrame:
        """Identity."""
        return data

    @staticmethod
    def join_z(data: DataFrame, args) -> DataFrame:
        """Join --z columns."""
        cols = args.z.split(",")
        concat_col = " * ".join(cols)
        args.z = concat_col

        data[concat_col] = data[cols].apply(
            lambda row: " * ".join(row.values.astype(str)), axis=1
        )
        return data

    @staticmethod
    def div(data: DataFrame, args) -> DataFrame:
        """Divide."""
        grouped = data.groupby([args.x, args.z])
        agg = grouped.agg(
            {args.y: lambda x: x.iloc[1] / x.iloc[0] if len(x) == 2 else None}
        )[args.y].reset_index()
        return agg


class Style:
    """Style plot."""

    PAPER_RC = {
        "font.size": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 19,
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.25,
    }

    @classmethod
    def none(cls, facet_grid):
        pass

    @classmethod
    def default(cls, facet_grid):
        """Default seaborn styling."""
        sns.set()

    @classmethod
    def paper(cls, facet_grid):
        """Publication styling (big text)."""
        logger.info("Setting paper style.")
        # Enable borders.
        if hasattr(facet_grid.axes, "flatten"):
            for ax in facet_grid.axes.flatten():
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1)
        else:
            logger.info("Borders already enabled?")
        sns.set(style="white", rc=cls.PAPER_RC)
        plt.tight_layout()


class StyleUtil:

    @staticmethod
    def legend_loc(grid, x, y):
        """Set location of legend."""
        grid._legend.set_bbox_to_anchor([x, y])

    @staticmethod
    def invert_axis(grid):
        pass
        # grid.fig.axes[0].invert_xaxis()
        # grid.invert_xaxis()


class Plot:
    """Data plotting namespace."""

    @staticmethod
    def line(data, args):
        """Line plot"""
        return Plot.scatter(data, args)

    @staticmethod
    def scatter(data, args):
        """Scatter plot"""
        args.x, args.y, args.z = Plot._cols(
            data.columns, args.x, args.y, args.z)
        # Workaround since hue uses duck-typing for numerics.
        if args.z and data.dtypes[args.z] == int:
            data[args.z] = data[args.z].astype(str)
            data[args.z] = "$" + data[args.z] + "$"

        return sns.relplot(
            x=args.x,
            y=args.y,
            hue=args.z,
            kind=args.plt,
            data=data,
        )

    @staticmethod
    def reg(data, args):
        """Scatter plot with fitted function"""
        args.x, args.y, args.z = Plot._cols(
            data.columns, args.x, args.y, args.z)
        return sns.lmplot(x=args.x, y=args.y, hue=args.z, data=data)

    @staticmethod
    def bar(data, args):
        """Bar plot"""
        args.x, args.y = Plot._cols(data.columns, args.x, args.y)
        return sns.barplot(x=args.x, y=args.y, hue=args.z, data=data)

    @staticmethod
    def box(data, args):
        """Box plot"""
        args.x, args.y = Plot._cols(data.columns, args.x, args.y)
        # data = data[[args.x, args.y]]
        # aggs = data.groupby(args.x).agg(
        #     ["mean", "std", p(25), p(50), p(75), p(90), p(95), p(99)])
        return sns.boxplot(
            x=args.x,
            y=args.y,
            hue=args.z,
            data=data,
            palette="Blues",
            width=0.35,
        )

    @staticmethod
    def cdf(data, args):
        """CDF plot. Dist plot type is implicit (args.plt)."""
        return Plot._dist(data, args)

    @staticmethod
    def pdf(data, args):
        """PDF plot. Dist plot type is implicit (args.plt)."""
        return Plot._dist(data, args)

    @staticmethod
    def _dist(data, args):
        """Distribution plot"""
        args.x, args.z = Plot._cols(data.columns, args.x, args.z)
        # Workaround since hue uses duck-typing for numerics.
        if args.z and data.dtypes[args.z] == int:
            data[args.z] = data[args.z].astype(str)
            data[args.z] = "$" + data[args.z] + "$"

        if args.z:
            aggs = data.groupby(args.z)
        else:
            aggs = [(args.x, {args.x: data[args.x]})]

        for agg in aggs:
            logger.info(
                "Stats: %s",
                ",".join(percentiles(25, 50, 75, 90, 95, 99)))
            sns.distplot(
                agg[1][args.x],
                hist=False,
                hist_kws={"cumulative": args.plt == "cdf"},
                kde_kws={"cumulative": args.plt == "cdf"},
                bins=50,
                label=f"{args.x} - {agg[0]}")

    @staticmethod
    def _cols(all_cols: list, *cols):
        if all(col is None for col in cols):
            default_cols = tuple(all_cols[:len(cols)])
            pad = len(cols) - len(default_cols)
            default_cols += tuple(None for _ in range(pad))
            logger.warning("Using {default_cols} by default.")
            return default_cols
        return tuple(cols)


def percentiles(x, *n):
    return [np.percentile(x, _n) for _n in n]


def p(n):
    """Gets nth percentile func"""
    def p_(x):
        return np.percentile(x, n)

    p_.__name__ = "p_%s" % n
    return p_


PLOTTERS = dict(getmembers(Plot, predicate=isfunction))
PROCESSORS = dict(getmembers(DataProcessor, predicate=isfunction))
STYLES = dict(getmembers(Style, predicate=ismethod))


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(description="plot some dataz")
    # Data processing
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--plt", choices=list(PLOTTERS.keys()), default="scatter")
    parser.add_argument(
        "--process", choices=list(PROCESSORS.keys()), default="identity")
    parser.add_argument("--filter", help="filter query")
    parser.add_argument("--x", help="x column name")
    parser.add_argument("--y", nargs="+", help="y column name")
    parser.add_argument("--z", help="z column name (used for legend)")
    parser.add_argument("--delim", default=",", help="input csv delimeter")
    # Decorate plot. Omitting some of these may trigger input request.
    parser.add_argument(
        "--style",
        choices=list(STYLES.keys()),
        default="paper",
        help="style mode")
    parser.add_argument("--x-min", type=float, help="min on x-axis")
    parser.add_argument("--y-min", type=float, help="min on y-axis")
    parser.add_argument("--x-max", type=float, help="max on x-axis")
    parser.add_argument("--y-max", type=float, help="max on y-axis")
    parser.add_argument("--x-scale", help="x-scale type; eg log or linear")
    parser.add_argument("--y-scale", help="y-scale type; eg log or linear")
    parser.add_argument("--xlabel", help="x-axis label; default = column name")
    parser.add_argument("--ylabel", help="y-axis label; default = column name")
    parser.add_argument("--title", help="title of plot")
    parser.add_argument("--out", help="save to out file")
    return parser.parse_args()


def clean_and_validate(args):
    """Clean args, validate."""
    err = None
    if args.plt == "cdf" and args.y:
        err = f"--y-col must not be set for {args.plt}"

    args.y = args.y[0] if args.y else None

    if err:
        raise ValueError(err)
    return args


if __name__ == "__main__":
    main(clean_and_validate(parse()))
