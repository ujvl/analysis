#!/usr/bin/env python
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
from pandasgui import show

from analysis.plotter import PLOTTERS, PROCESSORS
from analysis.style import style, STYLES

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """main"""
    logger.debug(f"Args: {args}")
    processor = PROCESSORS[args.process]
    data = processor(pd.read_csv(args.csv, sep=args.delim), args)
    if args.filter:
        data = data.query(args.filter)
    if args.interactive:
        show(data)
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

    style(args)
    PLOTTERS[args.plt](data, args)
    logger.info("Showing plot...")
    if args.out:
        plt.savefig(args.out)
    plt.show()


def pandas_gui(data):
    show(data)


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(description="plot some dataz")
    parser.add_argument("-i", "--interactive", action="store_true", help="pandasgui")
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
        default="default",
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
