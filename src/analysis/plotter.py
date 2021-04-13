import logging
from inspect import getmembers, isfunction

import numpy as np
import seaborn as sns
from pandas import DataFrame

logger = logging.getLogger(__name__)


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
            logger.info("Stats: %s", percentiles(25, 50, 75, 90, 95, 99))
            a = sns.distplot(
                agg[1][args.x],
                hist=False,
                hist_kws={"cumulative": args.plt == "cdf"},
                kde_kws={"cumulative": args.plt == "cdf"},
                bins=50,
                label=f"{args.x} - {agg[0]}")
            sns.set()
        return a

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
