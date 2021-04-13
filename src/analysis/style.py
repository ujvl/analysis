import logging
from inspect import getmembers, ismethod

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def style(args):
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
        logger.info(f"Applying style {args.style}")
        STYLES[args.style]()
    elif callable(args.style):
        args.style()
    else:
        logger.warning("No styling applied.")


class Style:
    """Style plot."""

    PAPER_RC = {
        "font.size": 30,
        "axes.titlesize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 19,
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.25,
    }

    @classmethod
    def none(cls):
        pass

    @classmethod
    def default(cls):
        """Default seaborn styling."""
        sns.set_theme(style="darkgrid")

    @classmethod
    def paper(cls):
        """Publication styling (big text)."""
        sns.set(style="white", font_scale=2, rc=cls.PAPER_RC)
        plt.tight_layout()


class StyleUtil:

    @staticmethod
    def legend_loc(grid, x, y):
        """Set location of legend."""
        grid._legend.set_bbox_to_anchor([x, y])

    @staticmethod
    def enable_borders(grid):
        # Enable borders.
        if hasattr(grid.axes, "flatten"):
            for ax in grid.axes.flatten():
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1)
        else:
            logger.info("Borders already enabled?")

    @staticmethod
    def invert_axis(grid):
        """Invert axis."""
        # grid.fig.axes[0].invert_xaxis()
        # grid.invert_xaxis()


STYLES = dict(getmembers(Style, predicate=ismethod))
