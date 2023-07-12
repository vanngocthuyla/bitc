import os
import argparse
import pickle

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
sns.set_style("white")


def _plot_1d_kde(x, out, xlabel=None, ylabel=None, den_color="r", horizontal_line=None,
                 lw=1.5, marker_at=None, line_color="k", 
                 marker_color="k", markersize=8, marker="v", nticks=6):
    """
    """
    figure_size = (3.2, 2.4)
    dpi = 300
    fontsize = 8
    font = { "fontname": "Arial"}

    plt.figure(figsize=figure_size)
    ax = sns.kdeplot(x, shade=True, color=den_color)

    if horizontal_line is not None:
        x1, x2, y = horizontal_line
        ax.plot([x1, x2], [y, y], marker="|", markersize=markersize, color=line_color, lw=lw)

    if marker_at is not None:
        x1, y1 = marker_at
        ax.plot([x1], [y1], marker=marker, markersize=markersize, color=marker_color)

    ax.locator_params(axis="x", nbins=nticks)
    ax.locator_params(axis="y", nbins=nticks)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize, **font)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize, **font)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None


def _plot_2d_kde(x, y, out, xlabel=None, ylabel=None, zlabel=None, nticks=10):
    """
    """
    sns.set(font_scale=0.75)
    figure_size = (3.2, 2.4)
    dpi = 300
    fontsize = 8
    font = { "fontname": "Arial"}

    # for more colormaps see http://matplotlib.org/examples/color/colormaps_reference.html
    my_cmap = plt.get_cmap('hot')

    plt.figure(figsize=figure_size)
    ax = sns.kdeplot(x, y, shade=True, cbar=True, cmap=my_cmap, cbar_kws={"label":zlabel} )

    ax.locator_params(axis="x", nbins=nticks)
    ax.locator_params(axis="y", nbins=nticks)

    #for tick in ax.xaxis.get_major_ticks():
        #tick.label.set_fontsize(fontsize)

    #for tick in ax.yaxis.get_major_ticks():
        #tick.label.set_fontsize(fontsize)

    if xlabel is not None:
        #plt.xlabel(xlabel, fontsize=fontsize, **font)
        plt.xlabel(xlabel)

    if ylabel is not None:
        #plt.ylabel(ylabel, fontsize=fontsize, **font)
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None