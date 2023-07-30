
"""
contains functions that compute Gaussian CIs and Bayesian credible intervals
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import arviz as az


def plot_vertically_stacked_cis(lowers, uppers, xlabel, out,
                                lower_errors=None, upper_errors=None, centrals=None,
                                xlimits=None, nticks=6, main_lw=2.0, error_lw=4.0, central_lw=1.5,
                                main_color="k", error_color="r", central_color="g", fontsize=8, 
                                figure_size=(3.2, 2.4), dpi=300, font = {"fontname": "Arial"}):
    """
    lowers  :   array-like, float
    uppers  :   array-like, float
    xlabel  :   str
    out     :   str
    lower_stds  :   None or array-like of floats
    upper_stds  :   None or array-like of floats
    centrals    :   None or list of floats
    xlimits     :   None or [float, float]
    """
    assert len(lowers) == len(uppers), "lowers and uppers must have the same len"
    if (lower_errors is not None) and (upper_errors is not None):
        assert len(lower_errors) == len(upper_errors) == len(lowers), "lowers, lower_errors and upper_errors must have the same len"

    plt.figure(figsize=figure_size)
    ax = plt.axes()

    xs = list(zip(lowers, uppers))
    ys = [ [i, i] for i in range(len(lowers)) ]
    
    for i in range(len(xs)):
        ax.plot(xs[i], ys[i], linestyle="-", color=main_color, linewidth=main_lw)

    if (lower_errors is not None) and (upper_errors is not None):
        l_err_bars = [ [val - err, val + err] for val, err in zip(lowers, lower_errors) ]
        u_err_bars = [ [val - err, val + err] for val, err in zip(uppers, upper_errors) ]

        for i in range( len(l_err_bars) ):
            ax.plot(l_err_bars[i], ys[i], linestyle="-", color=error_color, linewidth=error_lw)
            ax.plot(u_err_bars[i], ys[i], linestyle="-", color=error_color, linewidth=error_lw)

    if centrals is not None:
        y_min = np.min(ys)
        y_max = np.max(ys)

        for central in centrals:
            ax.plot([central, central], [y_min, y_max], linestyle="-", color=central_color, linewidth=central_lw)

    ax.locator_params(axis='x', nbins=nticks)

    lower_y = np.min(ys) - 1
    upper_y = np.max(ys) + 1
    ax.set_ylim([lower_y, upper_y])

    if xlimits is not None:
        ax.set_xlim([xlimits[0], xlimits[1]])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fontsize, **font)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    return None


def plot_containing_rates(predicted_rates, observed_rates, label=None, out=None, 
                          observed_rate_errors=None, show_diagonal_line=True,
                          xlabel="predicted", ylabel="observed",
                          xlimits=[0, 100], ylimits=[0, 100], nticks=6, 
                          colors=None, markers=None, markersize=5,
                          diagonal_line_style="-", diagonal_line_w=1., diagonal_line_c="k",
                          aspect="equal", figure_size=(3.2, 3.2),
                          dpi=300, fontsize=8, font = {"fontname": "Arial"}, ax=None):
    """
    """
    assert isinstance(predicted_rates, list), "predicted_rates must be a list"
    assert isinstance(observed_rates, list), "predicted_rates must be a list"

    assert len(predicted_rates) == len(observed_rates), "predicted_rates and observed_rates do not have the same len"

    if observed_rate_errors is not None:
        assert len(observed_rates) == len(observed_rate_errors), "observed_rates and observed_rate_errors do not have the same len"
    else:
        observed_rate_errors = [[None for _ in line] for line in predicted_rates]

    if colors is None:
        colors = ["k" for _ in range(len(predicted_rates))]

    if markers is None:
        markers = ["o" for _ in range(len(predicted_rates))]

    if ax == None:
        plt.figure(figsize=figure_size)
        ax = plt.axes()

    for i in range(len(predicted_rates)):
        for j in range(len(predicted_rates[i])):
            ax.errorbar(predicted_rates[i][j], observed_rates[i][j], yerr=observed_rate_errors[i][j], marker=markers[i],
                        markersize=markersize, color=colors[i], label=label, linestyle="None")

    if show_diagonal_line:
        ax.plot( xlimits, ylimits, linestyle=diagonal_line_style, linewidth=diagonal_line_w, color=diagonal_line_c )

    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    ax.set_aspect(aspect=aspect)

    ax.locator_params(axis='x', nbins=nticks)
    ax.locator_params(axis='y', nbins=nticks)

    try:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        
        ax.set_xlabel(xlabel, fontsize=fontsize, **font)
        ax.set_ylabel(ylabel, fontsize=fontsize, **font)
    except: 
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

    plt.tight_layout()
    if not out==None:
        plt.savefig(out, dpi=dpi)
    
    return None


def plot_ci_convergence(lowers, uppers, list_of_stops, xlabel, ylabel, out, xlimits=None, ylimits=None, 
                        repeats_linestyle="-", mean_linestyle="-", repeats_lw=0.8, mean_lw=1.0, 
                        repeats_colors=None, repeats_alpha=1.0, mean_color=None, mean_alpha=1.0,
                        x_nticks=6, y_nticks=10, figure_size=(3.2, 2.4),
                        dpi=300, fontsize=8, font = {"fontname": "Arial"}):
    """
    lowers  :   ndarray of shape = ( n_repeated_samples, n_stops )
    uppers  :   ndarray of shape = ( n_repeated_samples, n_stops )
    """
    assert lowers.shape == uppers.shape, "lowers and uppers must have the same shape"
    assert lowers.ndim == 2, "lowers must be 2d array"
    assert lowers.shape[-1] == len(list_of_stops), "lowers.shape[-1] must be the same as len(list_of_stops)"

    list_of_stops = np.asarray(list_of_stops)

    plt.figure(figsize=figure_size)
    ax = plt.axes()

    nrepeats = lowers.shape[0]

    if repeats_colors is None:
        repeats_colors = ["k" for _ in range(nrepeats)]

    if mean_color is None:
        mean_color = "r"

    for repeat in range(nrepeats):
        plt.plot(list_of_stops, lowers[repeat], linestyle=repeats_linestyle, color=repeats_colors[repeat], linewidth=repeats_lw, alpha=repeats_alpha)
        plt.plot(list_of_stops, uppers[repeat], linestyle=repeats_linestyle, color=repeats_colors[repeat], linewidth=repeats_lw, alpha=repeats_alpha)

    lower_mean  = lowers.mean(axis=0)
    lower_error = lowers.std(axis=0)

    upper_mean  = uppers.mean(axis=0)
    upper_error = uppers.std(axis=0)

    # error bars to be one standard error
    lower_error /= 2.
    upper_error /= 2.

    plt.errorbar(list_of_stops, lower_mean, yerr=lower_error, linestyle=mean_linestyle, color=mean_color, linewidth=mean_lw, alpha=mean_alpha)
    plt.errorbar(list_of_stops, upper_mean, yerr=upper_error, linestyle=mean_linestyle, color=mean_color, linewidth=mean_lw, alpha=mean_alpha)

    if xlimits is not None:
        ax.set_xlim(xlimits)

    if ylimits is not None:
        ax.set_ylim(ylimits)

    ax.locator_params(axis='x', nbins=x_nticks)
    ax.locator_params(axis='y', nbins=y_nticks)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    plt.xlabel(xlabel, fontsize=fontsize, **font)
    plt.ylabel(ylabel, fontsize=fontsize, **font)

    plt.tight_layout()
    plt.savefig(out, dpi=dpi)

    return None