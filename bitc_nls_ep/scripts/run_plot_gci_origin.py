
import os
import glob
import pickle
import argparse

import pandas as pd
import numpy as np

from _confidence_intervals import bayesian_credible_interval, gaussian_ci_from_mean_std
from _plot_confidence_intervals import plot_vertically_stacked_cis

parser = argparse.ArgumentParser()

parser.add_argument( "--nonlinear_fit_result_file",     type=str,   default="")

parser.add_argument( "--ordered_experiment_names_1",    type=str,   default=" " )
parser.add_argument( "--ordered_experiment_names_2",    type=str,   default=" " )

parser.add_argument( "--parameter",                     type=str,   default="DeltaG")
parser.add_argument( "--level",                         type=float, default=0.95)
parser.add_argument( "--central",                       type=str,   default="median")
parser.add_argument( "--xlimits",                       type=str,   default=None )

args = parser.parse_args()

assert args.central in ["mean", "median"], "wrong central"

print("ploting " + args.parameter)

TRACES_FILE = "traces.pickle"
XLABELS = {"DeltaG":"$\Delta G$ (kcal/mol)", "DeltaH":"$\Delta H$ (kcal/mol)",
           "DeltaH_0":"$\Delta H_0$ (cal)", "P0":"$[R]_0$ (mM)", "Ls":"$[L]_s$ (mM)"}

ordered_experiment_names_1 = args.ordered_experiment_names_1.split()
ordered_experiment_names_2 = args.ordered_experiment_names_2.split()

ordered_experiment_names = ordered_experiment_names_1 + ordered_experiment_names_2
print(ordered_experiment_names)

use_two_centrals_for_P0 = False
use_two_centrals_for_Ls = True
if len(ordered_experiment_names_2) > 0:
    use_two_centrals_for_P0 = True
    use_two_centrals_for_Ls = True
    print("use two centrals for P0 and Ls")
else:
    print("use one central for P0 and Ls")

if args.xlimits is None:
    xlimits = None
else:
    xlimits = [ float(s) for s in args.xlimits.split() ]
print("xlimits = ", xlimits)

# nonlinear ls cis

nonlinear_fit_results = pd.read_table(args.nonlinear_fit_result_file, sep='\s+')
nonlinear_fit_results = nonlinear_fit_results.reindex(ordered_experiment_names)

for exper in nonlinear_fit_results.index:
    if np.any(nonlinear_fit_results.loc[exper].isnull()):
        raise Exception(exper + " is null")

means = nonlinear_fit_results[ args.parameter ]
stds  = nonlinear_fit_results[ args.parameter + "_std" ]

g_lowers  = []
g_uppers  = []

for mu, sigma in zip(means, stds):
    l, u = gaussian_ci_from_mean_std(mu, sigma, args.level)
    g_lowers.append(l)
    g_uppers.append(u)

if args.central == "median":
    g_centrals = [np.median(means)]
elif args.central == "mean":
    g_centrals = [np.mean(means)]

if use_two_centrals_for_P0 and args.parameter == "P0":
    if args.central == "median":
        g_centrals =  [ np.median( [ means[exper] for exper in ordered_experiment_names_1 ] ) ]
        g_centrals += [ np.median( [ means[exper] for exper in ordered_experiment_names_2 ] ) ]
        
    elif args.central == "mean":
        g_centrals =  [ np.mean( [ means[exper] for exper in ordered_experiment_names_1 ] ) ]
        g_centrals += [ np.mean( [ means[exper] for exper in ordered_experiment_names_2 ] ) ]

out = args.parameter + "_nonlinear_ls_cis.pdf"

xlabel = XLABELS[args.parameter]

plot_vertically_stacked_cis(g_lowers, g_uppers, xlabel, out,
                            centrals=g_centrals, xlimits=xlimits)

