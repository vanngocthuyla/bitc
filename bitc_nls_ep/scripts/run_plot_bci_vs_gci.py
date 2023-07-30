
import os
import glob
import pickle
import argparse

import pandas as pd
import numpy as np

from _confidence_intervals import bayesian_credible_interval, gaussian_ci_from_mean_std
from _plot_confidence_intervals import plot_vertically_stacked_cis

parser = argparse.ArgumentParser()

parser.add_argument( "--bitc_mcmc_dir",                     type=str,   default="")
parser.add_argument( "--nonlinear_fit_result_file",         type=str,   default="")
parser.add_argument( "--propagation_error_result_file",     type=str,   default="")

parser.add_argument( "--ordered_experiment_names_1",        type=str,   default="" )
parser.add_argument( "--ordered_experiment_names_2",        type=str,   default="" )

parser.add_argument( "--use_two_centrals_for_P0",action="store_true",   default=False)
parser.add_argument( "--use_two_centrals_for_Ls",action="store_true",   default=True)

parser.add_argument( "--parameter",                         type=str,   default="DeltaG")
parser.add_argument( "--level",                             type=float, default=0.95)
parser.add_argument( "--central",                           type=str,   default="median")
parser.add_argument( "--xlimits",                           type=str,   default=None )

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

use_two_centrals_for_P0 = args.use_two_centrals_for_P0
use_two_centrals_for_Ls = args.use_two_centrals_for_Ls
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

xlabel = XLABELS[args.parameter]

# bayesian cis
if len(args.bitc_mcmc_dir)>0:
    mcmc_trace_files = {exper : os.path.join(args.bitc_mcmc_dir, exper, TRACES_FILE) for exper in ordered_experiment_names}
    samples = { exper : pickle.load( open(mcmc_trace_files[exper] , "rb") )[args.parameter] for exper in ordered_experiment_names }
    if args.parameter == 'DeltaH_0':
        samples = samples*1E6

    lowers          = []
    uppers          = []
    lower_errors    = []
    upper_errors    = []

    for exper in ordered_experiment_names:
        lower, upper, lower_error, upper_error = bayesian_credible_interval(samples[exper], args.level, bootstrap_repeats=1000) 

        lowers.append(lower)
        uppers.append(upper)
        lower_errors.append(lower_error)
        upper_errors.append(upper_error)

    if args.central == "median":
        b_centrals = [ np.median( [np.median(sample) for sample in samples.values()] ) ]

    elif args.central == "mean":
        b_centrals = [ np.mean( [ np.mean(sample) for sample in samples.values() ] ) ]

    if use_two_centrals_for_P0 and args.parameter == "P0":
        if args.central == "median":
            b_centrals =  [ np.median( [ np.median(samples[exper]) for exper in ordered_experiment_names_1 ] ) ]
            b_centrals += [ np.median( [ np.median(samples[exper]) for exper in ordered_experiment_names_2 ] ) ]

        elif args.central == "mean":
            b_centrals =  [ np.mean( [ np.mean(samples[exper]) for exper in ordered_experiment_names_1 ] ) ]
            b_centrals += [ np.mean( [ np.mean(samples[exper]) for exper in ordered_experiment_names_2 ] ) ]

    out = args.parameter + "_bayesian_cis.pdf"

    # error bars to be one standard error
    lower_errors = [l/2. for l in lower_errors]
    upper_errors = [u/2. for u in upper_errors]

    plot_vertically_stacked_cis(lowers, uppers, xlabel, out,
                                lower_errors=lower_errors, upper_errors=upper_errors,
                                centrals=b_centrals, xlimits=xlimits)

if len(args.nonlinear_fit_result_file)>0:

    MLE_results = pd.read_csv(args.nonlinear_fit_result_file, index_col=0)
    if 'DeltaH_0' in MLE_results.columns:
        MLE_results['DeltaH_0'] = MLE_results['DeltaH_0']*1E6
    MLE_results = MLE_results.reindex(ordered_experiment_names)

    means = MLE_results[args.parameter]
    stds  = MLE_results[args.parameter+"_std"]

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

    if (use_two_centrals_for_P0 and args.parameter == "P0") or (use_two_centrals_for_Ls and args.parameter == "Ls"):
        if args.central == "median":
            g_centrals =  [ np.median( [ means[exper] for exper in ordered_experiment_names_1 ] ) ]
            g_centrals += [ np.median( [ means[exper] for exper in ordered_experiment_names_2 ] ) ]
            
        elif args.central == "mean":
            g_centrals =  [ np.mean( [ means[exper] for exper in ordered_experiment_names_1 ] ) ]
            g_centrals += [ np.mean( [ means[exper] for exper in ordered_experiment_names_2 ] ) ]

    out = args.parameter + "_nls_cis.pdf"

    plot_vertically_stacked_cis(g_lowers, g_uppers, xlabel, out,
                                centrals=g_centrals, xlimits=xlimits)

# error-propagation nls cis
if len(args.propagation_error_result_file)>0: 
    
    Propagation_results = pd.read_csv(args.propagation_error_result_file, index_col=0)
    if 'DeltaH_0' in Propagation_results.columns:
        Propagation_results['DeltaH_0'] = Propagation_results['DeltaH_0']*1E6
    Propagation_results = Propagation_results.reindex(ordered_experiment_names)
    
    means = Propagation_results[args.parameter]
    stds  = Propagation_results[args.parameter+"_std"]

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

    if (use_two_centrals_for_P0 and args.parameter == "P0") or (use_two_centrals_for_Ls and args.parameter == "Ls"):
        if args.central == "median":
            g_centrals =  [ np.median( [ means[exper] for exper in ordered_experiment_names_1 ] ) ]
            g_centrals += [ np.median( [ means[exper] for exper in ordered_experiment_names_2 ] ) ]
            
        elif args.central == "mean":
            g_centrals =  [ np.mean( [ means[exper] for exper in ordered_experiment_names_1 ] ) ]
            g_centrals += [ np.mean( [ means[exper] for exper in ordered_experiment_names_2 ] ) ]

    out = args.parameter + "_nls_ep_cis.pdf"

    plot_vertically_stacked_cis(g_lowers, g_uppers, xlabel, out=out,
                                centrals=g_centrals, xlimits=xlimits)