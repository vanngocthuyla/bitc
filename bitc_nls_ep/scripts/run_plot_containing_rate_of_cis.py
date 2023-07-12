
import os
import argparse
import pickle

import numpy as np
import pandas as pd

from _confidence_intervals import rate_of_containing_from_sample, rate_of_containing_from_means_stds
from _plot_confidence_intervals import plot_containing_rates
from _data_files import read_experimental_design_parameters

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",    type=str, default="")
parser.add_argument( "--bitc_mcmc_dir",                         type=str, default="bitc_mcmc")
parser.add_argument( "--nonlinear_fit_result_file",             type=str, default="origin_dg_dh_in_kcal_per_mole.dat")

parser.add_argument( "--ordered_experiment_names",              type=str, default=" ")
parser.add_argument( "--parameter",                             type=str, default="DeltaG")
parser.add_argument( "--central",                               type=str, default="median")
parser.add_argument( "--true_value",                            type=int, default=1)

args = parser.parse_args()
args.true_value = bool(args.true_value)

assert args.true_value in [0, 1], "1. Using true value or 0. No using true value"
assert args.central in ["mean", "median"], "wrong central"

print("ploting " + args.parameter)

TRACES_FILE = "traces.pickle"
XLABEL = "predicted"
YLABEL = "observed"

LEVELS_PERCENT = np.linspace(10., 95., num=18)
print("LEVELS_PERCENT", LEVELS_PERCENT)
LEVELS = LEVELS_PERCENT / 100.

ordered_experiment_names = args.ordered_experiment_names.split()

assert len(args.experimental_design_parameters_dir)>0, "Please provide the directory of experimental design parameters."

print("Loading the TRUE experimental design parameter file.")
parameters_true = read_experimental_design_parameters(args.experimental_design_parameters_dir+'/true_experimental_design_parameters.dat')

if args.true_value==True: 
    if args.parameter=="Ls":
        true_value = []
        for name in ordered_experiment_names:
            true_value.append(abs(parameters_true[name]['syringe_concentration']))
        true_value = np.array(true_value)
    if args.parameter=="P0":
        true_value = []
        for name in ordered_experiment_names:
            true_value.append(abs(parameters_true[name]['cell_concentration']))
        true_value = np.array(true_value)
    if args.parameter=="DeltaG":
        true_value=-10
    if args.parameter=="DeltaH":
        true_value=-5
    if args.parameter=="DeltaH_0":
        true_value=5e-7
    print("True value:", true_value)
else: 
    true_value=None
    print("No true values of parameters provided.")


# bayesian cis
mcmc_trace_files = [os.path.join(args.bitc_mcmc_dir, exper_name, TRACES_FILE) for exper_name in ordered_experiment_names]
if len(mcmc_trace_files)>0:

    samples = [ pickle.load( open(trace_file , "rb") )[args.parameter] for trace_file in mcmc_trace_files ]

    b_rates = []
    b_rate_errors = []
    for level in LEVELS:
        print(level)
        rate, rate_error = rate_of_containing_from_sample(samples, level, estimate_of_true=args.central, true_val=true_value, ci_type="bayesian", bootstrap_repeats=2)

        rate *= 100
        rate_error *= 100

        b_rates.append(rate)
        b_rate_errors.append(rate_error)

    # error bars to be one standard error
    b_rate_errors = [e/2. for e in b_rate_errors]

    # dump result
    pickle.dump({"LEVELS_PERCENT":LEVELS_PERCENT, "Params:": args.parameter, "b_rates": b_rates, "b_rate_errors":b_rate_errors},
                open(args.parameter + "_bayesian.pkl", "wb"))

    out = args.parameter + "_bayesian.pdf"
    plot_containing_rates([LEVELS_PERCENT], [b_rates], observed_rate_errors=[b_rate_errors],
                          xlabel=XLABEL+' '+args.parameter, ylabel=YLABEL+' '+args.parameter,
                          xlimits=[0, 100], ylimits=[0, 100])


# # nonlinear ls cis
# if os.path.exists(args.nonlinear_fit_result_file):

#     nonlinear_fit_results = pd.read_table(args.nonlinear_fit_result_file, sep='\s+')
#     nonlinear_fit_results = nonlinear_fit_results.reindex(ordered_experiment_names)

#     for exper in nonlinear_fit_results.index:
#         if np.any(nonlinear_fit_results.loc[exper].isnull()):
#             raise Exception(exper + " is null")

#     means = nonlinear_fit_results[ args.parameter ]
#     stds  = nonlinear_fit_results[ args.parameter + "_std" ]

#     g_rates = []
#     for level in LEVELS:
#         g_rate = rate_of_containing_from_means_stds(means, stds, level, estimate_of_true=args.central, true_val=args.true_value)

#         g_rate *= 100
#         g_rates.append(g_rate)

#     out = args.parameter + "_nonlinear_ls.pdf"
#     plot_containing_rates([LEVELS_PERCENT], [g_rates], out, observed_rate_errors=None,
#                           xlabel=XLABEL, ylabel=YLABEL, xlimits=[0, 100], ylimits=[0, 100])

#     # dump result
#     pickle.dump({"LEVELS_PERCENT":LEVELS_PERCENT, "g_rates":g_rates},
#                 open(args.parameter + "_nls_results.pkl", "wb"))

print("DONE")