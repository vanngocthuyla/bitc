"""
combine three curves of Bayesian model, MLE and MLE with Error Propagation into one plot
"""
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import argparse
import pickle

import matplotlib
import matplotlib.pyplot as plt

from _confidence_intervals import rate_of_containing_from_sample, rate_of_containing_from_means_stds
from _plot_confidence_intervals import plot_containing_rates
from _data_files import read_experimental_design_parameters

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",    type=str, default="")
parser.add_argument( "--bitc_mcmc_dir",                         type=str, default="bitc_mcmc")
parser.add_argument( "--nonlinear_fit_result_file",             type=str, default="MLE_parameters.csv")
parser.add_argument( "--propagation_error_result_file",         type=str, default="Propagation_parameters.csv")

parser.add_argument( "--ordered_experiment_names_1",            type=str, default="" )
parser.add_argument( "--ordered_experiment_names_2",            type=str, default="" )

parser.add_argument( "--parameter",                             type=str, default="DeltaG DeltaH")
parser.add_argument( "--central",                               type=str, default="median")
parser.add_argument( "--true_value",                 action="store_true", default=True)
parser.add_argument( "--use_two_centrals_for_P0_Ls", action="store_true", default=False)

args = parser.parse_args()

assert args.central in ["mean", "median"], "wrong central"

MARKERS = ("<", ">", "o", "s")
COLORS = ("r", "b", "g", "k")
params_name = args.parameter.split() #["P0", "Ls", "DeltaG", "DeltaH", "DeltaH_0"]
params_name_greek = {"DeltaG"    :   "$\Delta G$ (kcal/mol)", 
                     "DeltaH"    :   "$\Delta H$ (kcal/mol)", 
                     "DeltaH_0"  :   "$\Delta H_0$ (cal)",
                     "P0"        :   "$[R]_0$ (mM)", 
                     "Ls"        :   "$[L]_s$ (mM)", 
                     "log_sigma" :   "$\ln \sigma$" }

ordered_experiment_names_1 = args.ordered_experiment_names_1.split()
ordered_experiment_names_2 = args.ordered_experiment_names_2.split()
ordered_experiment_names = ordered_experiment_names_1 + ordered_experiment_names_2

print("Working with", args.nonlinear_fit_result_file, "and", args.propagation_error_result_file)

use_two_centrals_for_P0_Ls = args.use_two_centrals_for_P0_Ls
if len(ordered_experiment_names_2) > 0:
    use_two_centrals_for_P0_Ls = True
    print("use two true values for P0 and Ls")
else:
    print("use one true value for P0 and Ls")

TRACES_FILE = "traces.pickle"
XLABEL = "predicted"
YLABEL = "observed"

LEVELS_PERCENT = np.linspace(10., 95., num=18)
print("LEVELS_PERCENT", LEVELS_PERCENT)
LEVELS = LEVELS_PERCENT / 100.

if args.true_value==True:
    
    true_value_list = {}
    n1 = len(args.ordered_experiment_names_1.split())
    n2 = len(args.ordered_experiment_names_2.split())

    mcmc_trace_files = [os.path.join(args.bitc_mcmc_dir, exper_name, TRACES_FILE) for exper_name in ordered_experiment_names]
    for j in params_name:
        samples = [ pickle.load( open(trace_file , "rb") )[j] for trace_file in mcmc_trace_files ]
        if (j != 'P0') and (j != 'Ls') or not(use_two_centrals_for_P0_Ls):
            if args.central == "median":
                true_value_list[j] = np.median(samples)
            elif args.central == "mean":  
                true_value_list[j] = np.mean(samples)
        else: 
            if args.central == "median":
                true_value_list[j] = np.concatenate((np.repeat(np.median(samples[0:n1]), n1),
                                                    np.repeat(np.median(samples[n1:]), n2)), 
                                                    axis=0)
            elif args.central == "mean":  
                true_value_list[j] = np.concatenate((np.repeat(np.mean(samples[0:n1]), n1),
                                                    np.repeat(np.mean(samples[n1:]), n2)), 
                                                    axis=0)
else: 
    true_value_list = {'P0': None, 'Ls': None, 'DeltaG': None, 'DeltaH': None, 'DeltaH_0': None}

# bayesian cis
try: 
    b_file = pickle.load(open(os.path.join("Bayesian.pkl"), "rb") )
    if b_file['LEVELS_PERCENT'].all() == LEVELS_PERCENT.all(): 
        b_rates_list = b_file['b_rates']
        b_rate_errors_list = b_file['b_rate_errors']
    else: 
        b_rates_list = {}
        b_rate_errors_list = {}
except:
    b_rates_list = {}
    b_rate_errors_list = {}

if len(b_rates_list) == 0: 
    mcmc_trace_files = [os.path.join(args.bitc_mcmc_dir, exper_name, TRACES_FILE) for exper_name in ordered_experiment_names]
    for j in params_name:
        print(j)
        try: 
            b_file = pickle.load(open(os.path.join(j+"_bayesian.pkl"), "rb") )
            if b_file['LEVELS_PERCENT'].all() == LEVELS_PERCENT.all(): 
                b_rates = b_file['b_rates']
                b_rate_errors = b_file['b_rate_errors']
        except: 
            samples = [ pickle.load( open(trace_file , "rb") )[j] for trace_file in mcmc_trace_files ]
            b_rates = []
            b_rate_errors = []
            for level in LEVELS:
                print(level)
                rate, rate_error = rate_of_containing_from_sample(samples, level, estimate_of_true=args.central,
                                                                  true_val = true_value_list[j],
                                                                  ci_type="bayesian", bootstrap_repeats=1000)

                rate *= 100
                rate_error *= 100

                b_rates.append(rate)
                b_rate_errors.append(rate_error)
            # error bars to be one standard error
            b_rate_errors = [e/2. for e in b_rate_errors]

            # out = j + "_bayesian.pdf"
            # plot_containing_rates([LEVELS_PERCENT], [b_rates], out, observed_rate_errors=[b_rate_errors],
            #                         xlabel=XLABEL, ylabel=YLABEL, xlimits=[0, 100], ylimits=[0, 100])

            #dump result
            pickle.dump({"LEVELS_PERCENT": LEVELS_PERCENT, "b_rates": b_rates, "b_rate_errors": b_rate_errors},
                        open(j + "_bayesian.pkl", "wb"))

        b_rates_list[j] = b_rates
        b_rate_errors_list[j] = b_rate_errors

    pickle.dump({"LEVELS_PERCENT": LEVELS_PERCENT, "b_rates": b_rates_list, "b_rate_errors": b_rate_errors_list},
                open("Bayesian.pkl", "wb"))

# nonlinear ls cis
if len(args.nonlinear_fit_result_file)>0:
    MLE_results = pd.read_csv(args.nonlinear_fit_result_file, index_col=0)
    MLE_results = MLE_results.reindex(ordered_experiment_names)

    for exper in MLE_results.index:
        if np.any(MLE_results.loc[exper].isnull()):
            raise Exception(exper + " is null")

    if args.true_value==True:
        true_value_list = {}    
        for j in params_name:
            try:
                means = MLE_results[j]
                if (j != 'P0') and (j != 'Ls') or not(use_two_centrals_for_P0_Ls):  
                    true_value_list[j] = np.mean(means)
                else: 
                    true_value_list[j] = np.concatenate((np.repeat(np.mean(means[0:n1]), n1),
                                                        np.repeat(np.mean(means[n1:]), n2)), 
                                                        axis=0)
            except:
                continue
    else: 
        true_value_list = {'P0': None, 'Ls': None, 'DeltaG': None, 'DeltaH': None, 'DeltaH_0': None}

    g_rate_list = {}
    for j in params_name:
        try:
            means = MLE_results[j]
            stds  = MLE_results[j+"_std"]
            g_rates = []
            for level in LEVELS:
                g_rate = rate_of_containing_from_means_stds(means, stds, level, 
                                                            estimate_of_true=args.central,
                                                            true_val = true_value_list[j])
                g_rate *= 100
                g_rates.append(g_rate)

            g_rate_list[j] = g_rates
        except:
            continue

    pickle.dump({"LEVELS_PERCENT": LEVELS_PERCENT, "g_rates": g_rate_list},
                open(''.join(['MLE_', str(int(len(MLE_results.columns)/2)), '.pkl']), "wb"))
else:
    print("No NLS file.")

# error_propagation ls cis
if len(args.propagation_error_result_file)>0:
    
    Propagation_results = pd.read_csv(args.propagation_error_result_file, index_col=0)
    Propagation_results = Propagation_results.reindex(ordered_experiment_names)

    for exper in Propagation_results.index:
        if np.any(Propagation_results.loc[exper].isnull()):
            raise Exception(exper + " is null")

    p_rate_list = {}
    for j in params_name:
        try:
            means = Propagation_results[j]
            stds  = Propagation_results[j+"_std"]
            p_rates = []
            for level in LEVELS:
                p_rate = rate_of_containing_from_means_stds(means, stds, level, estimate_of_true=args.central)

                p_rate *= 100
                p_rates.append(p_rate)

            p_rate_list[j] = p_rates
        except: 
            continue

    pickle.dump({"LEVELS_PERCENT": LEVELS_PERCENT, "p_rates": p_rate_list},
                open("Error_Propagation.pkl", "wb"))
else:
    print("No error propagation file.")

# Plotting
plt.rcParams["figure.autolayout"] = True
nrows = 1
fig, axes = plt.subplots(nrows, 2, figsize=(11, 5*nrows), sharex=False, sharey=False)
axes = axes.flatten()
for i in range(len(params_name)): 
    plot_containing_rates([LEVELS_PERCENT], [b_rates_list[params_name[i]]],
                          observed_rate_errors=[b_rate_errors_list[params_name[i]]],
                          xlabel=XLABEL+' '+params_name_greek[params_name[i]], ylabel=YLABEL+' '+params_name_greek[params_name[i]], 
                          xlimits=[0, 100], ylimits=[0, 100], markers=MARKERS[0], fontsize=14,
                          ax=axes[i], colors=COLORS[0], label='Bayesian')
    try: 
        plot_containing_rates([LEVELS_PERCENT], [p_rate_list[params_name[i]]],
                              observed_rate_errors=None,
                              xlabel=XLABEL+' '+params_name_greek[params_name[i]], ylabel=YLABEL+' '+params_name_greek[params_name[i]], 
                              xlimits=[0, 100], ylimits=[0, 100], markers=MARKERS[1], fontsize=14,
                              ax=axes[i], colors=COLORS[1], label='NLS+EP')
        plot_containing_rates([LEVELS_PERCENT], [g_rate_list[params_name[i]]],
                              observed_rate_errors=None,
                              xlabel=XLABEL+' '+params_name_greek[params_name[i]], ylabel=YLABEL+' '+params_name_greek[params_name[i]], 
                              xlimits=[0, 100], ylimits=[0, 100], markers=MARKERS[2], fontsize=14, 
                              ax=axes[i], colors=COLORS[2], label='NLS')
    except: 
        continue
    handles, labels = axes[i].get_legend_handles_labels()

by_label = dict(zip(labels, handles))
# axes[5].set_visible(False)
# legend = fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.1, 0.5));
plt.savefig('Containing_Plot.pdf', dpi=100)#, bbox_extra_artists=[legend], bbox_inches='tight')
