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

from matplotlib import font_manager
font_dirs = ['/home/vla/python/fonts/arial']
for font in font_manager.findSystemFonts(font_dirs):
    font_manager.fontManager.addfont(font)
matplotlib.rcParams['font.family'] = ['arial']

from _confidence_intervals import rate_of_containing_from_sample, rate_of_containing_from_means_stds
from _plot_confidence_intervals import plot_containing_rates
from _data_files import read_experimental_design_parameters

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",    type=str,               default="")
parser.add_argument( "--bitc_mcmc_dir",                         type=str,               default="bitc_mcmc")
parser.add_argument( "--nonlinear_fit_result_file",             type=str,               default="MLE_parameters.csv")
parser.add_argument( "--propagation_error_result_file",         type=str,               default="Propagation_parameters.csv")

parser.add_argument( "--ordered_experiment_names",              type=str,               default="" )
parser.add_argument( "--parameter",                             type=str,               default="P0 Ls DeltaG DeltaH DeltaH_0")
parser.add_argument( "--central",                               type=str,               default="median")
parser.add_argument( "--true_value",                            action="store_true",    default=True)
parser.add_argument( "--true_value_concentration",              action="store_true",    default=True)

args = parser.parse_args()

MARKERS = ("<", ">", "o", "s")
COLORS = ("r", "b", "g", "k")
params_name = args.parameter.split() #["P0", "Ls", "DeltaG", "DeltaH", "DeltaH_0"]
params_name_greek = {"DeltaG"    :   "$\Delta G$ (kcal/mol)", 
                     "DeltaH"    :   "$\Delta H$ (kcal/mol)", 
                     "DeltaH_0"  :   "$\Delta H_0$ (cal)",
                     "P0"        :   "$[R]_0$ (mM)", 
                     "Ls"        :   "$[L]_s$ (mM)", 
                     "log_sigma" :   "$\ln \sigma$" }

ordered_experiment_names = args.ordered_experiment_names.split()
print("Working with", args.nonlinear_fit_result_file, "and", args.propagation_error_result_file)

assert args.central in ["mean", "median"], "wrong central"

TRACES_FILE = "traces.pickle"
XLABEL = "predicted"
YLABEL = "observed"

LEVELS_PERCENT = np.linspace(10., 95., num=18)
print("LEVELS_PERCENT", LEVELS_PERCENT)
LEVELS = LEVELS_PERCENT / 100.

assert len(args.experimental_design_parameters_dir)>0, "Please provide the directory of experimental design parameters."

print("Loading the TRUE experimental design parameter file.")
parameters_true = read_experimental_design_parameters(args.experimental_design_parameters_dir+'/true_experimental_design_parameters.dat')

Ls_list = []
P0_list = []
for name in ordered_experiment_names:
    Ls_list.append(abs(parameters_true[name]['syringe_concentration']))
    P0_list.append(abs(parameters_true[name]['cell_concentration']))
Ls_list = np.array(Ls_list)
P0_list = np.array(P0_list)

if args.true_value==True: 
    if args.true_value_concentration==True:
        true_value_list = {'P0': P0_list, 'Ls': Ls_list, 'DeltaG': -10, 'DeltaH': -5, 'DeltaH_0': 5e-7}
    else:
        true_value_list = {'P0': 0.1, 'Ls': 1.0, 'DeltaG': -10, 'DeltaH': -5, 'DeltaH_0': 5e-7}
    print("True value of parameters: ", true_value_list)
else: 
    true_value_list = {'P0': None, 'Ls': None, 'DeltaG': None, 'DeltaH': None, 'DeltaH_0': None}
    print("No true values of parameters provided.")


# bayesian cis
mcmc_trace_files = [os.path.join(args.bitc_mcmc_dir, exper_name, TRACES_FILE) for exper_name in ordered_experiment_names]

try: 
    b_file = pickle.load(open("Bayesian.pkl", "rb"))
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
    b_rates_list = {}
    b_rate_errors_list = {}
    for j in params_name:
        print(j)
        try: 
            b_file = pickle.load(open(j+"_bayesian.pkl", "rb"))
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

        b_rates_list[j] = b_rates
        b_rate_errors_list[j] = b_rate_errors

    pickle.dump({"LEVELS_PERCENT": LEVELS_PERCENT, "b_rates": b_rates_list, "b_rate_errors": b_rate_errors_list},
                open("Bayesian.pkl", "wb"))

# nonlinear ls cis
MLE_results = pd.read_csv(args.nonlinear_fit_result_file, index_col=0)
MLE_results = MLE_results.reindex(ordered_experiment_names)

for exper in MLE_results.index:
    if np.any(MLE_results.loc[exper].isnull()):
        raise Exception(exper + " is null")

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
            open("MLE.pkl", "wb"))


# error_propagation ls cis
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

# Plotting
plt.rcParams["figure.autolayout"] = True
fig, axes = plt.subplots(3, 2, figsize=(11, 5*3), sharex=False, sharey=False)
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
                              xlimits=[0, 100], ylimits=[0, 100], markers=MARKERS[1], 
                              ax=axes[i], colors=COLORS[1], label='NLS+EP')
        plot_containing_rates([LEVELS_PERCENT], [g_rate_list[params_name[i]]],
                              observed_rate_errors=None,
                              xlabel=XLABEL+' '+params_name_greek[params_name[i]], ylabel=YLABEL+' '+params_name_greek[params_name[i]], 
                              xlimits=[0, 100], ylimits=[0, 100], markers=MARKERS[2], 
                              ax=axes[i], colors=COLORS[2], label='NLS')
    except: 
        continue
    handles, labels = axes[i].get_legend_handles_labels()

by_label = dict(zip(labels, handles))
axes[5].set_visible(False)
legend = fig.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(1.1, 0.5));
plt.savefig('Containing_Plot', bbox_extra_artists=[legend], bbox_inches='tight')
