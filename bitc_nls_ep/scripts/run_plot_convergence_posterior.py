"""
to plot convergence curve for the percentiles of posteriors
"""

import argparse
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",       type=str,   default="")
parser.add_argument("--out_dir",        type=str,   default="")
parser.add_argument("--experiments",    type=str,   default="")

parser.add_argument("--percentiles",    type=str,   default="5 25 50 75 95")
parser.add_argument("--vars",           type=str,   default="DeltaG DeltaH DeltaH_0 P0 Ls log_sigma")
parser.add_argument("--xlabel",         type=str,   default="Sample proportion")

parser.add_argument("--font_scale",     type=float, default=0.75)

args = parser.parse_args()

sns.set(font_scale=args.font_scale)

experiments = args.experiments.split()
print("experiments:", experiments)

vars = args.vars.split()
assert len(vars) in [4, 6], "len of vars must be 4 or 6"
print("vars:", vars)

ylabels = {}
ylabels["P0"]       = "$[R]_0$ (mM)"
ylabels["Ls"]       = "$[L]_s$ (mM)"
ylabels["log_sigma"]= "$\ln \sigma$"
ylabels["DeltaH"]   = "$\Delta H$ (kcal/mol)"
ylabels["DeltaH_0"] = "$\Delta H_0$ ($\mu$cal)"
ylabels["DeltaH1"]  = "$\Delta H_1$ (kcal/mol)"
ylabels["DeltaH2"]  = "$\Delta H_2$ (kcal/mol)"
ylabels["DeltaG"]   = "$\Delta G$ (kcal/mol)"
ylabels["DeltaG1"]  = "$\Delta G_1$ (kcal/mol)"
ylabels["DeltaDeltaG"] = "$\Delta \Delta G$ (kcal/mol)"

xlabel = args.xlabel

qs = [float(s) for s in args.percentiles.split()]
data_cols = ["%0.1f-th" % q for q in qs]
print("data_cols:", data_cols)
err_cols = ["%0.1f-error" % q for q in qs]
print("err_cols:", err_cols)
legends = ["%d-th" % q for q in qs]

colors = ["b", "g", "r", "c", "m"]
#line_styles = ["solid", "dotted", "dashed", "dashdot", "solid"]
line_styles = ["solid" for _ in range(5)]
markers = ["o", "s", "d", "^", "v"]

if len(args.out_dir)>0:
    os.chdir(args.out_dir)

for exper in experiments:
    print("Ploting " + exper)

    if len(vars) == 4:
        fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(6.4, 4.4))
        plt.subplots_adjust(wspace=0., hspace=0.)
    elif len(vars) == 6:
        fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(6.4, 6.6))
        plt.subplots_adjust(wspace=0., hspace=0.)

    axes = axes.flatten()

    for var, ax in zip(vars, axes):
        # print(var)
        ylabel = ylabels[var]
        # print(ylabel)
        inp_file = os.path.join(args.data_dir, exper + "_" + var + ".dat")
        # print(inp_file)
        data = pd.read_csv(inp_file, sep="\s+")
        x = data["proportion"]

        for i, data_col in enumerate(data_cols):
            err_col = err_cols[i]
            color = colors[i]
            line_style = line_styles[i]
            marker = markers[i]
            legend = legends[i]

            y = data[data_col]
            yerr = data[err_col]

            # if var == 'DeltaH_0':
            #     y = y*1E6
            #     yerr = yerr*1E6

            ax.errorbar(x, y, yerr=yerr, linestyle=line_style, c=color, marker=marker, markersize=5, label=legend)

        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(xlabel)
    axes[-2].set_xlabel(xlabel)

    fig.tight_layout()
    fig.savefig(exper+'.pdf', dpi=300)

print("DONE")
