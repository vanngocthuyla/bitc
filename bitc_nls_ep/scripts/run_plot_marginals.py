

import os
import argparse
import pickle

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
sns.set_style("white")

from _trace_analysis import _extract_dg_dh_p0_ls
from _confidence_intervals import bayesian_credible_interval
from _plot_marginals import _plot_1d_kde, _plot_2d_kde
from _data_files import cell_concentration


parser = argparse.ArgumentParser()
parser.add_argument( "--bitc_mcmc_dir",     type=str, default="bitc_mcmc")
parser.add_argument( "--itc_files_dir",     type=str, default="itc_origin_files")
parser.add_argument( "--experiment",        type=str, default=" " )
parser.add_argument( "--level",             type=float, default=0.95)
args = parser.parse_args()

TRACES_FILE = "traces.pickle"
YLABEL = "Probability density"

itc_file = os.path.join(args.itc_files_dir, args.experiment+".itc") 
print itc_file

trace_file = os.path.join(args.bitc_mcmc_dir, args.experiment, TRACES_FILE)
print trace_file

dg, dh, tds, p0, ls = _extract_dg_dh_p0_ls(trace_file)

# dg
lower_dg, upper_dg, _, _ = bayesian_credible_interval(dg, args.level, bootstrap_repeats=1)
out = args.experiment + "_1d_dg.pdf"
_plot_1d_kde( dg, out, xlabel="$\Delta G$ (kcal/mol)", ylabel=YLABEL, horizontal_line=[lower_dg, upper_dg, 3], marker_at=None )

# dh
lower_dh, upper_dh, _, _ = bayesian_credible_interval(dh, args.level, bootstrap_repeats=1)
out = args.experiment + "_1d_dh.pdf"
_plot_1d_kde(dh, out, xlabel="$\Delta H$ (kcal/mol)", ylabel=YLABEL, horizontal_line=[lower_dh, upper_dh, 1.5], marker_at=None )

# dh
lower_tds, upper_tds, _, _ = bayesian_credible_interval(tds, args.level, bootstrap_repeats=1)
out = args.experiment + "_1d_tds.pdf"
_plot_1d_kde(tds, out, xlabel="$T \Delta S$ (kcal/mol)", ylabel=YLABEL, horizontal_line=[lower_tds, upper_tds, 1.5], marker_at=None )

# p0
lower_p0, upper_p0, _, _ = bayesian_credible_interval(p0, args.level, bootstrap_repeats=1)
stated_value = cell_concentration(itc_file, is_dummy_itc_file=True) 
out = args.experiment + "_1d_p0.pdf"
_plot_1d_kde( p0, out, xlabel="$[R]_0$ (mM)", ylabel=YLABEL, horizontal_line=[lower_p0, upper_p0, 35], marker_at=[stated_value, 4] )


# 2d
# dg dh
out = args.experiment + "_2d_dg_dh.pdf"
_plot_2d_kde(dg, dh, out, xlabel="$\Delta G$ (kcal/mol)", ylabel="$\Delta H$ (kcal/mol)", zlabel="Probability density")

# dg tds
out = args.experiment + "_2d_dg_tds.pdf"
_plot_2d_kde(dg, tds, out, xlabel="$\Delta G$ (kcal/mol)", ylabel="$T \Delta S$ (kcal/mol)", zlabel="Probability density")

# dh tds
out = args.experiment + "_2d_dh_tds.pdf"
_plot_2d_kde(dh, tds, out, xlabel="$\Delta H$ (kcal/mol)", ylabel="$T \Delta S$ (kcal/mol)", zlabel="Probability density")

# p0 dh
out = args.experiment + "_2d_p0_dh.pdf"
_plot_2d_kde(p0, dh, out, xlabel="$[R]_0$ (mM)", ylabel="$\Delta H$ (kcal/mol)", zlabel="Probability density")