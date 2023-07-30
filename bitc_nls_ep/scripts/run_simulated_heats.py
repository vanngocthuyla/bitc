"""
run simulated heat
"""
from __future__ import print_function
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pymc

from _simulated_heats import _sample_lognormal, expected_injection_heats, _write_heat_to_csv, _write_exper_design_file

from matplotlib import font_manager
font_dirs = ['/home/vla/python/fonts/arial']
for font in font_manager.findSystemFonts(font_dirs):
    font_manager.fontManager.addfont(font)
matplotlib.rcParams['font.family'] = ['arial']

parser = argparse.ArgumentParser()

parser.add_argument( "--true_DeltaG_kcal_per_mol",          type=float, default=-10)
parser.add_argument( "--true_DeltaH_kcal_per_mol",          type=float, default=-5)
parser.add_argument( "--true_DeltaH_0_micro_cal",           type=float, default=0.5)
parser.add_argument( "--sigma_micro_cal",                   type=float, default=1.)

parser.add_argument( "--stated_Ls_mM",                      type=float, default=1.)
parser.add_argument( "--stated_R0_mM",                      type=float, default=0.1)
parser.add_argument( "--concentration_uncertainty",         type=float, default=0.1)

parser.add_argument( "--cell_vol_mili_litter",              type=float, default=1.43)
parser.add_argument( "--injection_vol_micro_litter",        type=float, default=12.)

parser.add_argument( "--number_of_injections",              type=int,   default=24)
parser.add_argument( "--number_of_sim_exper",               type=int,   default=1000)
parser.add_argument( "--name_suffix",                       type=str,   default="sim")

parser.add_argument( "--experimental_design_out",           type=str,   default="experimental_design_parameters.dat")
parser.add_argument( "--sim_heat_out",                      type=str,   default="sim_heats.pdf")

args = parser.parse_args()

figure_size=(3.2, 2.4)
lw=1.0
font = {"fontname": "Arial"}
fontsize=14
dpi=300

TEMPERATURE = 298.15
KB = 0.0019872041      # in kcal/mol/K
BETA = 1 / TEMPERATURE / KB

V0_litter = args.cell_vol_mili_litter * 10**(-3)
DeltaVn_liiter = [args.injection_vol_micro_litter * 10**(-6) for _ in range(args.number_of_injections)]

P0s = _sample_lognormal("P0", args.stated_R0_mM, args.concentration_uncertainty, args.number_of_sim_exper)
Lss = _sample_lognormal("Ls", args.stated_Ls_mM, args.concentration_uncertainty, args.number_of_sim_exper)

true_DeltaH_0_cal = args.true_DeltaH_0_micro_cal * 10**(-6)
sigma_cal = args.sigma_micro_cal * 10**(-6)

plt.figure(figsize=figure_size)
plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))

numpy.random.seed(0)

for i in range(args.number_of_sim_exper):
    qn_cal = expected_injection_heats(V0_litter, DeltaVn_liiter, P0s[i], Lss[i],
                                      args.true_DeltaG_kcal_per_mol, args.true_DeltaH_kcal_per_mol,
                                      true_DeltaH_0_cal, BETA, args.number_of_injections)

    qn_cal = qn_cal + numpy.random.normal(loc=0., scale=sigma_cal, size=args.number_of_injections)

    out_file = "%d_"%i + args.name_suffix + ".csv"
    _write_heat_to_csv(qn_cal, args.injection_vol_micro_litter, args.stated_Ls_mM, out_file)

    injection_indices = numpy.arange(args.number_of_injections) + 1
    plt.plot(injection_indices, qn_cal, linestyle="-", marker=None, ms=3, lw=lw)

plt.xlabel("$n$", fontsize=fontsize, **font)
plt.ylabel("$q^*_n$ (cal)", fontsize=fontsize, **font)
plt.tight_layout()
plt.savefig(args.sim_heat_out)

_write_exper_design_file(args.number_of_sim_exper, args.name_suffix,
                         args.stated_Ls_mM, args.stated_R0_mM,
                         args.number_of_injections,
                         args.injection_vol_micro_litter,
                         args.experimental_design_out)

_write_exper_design_file(args.number_of_sim_exper, args.name_suffix,
                         Lss, P0s, 
                         args.number_of_injections,
                         args.injection_vol_micro_litter,
                         'true_'+args.experimental_design_out)

print("DONE")
