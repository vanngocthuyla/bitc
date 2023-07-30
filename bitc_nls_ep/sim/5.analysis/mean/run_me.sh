#!/bin/bash

export SCRIPT="/home/vla/python/bitc_nls_ep/scripts/run_plot_containing_rate_of_cis.py"

export MCMC_DIR="/home/vla/python/bitc_nls_ep/sim/4.numpyro"

export NONLIN_RES="None"

export EXP_DES_PAR_DIR="/home/vla/python/bitc_nls_ep/sim/1.run_simulated_heats"

export EXPERIMENTS="0_sim 1_sim 2_sim 3_sim 4_sim 5_sim 6_sim 7_sim 8_sim 9_sim 10_sim 11_sim 12_sim 13_sim 14_sim 15_sim 16_sim 17_sim 18_sim 19_sim 20_sim 21_sim 22_sim 23_sim 24_sim 25_sim 26_sim 27_sim 28_sim 29_sim 30_sim 31_sim 32_sim 33_sim 34_sim 35_sim 36_sim 37_sim 38_sim 39_sim 40_sim 41_sim 42_sim 43_sim 44_sim 45_sim 46_sim 47_sim 48_sim 49_sim"

export CENTRAL="mean"

# DeltaG
python $SCRIPT --bitc_mcmc_dir $MCMC_DIR --nonlinear_fit_result_file $NONLIN_RES --ordered_experiment_names "$EXPERIMENTS" --central $CENTRAL --parameter "DeltaG" --experimental_design_parameters_dir $EXP_DES_PAR_DIR


# DeltaH
python $SCRIPT --bitc_mcmc_dir $MCMC_DIR --nonlinear_fit_result_file $NONLIN_RES --ordered_experiment_names "$EXPERIMENTS" --central $CENTRAL --parameter "DeltaH" --experimental_design_parameters_dir $EXP_DES_PAR_DIR
