#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_plot_containing_rate_of_cis.py"

export MCMC_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/2.bitc_numpyro_unifP0_sharplognomLs"

export NONLIN_RES="/home/vla/python/bitc_Mg/Mg2EDTA/3.nonlinear_fit_results/origin_dg_dh_in_kcal_per_mole.dat"

export EXPERIMENTS="Mg1EDTAp1a Mg1EDTAp1b Mg1EDTAp1c Mg1EDTAp1d Mg1EDTAp1e Mgp5EDTAp05a Mgp5EDTAp05b Mgp5EDTAp05c Mgp5EDTAp05e Mgp5EDTAp05f Mgp5EDTAp05g Mgp5EDTAp05h Mgp5EDTAp05i Mgp5EDTAp05j"

# DeltaG
python $SCRIPT --bitc_mcmc_dir $MCMC_DIR --nonlinear_fit_result_file $NONLIN_RES --ordered_experiment_names "$EXPERIMENTS" --parameter "DeltaG" 

# DeltaH
python $SCRIPT --bitc_mcmc_dir $MCMC_DIR --nonlinear_fit_result_file $NONLIN_RES --ordered_experiment_names "$EXPERIMENTS" --parameter "DeltaH"

# DeltaH_0
python $SCRIPT --bitc_mcmc_dir $MCMC_DIR --ordered_experiment_names "$EXPERIMENTS" --parameter "DeltaH_0"