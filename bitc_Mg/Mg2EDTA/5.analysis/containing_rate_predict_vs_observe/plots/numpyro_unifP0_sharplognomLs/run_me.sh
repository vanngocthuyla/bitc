#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_plot_containing_rate_of_cis_combine_all_curves_3.py"

export MCMC="/home/vla/python/bitc_Mg/Mg2EDTA/2.bitc_numpyro_unifP0_sharplognomLs"

export MLE="/home/vla/python/bitc_Mg/Mg2EDTA/2.nls/MLE_5_parameters.csv"

export EP="/home/vla/python/bitc_Mg/Mg2EDTA/2.nls/Propagation_5_parameters.csv"

export EXPE1="Mg1EDTAp1a Mg1EDTAp1b Mg1EDTAp1c Mg1EDTAp1d Mg1EDTAp1e"

export EXPE2="Mgp5EDTAp05a Mgp5EDTAp05b Mgp5EDTAp05c Mgp5EDTAp05e Mgp5EDTAp05f Mgp5EDTAp05g Mgp5EDTAp05h Mgp5EDTAp05i Mgp5EDTAp05j"

export CENTRAL="median"

python $SCRIPT --bitc_mcmc_dir $MCMC --MLE_result_file $MLE --propagation_error_result_file $EP --central $CENTRAL --ordered_experiment_names_1 "$EXPE1" --ordered_experiment_names_2 "$EXPE2"
