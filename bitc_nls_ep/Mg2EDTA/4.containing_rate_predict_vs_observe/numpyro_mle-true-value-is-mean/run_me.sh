#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_plot_containing_rate_of_cis_Mg2EDTA.py"

export MCMC="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/3.bitc_numpyro_lognomP0_lognomLs"

export NLS="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/3.nls/MLE_5_parameters.csv"

export EP="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/3.nls/Propagation_5_parameters.csv"

export EXPE1="Mg1EDTAp1a Mg1EDTAp1b Mg1EDTAp1c Mg1EDTAp1d Mg1EDTAp1e"

export EXPE2="Mgp5EDTAp05a Mgp5EDTAp05b Mgp5EDTAp05c Mgp5EDTAp05e Mgp5EDTAp05f Mgp5EDTAp05g Mgp5EDTAp05h Mgp5EDTAp05i Mgp5EDTAp05j"

export CENTRAL="mean"

python $SCRIPT --bitc_mcmc_dir $MCMC --nonlinear_fit_result_file $NLS --propagation_error_result_file $EP --central $CENTRAL --ordered_experiment_names_1 "$EXPE1" --ordered_experiment_names_2 "$EXPE2"
