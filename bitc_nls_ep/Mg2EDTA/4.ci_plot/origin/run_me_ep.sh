#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_plot_bci_vs_gci.py"

export NLS_EP="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/3.nonlinear_fit_results/origin_error_propagation.csv"

export EXPE1="Mg1EDTAp1a Mg1EDTAp1b Mg1EDTAp1c Mg1EDTAp1d Mg1EDTAp1e"

export EXPE2="Mgp5EDTAp05a Mgp5EDTAp05b Mgp5EDTAp05c Mgp5EDTAp05e Mgp5EDTAp05f Mgp5EDTAp05g Mgp5EDTAp05h Mgp5EDTAp05i Mgp5EDTAp05j"

# DeltaG
python $SCRIPT --propagation_error_result_file $NLS_EP --ordered_experiment_names_1 "$EXPE1" --ordered_experiment_names_2 "$EXPE2" --parameter "DeltaG"

# DeltaH
python $SCRIPT --propagation_error_result_file $NLS_EP --ordered_experiment_names_1 "$EXPE1" --ordered_experiment_names_2 "$EXPE2"  --parameter "DeltaH"

# P0
python $SCRIPT --propagation_error_result_file $NLS_EP --ordered_experiment_names_1 "$EXPE1" --ordered_experiment_names_2 "$EXPE2"  --parameter "P0"

