#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_plot_containing_rate_of_cis_combine_all_curves.py"

export LOG_NOR_RES_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/5.analysis/containing_rate_predict_vs_observe/lognomP0_lognomLs"

export UNIF_RES_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/5.analysis/containing_rate_predict_vs_observe/unifP0_sharplognomLs"

python $SCRIPT --log_normal_results_dir $LOG_NOR_RES_DIR --uniform_results_dir $UNIF_RES_DIR --parameter "DeltaG"

python $SCRIPT --log_normal_results_dir $LOG_NOR_RES_DIR --uniform_results_dir $UNIF_RES_DIR --parameter "DeltaH"
