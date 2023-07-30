#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_plot_convergence_posterior.py"

export DATA_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/5.convergence/dat"

export EXPERIMENT="Mg1EDTAp1a Mg1EDTAp1b Mg1EDTAp1c Mg1EDTAp1d Mg1EDTAp1e Mgp5EDTAp05a Mgp5EDTAp05b Mgp5EDTAp05c Mgp5EDTAp05e Mgp5EDTAp05f Mgp5EDTAp05g Mgp5EDTAp05h Mgp5EDTAp05i Mgp5EDTAp05j"

python $SCRIPT --data_dir $DATA_DIR --experiment "$EXPERIMENT"