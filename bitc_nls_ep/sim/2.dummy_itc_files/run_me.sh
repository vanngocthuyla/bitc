#!/bin/bash

export SCRIPT="/home/vla/python/bitc_nls_ep/scripts/run_write_dummy_itc_files.py"

export EXP_DES_PAR_FILE="/home/vla/python/bitc_nls_ep/sim/1.run_simulated_heats/experimental_design_parameters.dat"

export DIG_HEAT_DIR="/home/vla/python/bitc_nls_ep/sim/1.run_simulated_heats"

python $SCRIPT --experimental_design_parameters_file $EXP_DES_PAR_FILE --digitized_heat_dir $DIG_HEAT_DIR
