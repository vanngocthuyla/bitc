#!/bin/bash

export SCRIPT="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/scripts/run_write_dummy_itc_files.py"

export EXP_DES_PAR_FILE="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/1.run_simulated_heats/experimental_desgin_parameters.dat"

export DIG_HEAT_DIR="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/1.run_simulated_heats"

python $SCRIPT --experimental_desgin_parameters_file $EXP_DES_PAR_FILE --digitized_heat_dir $DIG_HEAT_DIR

