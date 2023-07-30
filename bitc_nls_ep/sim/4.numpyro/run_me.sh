#!/bin/bash

export SCRIPT="/home/vla/python/bitc_nls_ep/scripts/submit_bitc_numpyro_sim.py"

export ITC_DIR="/home/vla/python/bitc_nls_ep/sim/2.dummy_itc_files"

export HEAT_DIR="/home/vla/python/bitc_nls_ep/sim/3.heat_in_origin_format"

export EXP_DES_PAR_DIR="/home/vla/python/bitc_nls_ep/sim/1.run_simulated_heats"

export HEAT_FILE_SUF=".DAT"

export DC=0.1

export DS=0.1

export DUM_ITC=" --dummy_itc_file "

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --experimental_design_parameters_dir $EXP_DES_PAR_DIR --heat_file_suffix $HEAT_FILE_SUF --dc $DC --ds $DS $DUM_ITC
