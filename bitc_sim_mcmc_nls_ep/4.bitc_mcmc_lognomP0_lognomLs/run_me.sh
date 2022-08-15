#!/bin/bash

export SCRIPT="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/scripts/run_submit_bitc_mcmc_jobs.py"

export ITC_DIR="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/2.dummy_itc_files"

export HEAT_DIR="/home/vla/python/bitc/bitc_sim_mcmc_nls_ep/3.heat_in_origin_format"

export HEAT_FILE_SUF=".DAT"

export DC=0.1

export DS=0.1

export DUM_ITC=" --dummy_itc_file "

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --heat_file_suffix $HEAT_FILE_SUF --dc $DC --ds $DS $DUM_ITC

