#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_submit_bitc_mcmc_jobs.py"

export ITC_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_FILE_SUF=".DAT"

export DC=0.1

export DS=0.1

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --heat_file_suffix $HEAT_FILE_SUF --dc $DC --ds $DS
