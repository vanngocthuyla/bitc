#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_submit_bitc_mcmc_jobs.py"

export ITC_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_FILE_SUF=".DAT"

export DS=0.001

export UNIFP0=" --uniform_cell_concentration"

export CONC_RANGE_FAC=10

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --heat_file_suffix $HEAT_FILE_SUF --ds $DS $UNIFP0 --concentration_range_factor $CONC_RANGE_FAC
