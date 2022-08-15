#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_submit_bitc_numpyro_jobs.py"

export ITC_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export EXP_DES_PAR_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.experimental_information"

export HEAT_FILE_SUF=".DAT"

export KEY=10

export DS=0.001

export UNIFP0=" --uniform_cell_concentration"

export CONC_RANGE_FAC=10

export split=0

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --experimental_design_parameters_dir $EXP_DES_PAR_DIR --heat_file_suffix $HEAT_FILE_SUF --ds $DS $UNIFP0 --concentration_range_factor $CONC_RANGE_FAC --split_by $split --random_key $KEY
