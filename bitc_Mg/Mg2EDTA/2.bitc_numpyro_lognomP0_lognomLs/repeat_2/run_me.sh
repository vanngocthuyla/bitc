#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_submit_bitc_numpyro_jobs.py"

export ITC_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export EXP_DES_PAR_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.experimental_information"

export HEAT_FILE_SUF=".DAT"

export KEY=2

export DC=0.1

export DS=0.1

export split=0

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --experimental_design_parameters_dir $EXP_DES_PAR_DIR --heat_file_suffix $HEAT_FILE_SUF --dc $DC --ds $DS --split_by $split --random_key $KEY
