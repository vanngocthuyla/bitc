#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_nls_with_error_propagation.py"

export ITC_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/1.itc_origin_heat_files"

export EXP_DES_PAR_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/2.experimental_information"

export HEAT_FILE_SUF=".DAT"

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --experimental_design_parameters_dir $EXP_DES_PAR_DIR --heat_file_suffix $HEAT_FILE_SUF
