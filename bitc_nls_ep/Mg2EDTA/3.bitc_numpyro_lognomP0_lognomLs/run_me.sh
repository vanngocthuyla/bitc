#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_numpyro_Mg2EDTA.py"

export ITC_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/1.itc_origin_heat_files"

export EXP_DES_PAR_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/2.experimental_information"

export HEAT_FILE_SUF=".DAT"

export DC=0.1

export DS=0.1

export N_BURN=10000

export N_ITER=100000

export N_THIN=10

python $SCRIPT --itc_data_dir $ITC_DIR --heat_data_dir $HEAT_DIR --experimental_design_parameters_dir $EXP_DES_PAR_DIR --heat_file_suffix $HEAT_FILE_SUF --dc $DC --ds $DS --nburn $N_BURN --niters $N_ITER --nthin $N_THIN
