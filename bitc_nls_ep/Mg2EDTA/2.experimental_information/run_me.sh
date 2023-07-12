#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_write_expt_infor_Mg2EDTA.py"

export OUT_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/2.experimental_information"

export ITC_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/1.itc_origin_heat_files"

export HEAT_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/1.itc_origin_heat_files"

python $SCRIPT --out_dir $OUT_DIR --dat_file_dir $ITC_DIR --heat_file_dir $HEAT_DIR