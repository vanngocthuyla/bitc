#!/bin/bash

export SCRIPT="/home/vla/python/bitc_Mg/scripts/run_convert_ka_to_dg.py"

export ITC_DIR="/home/vla/python/bitc_Mg/Mg2EDTA/1.itc_origin_heat_files"

export ORIGIN_PAR_FILE="origin_ka_dh_in_cal_per_mole.dat"

export ENERGY_UNIT="cal_per_mole"

export OUT="origin_dg_dh_in_kcal_per_mole.dat"

python $SCRIPT --origin_par_file $ORIGIN_PAR_FILE --itc_file_dir $ITC_DIR --input_energy_unit $ENERGY_UNIT --write_header --out $OUT

