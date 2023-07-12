#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_write_heat_in_origin_format.py"

export EXP_DES_PAR_FILE="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/sim/1.run_simulated_heats/experimental_design_parameters.dat"

export DIG_HEAT_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/sim/1.run_simulated_heats"

python $SCRIPT --experimental_design_parameters_file $EXP_DES_PAR_FILE --digitized_heat_dir $DIG_HEAT_DIR

