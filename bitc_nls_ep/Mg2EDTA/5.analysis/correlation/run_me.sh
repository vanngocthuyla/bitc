#!/bin/bash

export SCRIPT="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/scripts/run_display_correlation_matrix.py"

export MCMC_DIR="/Users/seneysophie/Work/Python/Local/bitc_nls_ep/Mg2EDTA/2.bitc_mcmc_lognomP0_lognomLs"

export EXPERIMENT="Mg1EDTAp1a"

python $SCRIPT --repeated_bitc_mcmc_dir $MCMC_DIR --experiment $EXPERIMENT

