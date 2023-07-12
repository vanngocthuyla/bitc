import os
import glob
import argparse

import numpy as np
import pickle
import arviz as az
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from _data_files import read_experimental_design_parameters
from _mcmc_numpyro import load_heat_micro_cal
from _mcmc_numpyro import Bayesian_multiple_expt_fitting

print(f'Using numpyro {numpyro.__version__}')
print(f'Using jax {jax.__version__}')

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",    type=str,       default="")
parser.add_argument( "--itc_data_dir",                          type=str,       default="")
parser.add_argument( "--heat_data_dir",                         type=str,       default="")
parser.add_argument( "--experiments",                           type=str,       default="")
parser.add_argument( "--heat_file_suffix",                      type=str,       default=".DAT")

parser.add_argument( "--dc",                                    type=float,     default=0.1)      # cell concentration relative uncertainty
parser.add_argument( "--ds",                                    type=float,     default=0.1)      # syringe concentration relative uncertainty

parser.add_argument( "--dummy_itc_file",                action="store_true",    default=False)

parser.add_argument( "--uniform_cell_concentration",    action="store_true",    default=False)
parser.add_argument( "--uniform_syringe_concentration", action="store_true",    default=False)
parser.add_argument( "--concentration_range_factor",            type=float,     default=10.)

parser.add_argument( "--niters",                                type=int,       default=100000)
parser.add_argument( "--nburn",                                 type=int,       default=10000)
parser.add_argument( "--nthin",                                 type=int,       default=10)
parser.add_argument( "--nchain",                                type=int,       default=4)
parser.add_argument( "--random_key",                            type=int,       default=0)
parser.add_argument( "--verbosity",                             type=str,       default="-vvv")

args = parser.parse_args()

TRACES_FILE = "traces.pickle"

if len(args.experiments)==0: 
    itc_data_files = glob.glob(os.path.join(args.itc_data_dir, "*.itc"))
    itc_data_files = [os.path.basename(f) for f in itc_data_files]

    exper_names = [f.split(".itc")[0] for f in itc_data_files]
    for name in exper_names:
        if not os.path.isfile( os.path.join(args.heat_data_dir, name + args.heat_file_suffix) ):
            print("WARNING: Integrated heat file for " + name + " does not exist")
    exper_names = [name for name in exper_names if os.path.isfile( os.path.join(args.heat_data_dir, name + args.heat_file_suffix) ) ]

    exclude_experiments = args.exclude_experiments.split()
    exper_names = [name for name in exper_names if name not in exclude_experiments]
else: 
    exper_names = args.experiments.split()

assert len(args.experimental_design_parameters_dir)>0, "Please provide the directory of experimental design parameters."
parameters = read_experimental_design_parameters(args.experimental_design_parameters_dir+'/experimental_design_parameters.dat')

print("Will run these", exper_names)

rng_key = random.split(random.PRNGKey(args.random_key), args.nchain)
KB = 0.0019872041 # in kcal/mol/K

for name in exper_names: 
    if not os.path.exists(name): 
        os.mkdir(name)
    if not os.path.exists(name+'/'+'traces.pickle'):
        print("Running", name)
        itc_file = os.path.join(args.itc_data_dir, name+".itc")
        integ_file = os.path.join(args.heat_data_dir, name + args.heat_file_suffix)

        q_actual_micro_cal = load_heat_micro_cal(integ_file)
        q_actual_cal = q_actual_micro_cal * 1e-6
        
        n_injections = len(q_actual_cal)
        INJ_VOL = parameters[name]["injection_volume"]*1e-6               # in liter
        injection_volumes = [INJ_VOL for _ in range(n_injections)]        # in liter
        SYRINGE_CONCENTR = parameters[name]["syringe_concentration"]      # milli molar
        CELL_CONCENTR = parameters[name]['cell_concentration']            # milli molar
        Bayesian_multiple_expt_fitting(rng_key, q_actual_cal, injection_volumes, 
                                       CELL_CONCENTR, SYRINGE_CONCENTR, args.dc, args.ds,
                                       args.niters, args.nburn, args.nchain, args.nthin, 
                                       args.uniform_cell_concentration, args.uniform_syringe_concentration, 
                                       args.concentration_range_factor, name, name)

print("DONE")
