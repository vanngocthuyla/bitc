import sys
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

sys.path.append('/Users/seneysophie/Work/Python/Local/bitc_nls_ep/bayesian-itc')

from bitc.experiments import ExperimentMicroCal, ExperimentMicroCalWithDummyITC, ExperimentYaml
from bitc.instruments import known_instruments, Instrument
from bitc.units import Quantity

print(f'Using numpyro {numpyro.__version__}')
print(f'Using jax {jax.__version__}')

parser = argparse.ArgumentParser()

parser.add_argument( "--itc_data_dir",                          type=str,       default="")
parser.add_argument( "--heat_data_dir",                         type=str,       default="")
parser.add_argument( "--experimental_design_parameters_dir",    type=str,       default="")
parser.add_argument( "--experiments",                           type=str,       default="")
parser.add_argument( "--exclude_experiments",                   type=str,       default="")

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

rng_key = random.split(random.PRNGKey(args.random_key), args.nchain)
KB = 0.0019872041 # in kcal/mol/K

for name in exper_names: 
    if not os.path.exists(name): 
        os.mkdir(name)
    if not os.path.exists(name+'/'+'traces.pickle'):
        print("Running", name)
        try: 
            itc_file = os.path.join(args.itc_data_dir, name+".itc")
            integ_file = os.path.join(args.heat_data_dir, name+args.heat_file_suffix)
            experiment_design_file = os.path.join(args.experimental_design_parameters_dir, name+'.pickle')

            q_actual_micro_cal = load_heat_micro_cal(integ_file)
            q_actual_cal = q_actual_micro_cal * 1e-6

            experiment = pickle.load(open(experiment_design_file, "rb"))
            n_injections = experiment['number_of_injections']

            injection_volumes = []
            for inj, injection in enumerate(experiment['injections']):
                injection_volumes.append(injection.volume.magnitude)
            injection_volumes = np.array(injection_volumes)*1e-6                              #in liter
            SYRINGE_CONCENTR = experiment['syringe_concentration']['ligand'].magnitude        # milli molar
            CELL_CONCENTR = experiment['cell_concentration']['macromolecule'].magnitude       # milli molar

            Bayesian_multiple_expt_fitting(rng_key, q_actual_cal, injection_volumes, 
                                           CELL_CONCENTR, SYRINGE_CONCENTR, args.dc, args.ds,
                                           args.niters, args.nburn, args.nchain, args.nthin, 
                                           uniform_P0=args.uniform_cell_concentration, 
                                           uniform_Ls=args.uniform_syringe_concentration, 
                                           concentration_range_factor=args.concentration_range_factor,
                                           name=name, OUT_DIR=name)
        except Exception as e:
            print('Error occurred:\n' + str(e))

print("DONE")
