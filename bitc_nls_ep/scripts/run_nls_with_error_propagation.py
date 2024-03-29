"""
fit the MLE model with 3 or 5 parameters for ITC data, then adjust the error of parameters by the error propagation 
"""
import sys
import os
import numpy as np
import pickle
from scipy.optimize import minimize, curve_fit
import pandas as pd
import matplotlib.pyplot as plt

import glob
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from _data_files import read_experimental_design_parameters
from _load_data import load_heat_micro_cal
from _mcmc_numpyro import deltaH0_guesses
from _nls_with_error_propagation import fitted_function, fitted_function_fixed_Ls, fitted_function_fixed_P0_Ls, plot_ITC_curve

sys.path.append('/home/vla/python/bitc_nls_ep/bayesian-itc')

from bitc.experiments import ExperimentMicroCal, ExperimentMicroCalWithDummyITC, ExperimentYaml
from bitc.instruments import known_instruments, Instrument
from bitc.units import Quantity

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",    type=str,   default="")
parser.add_argument( "--itc_data_dir",                          type=str,   default="1.itc_origin_heat_files")
parser.add_argument( "--heat_data_dir",                         type=str,   default="1.itc_origin_heat_files")
parser.add_argument( "--exclude_experiments",                   type=str,   default="")
parser.add_argument( "--script",                                type=str,   default="")
parser.add_argument( "--heat_file_suffix",                      type=str,   default=".DAT")
parser.add_argument( "--fixed_Ls",                    action="store_true",  default=False)
parser.add_argument( "--fixed_P0",                    action="store_true",  default=False)
parser.add_argument( "--ordered_experiment_names",              type=str,   default=" ")

args = parser.parse_args()

assert args.fixed_Ls in [True, False], "The choice for fixed_concentration is only \"True\" or \"False\"."
assert args.fixed_P0 in [True, False], "The choice for fixed_concentration is only \"True\" or \"False\"."
assert len(args.experimental_design_parameters_dir)>0, "Please provide the directory of experimental design parameters."

if not args.ordered_experiment_names == ' ':
    exper_names = args.ordered_experiment_names.split()
else: 
    itc_data_files = glob.glob(os.path.join(args.itc_data_dir, "*.itc"))
    itc_data_files = [os.path.basename(f) for f in itc_data_files]
    exper_names = [f.split(".itc")[0] for f in itc_data_files]

    for name in exper_names:
        if not os.path.isfile( os.path.join(args.heat_data_dir, name + args.heat_file_suffix) ):
            print("WARNING: Integrated heat file for " + name + " does not exist")
    exper_names = [name for name in exper_names if os.path.isfile( os.path.join(args.heat_data_dir, name + args.heat_file_suffix) ) ]

    exclude_experiments = args.exclude_experiments.split()
    exper_names = [name for name in exper_names if name not in exclude_experiments]

TEMPERATURE = 298.15
KB = 0.0019872041      # in kcal/mol/K
BETA = 1 / TEMPERATURE / KB

print("Fitting the NLS for ITC data: ")

data_params = {}
data_error = {}
for name in exper_names:
    print("Running the", name)
    itc_file = os.path.join(args.itc_data_dir, name+".itc")
    integ_file = os.path.join(args.heat_data_dir, name + args.heat_file_suffix)
    experiment_design_file = os.path.join(args.experimental_design_parameters_dir, name+'.pickle')

    q_actual_micro_cal = load_heat_micro_cal(integ_file)
    q_actual_cal = q_actual_micro_cal * 1e-6

    experiment = pickle.load(open(experiment_design_file, "rb") )
    n_injections = experiment['number_of_injections']
    injection_volumes = []
    for inj, injection in enumerate(experiment['injections']):
        injection_volumes.append(injection.volume.magnitude)
    injection_volumes = np.array(injection_volumes)*1e-6                              # in liter
    SYRINGE_CONCENTR = experiment['syringe_concentration']['ligand'].magnitude        # milli molar
    CELL_CONCENTR = experiment['cell_concentration']['macromolecule'].magnitude

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal) #cal
    plt.figure()
    if args.fixed_Ls==False and args.fixed_P0==False:
        params_guess = [CELL_CONCENTR, SYRINGE_CONCENTR, -8, -7, DeltaH_0_min] #P0, Ls, DeltaG, DeltaH, DeltaH_0  
        lower = [CELL_CONCENTR*0.5, SYRINGE_CONCENTR*0.5, -40.0, -100, DeltaH_0_min]
        upper = [CELL_CONCENTR*2, SYRINGE_CONCENTR*2, 0, 100, DeltaH_0_max]
        fit_f, sigma = fitted_function(injection_volumes, q_actual_cal, 1.43*10**(-3), 
                                       BETA, n_injections, params_guess, (lower, upper))
        data_params[name] = params = fit_f 
        data_error[name] = sigma
        plot_ITC_curve(q_actual_cal, fit_f, 1.43*10**(-3), injection_volumes, BETA, n_injections, os.path.join('5_params', str('5_'+name)))
    elif args.fixed_Ls==True and args.fixed_P0==False:
        params_guess = [CELL_CONCENTR, -8, -7, DeltaH_0_min] #P0, DeltaG, DeltaH, DeltaH_0  
        lower = [CELL_CONCENTR*0.1, -40.0, -100, DeltaH_0_min]
        upper = [CELL_CONCENTR*10, 0, 100, DeltaH_0_max]
        fit_f, sigma = fitted_function_fixed_Ls(injection_volumes, q_actual_cal, SYRINGE_CONCENTR, 1.43*10**(-3), 
                                                BETA, n_injections, params_guess, (lower, upper))
        data_params[name] = fit_f
        data_error[name] = sigma
        params = [fit_f[0], SYRINGE_CONCENTR, *fit_f[1:]]
        plot_ITC_curve(q_actual_cal, params, 1.43*10**(-3), injection_volumes, BETA, n_injections, os.path.join('4_params', str('4_'+name)))
    elif args.fixed_Ls==True and args.fixed_P0==True:
        params_guess = [-10, -7, DeltaH_0_min] #DeltaG, DeltaH, DeltaH_0
        lower = [-40.0, -100, DeltaH_0_min]
        upper = [0, 100, DeltaH_0_max]
        fit_f, sigma = fitted_function_fixed_P0_Ls(injection_volumes, q_actual_cal, CELL_CONCENTR, 
                                                   SYRINGE_CONCENTR, 1.43*10**(-3), BETA, n_injections, 
                                                   params_guess, (lower, upper))
        data_params[name] = fit_f
        data_error[name] = sigma
        params = [CELL_CONCENTR, SYRINGE_CONCENTR, *fit_f]
        plot_ITC_curve(q_actual_cal, params, 1.43*10**(-3), injection_volumes, BETA, n_injections, os.path.join('3_params', str('3_'+name)))
    else: 
        print("P0 often varies while Ls is fixed.")


data_params = pd.DataFrame.from_dict(data_params, orient='index')
data_error = pd.DataFrame.from_dict(data_error, orient='index')
if args.fixed_Ls==False and args.fixed_P0==False: 
    data_params.columns = ['P0', 'Ls', 'DeltaG', 'DeltaH', 'DeltaH_0']
    data_error.columns = ['P0_std', 'Ls_std', 'DeltaG_std', 'DeltaH_std', 'DeltaH_0_std']
elif args.fixed_Ls==True and args.fixed_P0==True: 
    data_params.columns = ['DeltaG', 'DeltaH', 'DeltaH_0']
    data_error.columns = ['DeltaG_std', 'DeltaH_std', 'DeltaH_0_std']
elif args.fixed_Ls==True and args.fixed_P0==False:
    print("Fitting 4 parameters: P0, DeltaG, DeltaH, DeltaH_0")
    data_params.columns = ['P0', 'DeltaG', 'DeltaH', 'DeltaH_0']
    data_error.columns = ['P0_std', 'DeltaG_std', 'DeltaH_std', 'DeltaH_0_std']

MLE_data = data_params.join(data_error)

if args.fixed_Ls==False and args.fixed_P0==False: 
    print('Saving MLE model with 5 parameters')
    pickle.dump(MLE_data.to_pickle, open(os.path.join('MLE_5_parameters.pickle'), "wb"))
    MLE_data.to_csv('MLE_5_parameters.csv')
elif args.fixed_Ls==True and args.fixed_P0==True: 
    print('Saving MLE model with 3 parameters')
    pickle.dump(MLE_data.to_pickle, open(os.path.join('MLE_3_parameters.pickle'), "wb"))
    MLE_data.to_csv('MLE_3_parameters.csv')
elif args.fixed_Ls==True and args.fixed_P0==False:
    print('Saving MLE model with 4 parameters')
    pickle.dump(MLE_data.to_pickle, open(os.path.join('MLE_4_parameters.pickle'), "wb"))
    MLE_data.to_csv('MLE_4_parameters.csv')

# We don't know true values of Ls and P0, so we used mean of estimated values
MLE_data = pd.read_csv('MLE_5_parameters.csv', index_col=0)
true_Ls = np.concatenate((np.repeat(MLE_data[0:5].mean()['Ls'], 5), 
                          np.repeat(MLE_data[5:].mean()['Ls'], 9)), 
                         axis=0)
true_P0 = np.concatenate((np.repeat(MLE_data[0:5].mean()['P0'], 5), 
                          np.repeat(MLE_data[5:].mean()['P0'], 9)), 
                         axis=0)

if args.fixed_Ls==False and args.fixed_P0==False:
    propagation_params_list = ['P0_std', 'Ls_std', 'DeltaG_std', 'DeltaH_std']
elif args.fixed_Ls==True and args.fixed_P0==True: 
    propagation_params_list = ['DeltaG_std', 'DeltaH_std']
elif args.fixed_Ls==True and args.fixed_P0==False: 
    propagation_params_list = ['P0_std', 'DeltaG_std', 'DeltaH_std']

print("Parameters adjusted with error propagation: ", propagation_params_list)

propagation_std = {}
for i in range(len(exper_names)):
    name = exper_names[i]
    print("Running the", name)

    experiment_design_file = os.path.join(args.experimental_design_parameters_dir, name+'.pickle')
    experiment = pickle.load(open(experiment_design_file, "rb") )
    SYRINGE_CONCENTR = experiment['syringe_concentration']['ligand'].magnitude        # milli molar
    CELL_CONCENTR = experiment['cell_concentration']['macromolecule'].magnitude

    Ls_error = abs(true_Ls[i]-SYRINGE_CONCENTR)
    P0_error = abs(true_P0[i]-CELL_CONCENTR)
    
    error = []
    for i in data_error.columns: 
        if i in propagation_params_list:
            if i == 'P0_std':
                error.append(np.sqrt((data_error[i][name])**2+P0_error**2))
            else: 
                error.append(np.sqrt((data_error[i][name])**2+Ls_error**2))
        else: 
            error.append(data_error[i][name])
    propagation_std[name] = error

propagation_std = pd.DataFrame.from_dict(propagation_std, orient='index')
propagation_std.columns = data_error.columns

propagation_data = data_params.join(propagation_std)

if args.fixed_Ls==False and args.fixed_P0==False:
    print('MLE fitted with 5 parameters adjusted by error propagation')
    pickle.dump(propagation_data.to_pickle, open(os.path.join('Propagation_5_parameters.pickle'), "wb"))
    propagation_data.to_csv('Propagation_5_parameters.csv')
elif args.fixed_Ls==True and args.fixed_P0==True: 
    print('MLE fitted with 3 parameters adjusted by error propagation')
    pickle.dump(propagation_data.to_pickle, open(os.path.join('Propagation_3_parameters.pickle'), "wb"))
    propagation_data.to_csv('Propagation_3_parameters.csv')
elif args.fixed_Ls==True and args.fixed_P0==False: 
    print('MLE fitted with 4 parameters adjusted by error propagation')
    pickle.dump(propagation_data.to_pickle, open(os.path.join('Propagation_4_parameters.pickle'), "wb"))
    propagation_data.to_csv('Propagation_4_parameters.csv')

print("DONE")
