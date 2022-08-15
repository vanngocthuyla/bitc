"""
fit the MLE model with 3 or 5 parameters for ITC data, then adjust the error of parameters by the error propagation 
"""
import os
import numpy as np
import pickle
from scipy.optimize import minimize, curve_fit
import pandas as pd

import glob
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",  type=str, default="")
parser.add_argument( "--itc_data_dir",                type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--heat_data_dir",               type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--exclude_experiments",         type=str, default="")
parser.add_argument( "--script",                      type=str, default = dir)
parser.add_argument( "--heat_file_suffix",            type=str, default =".DAT")
parser.add_argument( "--fixed_Ls",                    type=int, default=0)
parser.add_argument( "--fixed_P0",                    type=int, default=0)
parser.add_argument( "--ordered_experiment_names",    type=str, default=" ")

args = parser.parse_args()
args.fixed_Ls = bool(args.fixed_Ls)
args.fixed_P0 = bool(args.fixed_P0)

# -----
# Function
def read_experimental_design_parameters(file_name):                                                                                                                       
    """
    file_name   :   str, this file has a specific format and was made by hand
    return
        parameters  :   dict    parameters[exper_name]  -> {"syringe_concentration":float, ...}
    """
    parameters = {}
    with open(file_name, "r") as handle:
        for line in handle:
            if not line.startswith("#"):
                entries = line.split()

                exper_name = entries[0] + "_" + entries[1]
                parameters[exper_name] = {}
                parameters[exper_name]["syringe_concentration"] = float(entries[2])
                parameters[exper_name]["cell_concentration"]    = float(entries[3])
                parameters[exper_name]["number_of_injections"]  = int(entries[5])
                parameters[exper_name]["injection_volume"]      = float(entries[6])
                parameters[exper_name]["spacing"]               = int(entries[7])
                parameters[exper_name]["stir_rate"]             = int(entries[8])
    return parameters

def load_heat_micro_cal(origin_heat_file):
    """
    :param origin_heat_file: str, name of heat file
    :return: 1d ndarray, heats in micro calorie
    """
    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6:
                heats.append(np.float(line.split()[0]))
    return np.array(heats)

def heats_TwoComponentBindingModel(P0, Ls, DeltaG, DeltaH, DeltaH_0, V0, DeltaVn, beta, N):
    """
    Expected heats of injection for two-component binding model.

    ARGUMENTS
    V0 - cell volume (liter)
    DeltaVn - injection volumes (liter)
    P0 - Cell concentration (millimolar)
    Ls - Syringe concentration (millimolar)
    DeltaG - free energy of binding (kcal/mol)
    DeltaH - enthalpy of binding (kcal/mol)
    DeltaH_0 - heat of injection (cal)
    beta - inverse temperature * gas constant (mole / kcal)
    N - number of injections

    Returns
    -------
    expected injection heats (calorie)

    """
    Kd = np.exp(beta * DeltaG) # dissociation constant (M)

    # Compute complex concentrations.
    # Pn[n] is the protein concentration in sample cell after n injections
    # (M)
    Pn = np.zeros([N])
    # Ln[n] is the ligand concentration in sample cell after n injections
    # (M)
    Ln = np.zeros([N])
    # PLn[n] is the complex concentration in sample cell after n injections
    # (M)
    PLn = np.zeros([N])

    dcum = 1.0  # cumulative dilution factor (dimensionless)
    for n in range(N):
        # Instantaneous injection model (perfusion)
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor
        # total quantity of protein in sample cell after n injections (mol)
        P = V0 * P0 * 1.e-3 * dcum
        # total quantity of ligand in sample cell after n injections (mol)
        L = V0 * Ls * 1.e-3 * (1. - dcum)
        # complex concentration (M)
        PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - np.sqrt((P + L + Kd * V0) ** 2 - 4 * P * L) ))
        # free protein concentration in sample cell after n injections (M)
        Pn[n] = P / V0 - PLn[n]
        # free ligand concentration in sample cell after n injections (M)
        Ln[n] = L / V0 - PLn[n]

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = np.zeros([N])

    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0

    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0

    return q_n

def LLH(q_obs, q_model):
    residuals = q_obs - q_model
    N = len(residuals)
    # ssd = np.sum(np.square(residuals))
    # sigma2 = ssd/(N-1)
    variance = np.mean(residuals**2)
    loglikelihood = (np.sum(residuals**2))/(2*variance) + N/2*np.log(variance)
    return loglikelihood

def fitted_function(params, q_actual_cal, V0, DeltaVn, beta, N):
    q_calculated = heats_TwoComponentBindingModel(*params, V0, DeltaVn, beta, N)
    return LLH(q_actual_cal, q_calculated)

def fitted_function_fixed_P0_Ls(params, P0, Ls, q_actual_cal, V0, DeltaVn, beta, N):
    q_calculated = heats_TwoComponentBindingModel(P0, Ls, *params, V0, DeltaVn, beta, N)
    return LLH(q_actual_cal, q_calculated)

def fitted_function_fixed_Ls(injection_volumes, q_actual_cal, Ls, V0, beta, N, 
                             params_guess=None, bounds=None):

    def heats_2C(DeltaVn, P0, DeltaG, DeltaH, DeltaH_0):
        Kd = np.exp(beta * DeltaG) # dissociation constant (M)

        Pn = np.zeros([N])
        Ln = np.zeros([N])
        PLn = np.zeros([N])

        dcum = 1.0  # cumulative dilution factor (dimensionless)
        for n in range(N):
            d = 1.0 - (DeltaVn[n] / V0)
            dcum *= d  # cumulative dilution factor
            P = V0 * P0 * 1.e-3 * dcum
            L = V0 * Ls * 1.e-3 * (1. - dcum)
            PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - np.sqrt((P + L + Kd * V0) ** 2 - 4 * P * L) ))
            Pn[n] = P / V0 - PLn[n]
            Ln[n] = L / V0 - PLn[n]

        q_n = np.zeros([N])
        q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0

        for n in range(1, N):
            d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
            q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0
        return q_n

    fit_f, var_matrix = curve_fit(heats_2C, xdata=injection_volumes, ydata=q_actual_cal,
                                  absolute_sigma=True, p0=params_guess, bounds=bounds)
    perr = np.sqrt(np.diag(var_matrix))
    SSR = np.sum((heats_2C(injection_volumes, *fit_f) - q_actual_cal)**2)
    sigma = np.sqrt(SSR/(N-len(fit_f)))
    return fit_f, perr*sigma

def deltaH0_guesses(q_n_cal):
    heat_interval = (q_n_cal.max() - q_n_cal.min())
    DeltaH_0_min = q_n_cal.min() - heat_interval
    DeltaH_0_max = q_n_cal.max() + heat_interval
    return DeltaH_0_min, DeltaH_0_max

# -----
assert args.fixed_Ls in [0, 1], "The choice for fixed_Ls is only \"0\"(True) or \"1\"(False)."
assert args.fixed_P0 in [0, 1], "The choice for fixed_Po is only \"0\"(True) or \"1\"(False)."
assert len(args.experimental_design_parameters_dir)>0, "Please provide the directory of experimental design parameters."

print("Loading the experimental design parameter file.")
parameters = read_experimental_design_parameters(args.experimental_design_parameters_dir+'/experimental_desgin_parameters.dat')

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

print("Fitting the MLE for ITC data: ")

data_params = {}
data_error = {}
for name in exper_names:
    print("Running the", name)
    itc_file = os.path.join(args.itc_data_dir, name+".itc")
    integ_file = os.path.join(args.heat_data_dir, name + args.heat_file_suffix)

    q_actual_micro_cal = load_heat_micro_cal(integ_file)
    q_actual_cal = q_actual_micro_cal * 1e-6
    
    n_injections = len(q_actual_cal)
    INJ_VOL = parameters[name]["injection_volume"]*1e-6               # in liter
    injection_volumes = [INJ_VOL for _ in range(n_injections)]        # in liter
    SYRINGE_CONCENTR = parameters[name]["syringe_concentration"]      # milli molar
    CELL_CONCENTR = parameters[name]['cell_concentration']            # milli molar

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    if args.fixed_Ls==False and args.fixed_P0==False: 
        params_guess = [CELL_CONCENTR, SYRINGE_CONCENTR, -8, -7, DeltaH_0_min] #P0, Ls, DeltaG, DeltaH, DeltaH_0

        optsol = minimize(fitted_function, x0=params_guess, 
                          args=(q_actual_cal, 1.43*10**(-3), injection_volumes, 
                                BETA, parameters[name]['number_of_injections']),
                          options={'disp': False, 'maxiter': 10000})
        error = np.sqrt(np.diag(np.array(optsol.hess_inv, dtype=np.float64)))
        data_params[name] = optsol.x
        data_error[name] = error
    elif args.fixed_Ls==True and args.fixed_P0==True:
        params_guess = [-8, -7, DeltaH_0_min] #DeltaG, DeltaH, DeltaH_0

        optsol = minimize(fitted_function_fixed_P0_Ls, x0=params_guess, 
                          args=(CELL_CONCENTR, SYRINGE_CONCENTR, 
                                q_actual_cal, 1.43*10**(-3), injection_volumes, 
                                BETA, parameters[name]['number_of_injections']),
                          options={'disp': False, 'maxiter': 10000})
        error = np.sqrt(np.diag(np.array(optsol.hess_inv, dtype=np.float64)))
        data_params[name] = optsol.x
        data_error[name] = error
    elif args.fixed_Ls==True and args.fixed_P0==False:
        params_guess = [CELL_CONCENTR, -8, -7, DeltaH_0_min] #P0, DeltaG, DeltaH, DeltaH_0
        lower = [CELL_CONCENTR*0.1, -40.0, -100, DeltaH_0_min]
        upper = [CELL_CONCENTR*10, 0, 100, DeltaH_0_max]
        fit_f, sigma = fitted_function_fixed_Ls(injection_volumes, q_actual_cal, SYRINGE_CONCENTR, 1.43*10**(-3), 
                                                BETA, parameters[name]['number_of_injections'],
                                                params_guess, (lower, upper))
        data_params[name] = fit_f
        data_error[name] = sigma
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

print("Loading the true experimental design parameter file.")
parameters_true = read_experimental_design_parameters(args.experimental_design_parameters_dir+'/true_experimental_desgin_parameters.dat')

if args.fixed_Ls==False and args.fixed_P0==False:
    propagation_params_list = ['P0_std', 'Ls_std', 'DeltaG_std', 'DeltaH_std']
elif args.fixed_Ls==True and args.fixed_P0==True: 
    propagation_params_list = ['DeltaG_std', 'DeltaH_std']
elif args.fixed_Ls==True and args.fixed_P0==False: 
    propagation_params_list = ['P0_std', 'DeltaG_std', 'DeltaH_std']

print("Parameters adjusted with error propagation: ", propagation_params_list)

propagation_std = {}
for name in exper_names:
    # print("Running the", name)
    Ls_error_percent = abs(parameters_true[name]['syringe_concentration']-1.0)/1.0
    P0_error_percent = abs(parameters_true[name]['cell_concentration']-0.1)/0.1
    error = []
    for i in data_params.columns: 
        if i+'_std' in propagation_params_list:
            old_std_percent = abs(data_error[i+'_std'][name]/data_params[i][name])
            if i is 'P0':
                new_std_percent = np.sqrt(old_std_percent**2+P0_error_percent**2)
            else: 
                new_std_percent = np.sqrt(old_std_percent**2+Ls_error_percent**2)
            error.append(abs(new_std_percent*data_params[i][name]))
        else: 
            error.append(data_error[i+'_std'][name])
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
