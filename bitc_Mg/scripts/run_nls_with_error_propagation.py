"""
fit the MLE model with 3 or 5 parameters for ITC data, then adjust the error of parameters by the error propagation 
"""
import os
import numpy as np
import pickle
from scipy.optimize import minimize, curve_fit
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import glob
import argparse

sys.path.append('/home/vla/python/bayesian-itc')
from bitc.experiments import ExperimentMicroCal, ExperimentMicroCalWithDummyITC, ExperimentYaml
from bitc.instruments import known_instruments, Instrument

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",  type=str, default="")
parser.add_argument( "--itc_data_dir",                type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--heat_data_dir",               type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--exclude_experiments",         type=str, default="")
parser.add_argument( "--script",                      type=str, default = dir)
parser.add_argument( "--heat_file_suffix",            type=str, default =".DAT")
parser.add_argument( "--ordered_experiment_names",    type=str, default=" ")
parser.add_argument( "--fixed_Ls",                    action="store_true", default=False)
parser.add_argument( "--fixed_P0",                    action="store_true", default=False)
parser.add_argument( "--true_value",                  action="store_true", default=False)

args = parser.parse_args()

def load_heat_micro_cal(origin_heat_file):
    """
    :param origin_heat_file: str, name of heat file
    :return: 1d ndarray, heats in micro calorie
    """
    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6 or len(line.split()) == 8:
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
    DeltaH_0 - heat of injection (microcal)
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
    q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0*1e-6

    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0*1e-6

    return q_n

def fitted_function(injection_volumes, q_actual_cal, V0, beta, N, params_guess=None, bounds=None):
    
    def heats_5_parameters(DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0):
        
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
        q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0*1e-6

        for n in range(1, N):
            d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
            q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0*1e-6
        return q_n

    fit_f, var_matrix = curve_fit(heats_5_parameters, xdata=injection_volumes, ydata=q_actual_cal,
                                  sigma=np.ones(len(q_actual_cal))*np.std(q_actual_cal, ddof=1),
                                  absolute_sigma=True, p0=params_guess, bounds=bounds,
                                  ftol=1e-13)
    perr = np.sqrt(np.diag(var_matrix))
    SSR = np.sum((heats_5_parameters(injection_volumes, *fit_f) - q_actual_cal)**2)
    sigma = np.sqrt(SSR/(N-5))
    return fit_f, perr*sigma

def fitted_function_fixed_Ls(injection_volumes, q_actual_cal, Ls, V0, beta, N, 
                             params_guess=None, bounds=None):

    def heats_4_parameters(DeltaVn, P0, DeltaG, DeltaH, DeltaH_0):
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
        q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0*1e-6

        for n in range(1, N):
            d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
            q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0*1e-6
        return q_n

    fit_f, var_matrix = curve_fit(heats_4_parameters, xdata=injection_volumes, ydata=q_actual_cal,
                                  sigma=np.ones(len(q_actual_cal))*np.std(q_actual_cal, ddof=1),
                                  absolute_sigma=True, p0=params_guess, bounds=bounds)
    fit_f, var_matrix = curve_fit(heats_4_parameters, xdata=injection_volumes, ydata=q_actual_cal,
                                  absolute_sigma=True, p0=fit_f, bounds=bounds)
    perr = np.sqrt(np.diag(var_matrix))
    SSR = np.sum((heats_4_parameters(injection_volumes, *fit_f) - q_actual_cal)**2)
    sigma = np.sqrt(SSR/(N-4))
    return fit_f, perr*sigma

def fitted_function_fixed_P0_Ls(injection_volumes, q_actual_cal, P0, Ls, V0, beta, N, 
                                params_guess=None, bounds=None):

    def heats_3_parameters(DeltaVn, DeltaG, DeltaH, DeltaH_0):
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
        q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0*1e-6

        for n in range(1, N):
            d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
            q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0*1e-6
        return q_n

    fit_f, var_matrix = curve_fit(heats_3_parameters, xdata=injection_volumes, ydata=q_actual_cal,
                                  sigma=np.ones(len(q_actual_cal))*np.std(q_actual_cal, ddof=1),
                                  absolute_sigma=True, p0=params_guess, bounds=bounds,
                                  xtol=1e-13)
    perr = np.sqrt(np.diag(var_matrix))
    SSR = np.sum((heats_3_parameters(injection_volumes, *fit_f) - q_actual_cal)**2)
    sigma = np.sqrt(SSR/(N-len(fit_f)))
    return fit_f, perr*sigma

def deltaH0_guesses(q_n_cal):
    heat_interval = (q_n_cal.max() - q_n_cal.min())
    DeltaH_0_min = q_n_cal.min() - heat_interval
    DeltaH_0_max = q_n_cal.max() + heat_interval
    return DeltaH_0_min*1e6, DeltaH_0_max*1e6

def plot_ITC_curve(q_actual_cal, params, V0, injection_volumes, BETA, n_injections, name, OUT_DIR=None):
    total_injected_volume = np.cumsum(injection_volumes)
    plt.plot(total_injected_volume*1e3, 1e6*q_actual_cal, '.')
    plt.plot(total_injected_volume*1e3, 1e6*heats_TwoComponentBindingModel(*params, 1.43*10**(-3), injection_volumes, BETA, n_injections), '-')
    plt.xlabel('Cumulative injection volume (mL)')
    plt.ylabel('Injection heat (ucal)')
    plt.title('MLE_' + name)
    plt.savefig(name)
    plt.tight_layout()
    plt.show(block=False);

# -----
assert args.fixed_Ls in [0, 1], "The choice for fixed_Ls is only \"0\"(True) or \"1\"(False)."
assert args.fixed_P0 in [0, 1], "The choice for fixed_Po is only \"0\"(True) or \"1\"(False)."
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

print("Fitting the MLE for ITC data: ")

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
    injection_volumes = np.array(injection_volumes)*1e-6                              #in liter
    SYRINGE_CONCENTR = experiment['syringe_concentration']['ligand'].magnitude        # milli molar
    CELL_CONCENTR = experiment['cell_concentration']['macromolecule'].magnitude

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    if args.fixed_Ls==False and args.fixed_P0==False: 
        params_guess = [CELL_CONCENTR, SYRINGE_CONCENTR, -8, -7, DeltaH_0_min] #P0, Ls, DeltaG, DeltaH, DeltaH_0  
        lower = [CELL_CONCENTR*0.9, SYRINGE_CONCENTR*0.9, -40.0, -100, DeltaH_0_min]
        upper = [CELL_CONCENTR*1.1, SYRINGE_CONCENTR*1.1, 0, 100, DeltaH_0_max]
        fit_f, sigma = fitted_function(injection_volumes, q_actual_cal, 1.43*10**(-3), 
                                       BETA, n_injections, params_guess, (lower, upper))
        data_params[name] = params = fit_f
        data_error[name] = sigma
        plot_ITC_curve(q_actual_cal, fit_f, 1.43*10**(-3), injection_volumes, BETA, n_injections, str('5_'+name))
    elif args.fixed_Ls==True and args.fixed_P0==False:
        params_guess = [CELL_CONCENTR, -8, -7, DeltaH_0_min] #P0, DeltaG, DeltaH, DeltaH_0  
        lower = [CELL_CONCENTR*0.1, -40.0, -100, DeltaH_0_min]
        upper = [CELL_CONCENTR*10, 0, 100, DeltaH_0_max]
        fit_f, sigma = fitted_function_fixed_Ls(injection_volumes, q_actual_cal, SYRINGE_CONCENTR, 1.43*10**(-3), 
                                                BETA, n_injections, params_guess, (lower, upper))
        data_params[name] = fit_f
        data_error[name] = sigma
        params = [fit_f[0], SYRINGE_CONCENTR, *fit_f[1:]]
        plot_ITC_curve(q_actual_cal, params, 1.43*10**(-3), injection_volumes, BETA, n_injections, str('4_'+name))
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
        plot_ITC_curve(q_actual_cal, params, 1.43*10**(-3), injection_volumes, BETA, n_injections, str('3_'+name))
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

if args.true_value==True:
    print("True parameters are estimated by the median of parameters from repeated experiments.")
    true_Ls = np.concatenate((np.repeat(MLE_data[0:5].median()['Ls'], 5), 
                              np.repeat(MLE_data[5:].median()['Ls'], 9)), 
                             axis=0)
    true_P0 = np.concatenate((np.repeat(MLE_data[0:5].median()['P0'], 5), 
                              np.repeat(MLE_data[5:].median()['P0'], 9)), 
                             axis=0)

if args.fixed_Ls==False and args.fixed_P0==False:
    propagation_params_list = ['P0_std', 'Ls_std', 'DeltaG_std', 'DeltaH_std']
elif args.fixed_Ls==True and args.fixed_P0==True: 
    propagation_params_list = ['DeltaG_std', 'DeltaH_std']
elif args.fixed_Ls==True and args.fixed_P0==False: 
    propagation_params_list = ['P0_std', 'DeltaG_std', 'DeltaH_std']

if len(propagation_params_list)>0:

    print("Parameters adjusted with error propagation: ", propagation_params_list)

    propagation_std = {}
    for i in range(len(exper_names)):
        name = exper_names[i]
        print("Running the", name)

        experiment_design_file = os.path.join(args.experimental_design_parameters_dir, name+'.pickle')
        experiment = pickle.load(open(experiment_design_file, "rb") )
        SYRINGE_CONCENTR = experiment['syringe_concentration']['ligand'].magnitude        # milli molar
        CELL_CONCENTR = experiment['cell_concentration']['macromolecule'].magnitude

        if args.true_value == True:
            Ls_error_percent = abs(true_Ls[i]-SYRINGE_CONCENTR)/SYRINGE_CONCENTR
            P0_error_percent = abs(true_P0[i]-CELL_CONCENTR)/CELL_CONCENTR
        else: 
            Ls_error_percent = 0.1
            P0_error_percent = 0.1
        
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
    print('MLE fitted with 5 parameters adjusted. No error propagation')
    # pickle.dump(propagation_data.to_pickle, open(os.path.join('Propagation_5_parameters.pickle'), "wb"))
    # propagation_data.to_csv('Propagation_5_parameters.csv')
elif args.fixed_Ls==True and args.fixed_P0==True: 
    print('MLE fitted with 3 parameters adjusted by error propagation')
    pickle.dump(propagation_data.to_pickle, open(os.path.join('Propagation_3_parameters.pickle'), "wb"))
    propagation_data.to_csv('Propagation_3_parameters.csv')
elif args.fixed_Ls==True and args.fixed_P0==False: 
    print('MLE fitted with 4 parameters adjusted by error propagation')
    pickle.dump(propagation_data.to_pickle, open(os.path.join('Propagation_4_parameters.pickle'), "wb"))
    propagation_data.to_csv('Propagation_4_parameters.csv')

print("DONE")
