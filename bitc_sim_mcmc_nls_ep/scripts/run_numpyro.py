import os
import numpy as np
import pickle
import arviz as az
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro

import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS, init_to_value
from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

print(f'Using numpyro {numpyro.__version__}')
print(f'Using jax {jax.__version__}')

import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",  type=str, default="")
parser.add_argument( "--itc_data_dir",                  type=str, default="")
parser.add_argument( "--heat_data_dir",                 type=str, default="")
parser.add_argument( "--experiments",                   type=str, default="")
parser.add_argument( "--heat_file_suffix",              type=str, default=".DAT")

parser.add_argument( "--dc",                            type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument( "--ds",                            type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument( "--dummy_itc_file",                action="store_true", default=False)

parser.add_argument( "--uniform_cell_concentration",    action="store_true", default=False)
parser.add_argument( "--uniform_syringe_concentration", action="store_true", default=False)
parser.add_argument( "--concentration_range_factor",    type=float, default=10.)

parser.add_argument( "--niters",                        type=int, default=100000)
parser.add_argument( "--nburn",                         type=int, default=10000)
parser.add_argument( "--nthin",                         type=int, default=10)
parser.add_argument( "--nchain",                        type=int, default=4)
parser.add_argument( "--verbosity",                     type=str, default="-vvv")

args = parser.parse_args()

TRACES_FILE = "traces.pickle"

# Functions #
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
    return jnp.array(heats)

def logsigma_guesses(q_n_cal):
    # log_sigma_guess = np.log(q_n_cal[-4:].std())
    log_sigma_guess = jnp.log(q_n_cal[-4:].std())
    log_sigma_min = log_sigma_guess - 10
    log_sigma_max = log_sigma_guess + 5
    return log_sigma_min, log_sigma_max

def deltaH0_guesses(q_n_cal):
    heat_interval = (q_n_cal.max() - q_n_cal.min())
    DeltaH_0_min = q_n_cal.min() - heat_interval
    DeltaH_0_max = q_n_cal.max() + heat_interval
    return DeltaH_0_min, DeltaH_0_max

def lognormal_prior(name, stated_value, uncertainty):
    """
    Define a pymc3 prior for a deimensionless quantity
    :param name: str
    :param stated_value: float
    :uncertainty: float
    :rerurn: numpyro.Lognormal
    """
    m = stated_value
    v = uncertainty ** 2
    # return pymc3.Lognormal(name,
    #                        mu=tt.log(m / tt.sqrt(1 + (v / (m ** 2)))),
    #                        tau=1.0 / tt.log(1 + (v / (m ** 2))),
    #                        testval=m)
    name = numpyro.sample(name, dist.LogNormal(loc=jnp.log(m / jnp.sqrt(1 + (v / (m ** 2)))), 
                                               scale=jnp.sqrt(jnp.log(1 + v / (m**2) )) ))

    return name

def uniform_prior(name, lower, upper):
    """
    :param name: str
    :param lower: float
    :param upper: float
    :return: pymc3.Uniform
    """
    # return pymc3.Uniform(name, lower=lower, upper=upper)
    name = numpyro.sample(name, dist.Uniform(low=lower, high=upper))
    return name

def heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, N):
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
    Kd = jnp.exp(beta * DeltaG)   # dissociation constant (M)

    # Compute complex concentrations.
    # Pn[n] is the protein concentration in sample cell after n injections
    # (M)
    Pn = jnp.zeros([N], dtype=jnp.float64)
    # Ln[n] is the ligand concentration in sample cell after n injections
    # (M)
    Ln = jnp.zeros([N], dtype=jnp.float64)
    # PLn[n] is the complex concentration in sample cell after n injections
    # (M)
    PLn = jnp.zeros([N], dtype=jnp.float64)

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
        #PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - jnp.sqrt((P + L + Kd * V0) ** 2 - 4 * P * L) ))
        PLn = jax.ops.index_add(PLn, jax.ops.index[n], 0.5 / V0 * ((P + L + Kd * V0) - jnp.sqrt( (P + L + Kd * V0) ** 2 - 4 * P * L) ))
        # PLn = PLn.at[n].add(0.5 / V0 * ((P + L + Kd * V0) - jnp.sqrt( (P + L + Kd * V0) ** 2 - 4 * P * L) ))
        
        # free protein concentration in sample cell after n injections (M)
        #Pn[n] = P / V0 - PLn[n]
        Pn = jax.ops.index_add(Pn, jax.ops.index[n], P/V0 - PLn[n])
        # Pn = Pn.at[n].add(P / V0 - PLn[n])
        # free ligand concentration in sample cell after n injections (M)
        #Ln[n] = L / V0 - PLn[n]
        Ln = jax.ops.index_add(Ln, jax.ops.index[n], L/V0 - PLn[n])
        # Ln = Ln.at[n].add(L / V0 - PLn[n])

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = jnp.zeros([N], dtype=jnp.float64)
    # Instantaneous injection model (perfusion)
    # first injection
    #q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0
    q_n = jax.ops.index_add(q_n, jax.ops.index[0], (DeltaH * V0 * PLn[0])*1000. + DeltaH_0)
    # q_n = q_n.at[0].add((DeltaH * V0 * PLn[0])*1000. + DeltaH_0)

    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        #q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0
        q_n = jax.ops.index_add(q_n, jax.ops.index[n], (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0)
        # q_n = q_n.at[n].add((DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0)

    return jnp.array(q_n)

def param_2C(q_actual_cal, injection_volumes, cell_concentration, syringe_concentration,
          cell_volume, temperature, dcell, dsyringe,
          uniform_P0=False, P0_min=None, P0_max=None, 
          uniform_Ls=False, Ls_min=None, Ls_max=None):
    """
    :param cell_concentration: concentration of the sample cell in milli molar, float
    :param syringe_concentration: concentration of the syringe in milli molar, float
    :param cell_volume: volume of sample cell in liter, float #check the instrutment 
    :param temperature: temprature in kelvin, float
    :param dcell: relative uncertainty in cell concentration, float
    :param dsyringe: relative uncertainty in syringe concentration, float
    :param uniform_P0: if True, use uniform prior for cell concentration, bool
    :param P0_min: only use if uniform_P0 is True, float
    :param P0_max: only use if uniform_P0 is True, float
    :param uniform_Ls: if True, use uniform prior for syringe concentration, bool
    :param Ls_min: only use if uniform_Ls is True, float
    :param Ls_max: only use if uniform_Ls is True, float
    """
    if uniform_P0 and (P0_min is None or P0_max is None):
        raise ValueError("If uniform_P0 is True, both P0_min and P0_max must be provided")
    
    if uniform_Ls and (Ls_min is None or Ls_max is None):
        raise ValueError("If uniform_Ls is True, both Ls_min and Ls_max must be provided")
    
    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = cell_concentration
    # print("Stated P0", stated_P0)
    uncertainty_P0 = dcell * stated_P0

    stated_Ls = syringe_concentration
    # print("Stated Ls", stated_Ls)
    uncertainty_Ls = dsyringe * stated_Ls

    # prior for receptor concentration
    if uniform_P0:
        print("Uniform prior for P0")
        P0 = uniform_prior("P0", lower=P0_min, upper=P0_max)
    else:
        # print("LogNormal prior for P0")
        P0 = lognormal_prior("P0", stated_value=stated_P0, uncertainty=uncertainty_P0)

    # prior for ligand concentration
    if uniform_Ls:
        print("Uniform prior for Ls")
        Ls = uniform_prior("Ls", lower=Ls_min, upper=Ls_max)
    else:
        # print("LogNormal prior for Ls")
        Ls = lognormal_prior("Ls", stated_value=stated_Ls, uncertainty=uncertainty_Ls)
  
    # prior for DeltaG
    DeltaG = uniform_prior("DeltaG", lower=-40., upper=4.)

    # prior for DeltaH
    DeltaH = uniform_prior("DeltaH", lower=-100., upper=100.)

    # prior for DeltaH_0
    DeltaH_0 = uniform_prior("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max)

    # prior for log_sigma
    log_sigma = uniform_prior("log_sigma", lower=log_sigma_min, upper=log_sigma_max)

    return P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma

def make_TwoComponentBindingModel(q_actual_cal, injection_volumes, cell_concentration, 
                                  syringe_concentration, dcell, dsyringe, 
                                  cell_volume=0.001434, temperature=298.15,
                                  uniform_P0=False, P0_min=None, P0_max=None, 
                                  uniform_Ls=False, Ls_min=None, Ls_max=None):
    """
    to create a model
    :param q_actual_cal: observed heats in calorie, array-like
    :param injection_volumes: injection volumes in liter, array-like
    :param cell_concentration: concentration of the sample cell in milli molar, float
    :param syringe_concentration: concentration of the syringe in milli molar, float
    :param cell_volume: volume of sample cell in liter, float #check the instrutment 
    :param temperature: temprature in kelvin, float
    :param dcell: relative uncertainty in cell concentration, float
    :param dsyringe: relative uncertainty in syringe concentration, float
    :param uniform_P0: if True, use uniform prior for cell concentration, bool
    :param P0_min: only use if uniform_P0 is True, float
    :param P0_max: only use if uniform_P0 is True, float
    :param uniform_Ls: if True, use uniform prior for syringe concentration, bool
    :param Ls_min: only use if uniform_Ls is True, float
    :param Ls_max: only use if uniform_Ls is True, float
    
    :return: an instance of numpyro.model
    """
    assert len(q_actual_cal) == len(injection_volumes), "q_actual_cal and injection_volumes must have the same len."
    
    V0 = cell_volume
    DeltaVn = injection_volumes
    beta = 1 / KB / temperature
    n_injections = len(q_actual_cal)
    
    P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma = param_2C(q_actual_cal, injection_volumes, 
                                                           cell_concentration, syringe_concentration, 
                                                           cell_volume, temperature, dcell, dsyringe,
                                                           uniform_P0, P0_min, P0_max,
                                                           uniform_Ls, Ls_min, Ls_max)

    sigma_cal = jnp.exp(log_sigma)
    q_model_cal = heats_TwoComponentBindingModel(cell_volume, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, n_injections)

    numpyro.sample('q_obs', dist.Normal(loc=q_model_cal, scale=sigma_cal), obs=q_actual_cal)


def Bayesian_multiple_expt_fitting(rng_key, q_actual_cal, injection_volumes, CELL_CONCENTR,
                                   SYRINGE_CONCENTR, dcell, dsyringe, niters, nburn, nchain, nthin, 
                                   name=None, OUT_DIR=''):
    kernel = NUTS(make_TwoComponentBindingModel)
    nuts = MCMC(kernel, num_warmup=nburn, num_samples=niters, num_chains=nchain, 
                thinning=nthin, progress_bar=False)
    nuts.run(rng_key, q_actual_cal, injection_volumes, CELL_CONCENTR, SYRINGE_CONCENTR, dcell, dsyringe)
    trace = nuts.get_samples(group_by_chain=False)
    pickle.dump(trace, open(os.path.join(OUT_DIR, TRACES_FILE), "wb"))
    nuts.print_summary()

    data = az.convert_to_inference_data(nuts.get_samples(group_by_chain=True))
    az.plot_trace(data)
    plt.tight_layout();
    plt.savefig(os.path.join(OUT_DIR,'trace_plot.pdf'))

# Experiments to run
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

# Model #
assert len(args.experimental_design_parameters_dir)>0, "Please provide the directory of experimental design parameters."
parameters = read_experimental_design_parameters(args.experimental_design_parameters_dir+'/experimental_desgin_parameters.dat')

print("Will run these", exper_names)

rng_key = random.split(random.PRNGKey(1), 4)
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
                                       name, name)

print("DONE")
