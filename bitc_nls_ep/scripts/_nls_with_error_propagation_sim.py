"""
fit the MLE model with 3 or 5 parameters for ITC data, then adjust the error of parameters by the error propagation 
"""
import os
import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt


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