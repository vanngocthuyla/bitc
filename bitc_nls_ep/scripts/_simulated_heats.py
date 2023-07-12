"""
run simulated heat
"""
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pymc


def expected_injection_heats(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, N):
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
    expected injection heats -

    """

    Kd = np.exp(beta * DeltaG)   # dissociation constant (M)
    N = N

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
        # TODO: Allow injection volume to vary for each injection.
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor
        # total quantity of protein in sample cell after n injections (mol)
        P = V0 * P0 * 1.e-3 * dcum
        # total quantity of ligand in sample cell after n injections (mol)
        L = V0 * Ls * 1.e-3 * (1. - dcum)
        # complex concentration (M)
        PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - ((P + L + Kd * V0) ** 2 - 4 * P * L) ** 0.5))
        # free protein concentration in sample cell after n injections (M)
        Pn[n] = P / V0 - PLn[n]
        # free ligand concentration in sample cell after n injections (M)
        Ln[n] = L / V0 - PLn[n]

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = np.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH * V0 * PLn[0])*1000 + DeltaH_0
    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000 + DeltaH_0

    return q_n


class _LogNormal(object):
    
    def __init__(self, name, stated_concentration, uncertainty_percent):
        """
        :param name: str
        :param stated_concentration: float, mM
        :param uncertainty: float, 0 < uncertainty < 1
        """
        m = stated_concentration
        uncertainty = stated_concentration * uncertainty_percent
        v = uncertainty ** 2
        model = pymc.Lognormal(name,
                              mu = np.log( m / np.sqrt(1 + ( v / (m**2) ) ) ),
                              tau = 1.0 / np.log( 1 + (v / (m**2)) ),
                              value = m)

        setattr(self, name, model)


def _run_mcmc(model, iter, burn, thin):
    mcmc = pymc.MCMC(model)
    mcmc.sample(iter=iter, burn=burn, thin=thin)
    #pymc.Matplot.plot(mcmc)
    traces = {}
    for s in mcmc.stochastics:
        traces[s.__name__] = s.trace(chain=None)
    return traces


def _sample_lognormal(name, stated_concentration, uncertainty_percent,
                      nsamples, burn=100, thin=100):
    model = _LogNormal(name, stated_concentration, uncertainty_percent)
    iter = nsamples * thin + burn
    traces = _run_mcmc(model, iter, burn, thin)
    return traces[name]


def _cal_2_kcal_per_mol_of_injectant(qn_cal, injection_vol_micro_litter, stated_Ls_mM):
    """
    """
    vol_in_liter = injection_vol_micro_litter * 10. ** (-6)
    concen_in_M = stated_Ls_mM * 10. ** (-3)
    number_of_mol = concen_in_M * vol_in_liter

    qn_kcal = qn_cal * 10**(-3)
    qn_kcal_per_mol_of_injectant = qn_kcal / number_of_mol
    return qn_kcal_per_mol_of_injectant


def _write_heat_to_csv(qn_cal, injection_vol_micro_litter, stated_Ls_mM, out):
    qn_kcal_per_mol_of_injectant = _cal_2_kcal_per_mol_of_injectant(qn_cal, injection_vol_micro_litter, stated_Ls_mM)
    with open(out, "w") as handle:
        for q in qn_kcal_per_mol_of_injectant:
            handle.write("%0.15f, %0.15f\n"%(0., q))
    return None


def _write_exper_design_file(number_of_sim_exper, device, stated_Ls_mM, stated_R0_mM,  
                             number_of_injections, injection_vol_micro_litter, out):
    """
    """
    # unimportant
    test_inj = 0.
    time_interval = 300
    stir_speed = 410

    out_str = "#   0               1                       2                           3                           4"
    out_str += "                   5                   6                   7                       8\n"
    out_str +="# index     Calorimeter     [CBS] (syringe) (mili M)    [CA II] (cell) (mili M)     "
    out_str += "micro L Test injection   No. of injections   micro L injected     Time interval(s)        Stir speed (rpm)\n"

    if not isinstance(stated_Ls_mM, np.ndarray):
        for i in range(number_of_sim_exper):
            out_str += "%5d %12s %23.5f %23.5f %25.5f %20d %25.5f %20d %20d\n" %(i, device, stated_Ls_mM, stated_R0_mM, test_inj,
                                                                                number_of_injections, injection_vol_micro_litter,
                                                                                time_interval, stir_speed)
    else: 
        assert len(stated_Ls_mM)==number_of_sim_exper, "Number of simulated experiments differs from number of syringe concentration"
        for i in range(number_of_sim_exper):
            out_str += "%5d %12s %23.5f %23.5f %25.5f %20d %25.5f %20d %20d\n" %(i, device, stated_Ls_mM[i], stated_R0_mM[i], test_inj,
                                                                                number_of_injections, injection_vol_micro_litter,
                                                                                time_interval, stir_speed)
    open(out, "w").write(out_str)
    return None