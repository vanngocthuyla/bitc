
import argparse
import pickle

import pymc
import pandas as pd

class _UncorrGaussians:
    def __init__(self, mus, sigmas):
        """
        mus, sigmas are dict, var -> value
        """
        for var in mus:
            assert var in sigmas, var + " in mus but not in sigmas"

        for var in sigmas:
            assert var in mus, var + " in sigmas but not in mus"

        variables = mus.keys()
        for var in variables:
            mu = mus[var]
            tau = 1./( sigmas[var]**2 )
            setattr( self, var, pymc.Normal(var, mu=mu, tau=tau) )


def _run_mcmc(model, iter, burn, thin):
    mcmc = pymc.MCMC(model)
    mcmc.sample(iter=iter, burn=burn, thin=thin)
    pymc.Matplot.plot(mcmc)
    traces = {}
    for s in mcmc.stochastics:
        traces[s.__name__] = s.trace(chain=None)
    return traces


def _read_nonlinear_fit_results(file_name, experiment_name, variables=["DeltaG", "DeltaH", "P0"]):
    """
    return mus, sigs which are dic mapping exper to dic mapping variable to float
    """
    results = pd.read_table(file_name, sep='\s+')

    mus  = { variable : results.loc[experiment_name, variable] for variable in variables}
    sigs = { variable : results.loc[experiment_name, variable+"_std"] for variable in variables} 
    return mus, sigs

