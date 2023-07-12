
import argparse
import pickle
import pymc
import pandas as pd

from _uncorr_gaussians_mcmc import _read_nonlinear_fit_results, _UncorrGaussians, _run_mcmc


parser = argparse.ArgumentParser()
parser.add_argument( "--nonlinear_fit_results_file",    type=str, default="origin_dg_dh_in_kcal_per_mole.dat")
parser.add_argument( "--experiment",                    type=str, default=" ")

parser.add_argument( "--iter",                          type=int, default = 310000)
parser.add_argument( "--burn",                          type=int, default = 10000)
parser.add_argument( "--thin",                          type=int, default = 100)

args = parser.parse_args()

VARIABLES = ["DeltaG", "DeltaH", "P0"]
print "variables", VARIABLES
mus, sigs = _read_nonlinear_fit_results(args.nonlinear_fit_results_file, args.experiment, variables=VARIABLES)
print "Inputs:"
for var in mus:
    print "%-10s(std) = %15.5f (%15.5f)"%(var, mus[var], sigs[var])

model = _UncorrGaussians(mus, sigs)
traces = _run_mcmc(model, iter=args.iter, burn=args.burn, thin=args.thin)
pickle.dump(traces, open("traces.pickle", "w"))

print "\n"
print "MCMC estimates:"
for var in traces:
    print "%-10s(std) = %15.5f (%15.5f)"%(var, traces[var].mean(), traces[var].std())
print "DONE"


