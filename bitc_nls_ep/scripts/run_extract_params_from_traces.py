import pandas as pd
import warnings
import numpy as np
import sys
import os
import glob
import shutil
import argparse

import pickle
import arviz as az
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",                              type=str,           default="")
parser.add_argument( "--out_dir",                               type=str,           default=None)

parser.add_argument( "--parameters",                            type=str,           default="P0 Ls DeltaG1 DeltaDeltaG DeltaH1 DeltaH2 DeltaH_0 rho")
parser.add_argument( "--statistic",                             type=str,           default="mean")

args = parser.parse_args()


def extract_parameters(DATA_DIR, DATA_FILE='traces.pickle', params=["theta1", "theta2", "theta3", "theta4"],
					   statistic='mean'): 
    """
    params DATA_DIR: directory of mcmc traces
    params DATA_FILE: name of mcmc file
    params params: parameters needed to be extraced from file
    params statistic: mean, median, or standard deviation

    Return mean/median of parameters in traces file
    """
    assert statistic in ['mean', 'median', 'std'], "Unknown statistic."

    traces = pickle.load(open(os.path.join(DATA_DIR, DATA_FILE), "rb"))
    if statistic == 'mean':
        output = [np.mean(traces[x]) for x in params]
    elif statistic == 'median':
    	output = [np.median(traces[x]) for x in params]
    elif statistic == 'std':
    	output = [np.std(traces[x]) for x in params]
    return np.array(output)

def extract_parameters_multiple_traces(DATA_DIR, params=["theta1", "theta2", "theta3", "theta4"], statistic='mean', OUT_DIR=None):
    """
    params DATA_DIR: directory of mcmc traces
    params DATA_FILE: name of mcmc file
    params params: parameters needed to be extraced from file
    params statistic: mean, median, or standard deviation

    Return mean/median of parameters in traces file
    """
    assert statistic in ['mean', 'median', 'std'], "Unknown statistic."
    data_files = glob.glob(os.path.join(DATA_DIR, "*.pickle"))
    data_files = [os.path.basename(f) for f in data_files]

    params_table = {}
    for file in data_files:
        params_table[file] = extract_parameters(DATA_DIR, file, params, statistic)
    output = pd.DataFrame.from_dict(params_table).T
    output.columns = params

    if OUT_DIR is not None:
        output_name = os.path.commonprefix(data_files)[:-1]
        output.to_csv(os.path.join(OUT_DIR, output_name+'.csv'))
    return output

assert args.statistic in ['mean', 'median', 'std'], "Unknown statistic."

print("Data directory:", args.data_dir)
print("Output directory:", args.out_dir)
print("Statistic:", args.statistic)

output = extract_parameters_multiple_traces(args.data_dir, args.parameters.split(), args.statistic, args.out_dir)
print(output)

print("DONE")
