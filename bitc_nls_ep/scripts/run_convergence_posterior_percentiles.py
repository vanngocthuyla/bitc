"""
calculate convergence of percentiles for parameters in traces
"""

import os
import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",           type=str,           default="")
parser.add_argument("--out_dir",            type=str,           default="")

parser.add_argument("--file_name",          type=str,           default="")
parser.add_argument("--experiments",        type=str,           default="")
parser.add_argument("--percentiles",        type=str,           default="5 25 50 75 95")
parser.add_argument("--vars",               type=str,           default="DeltaG DeltaH DeltaH_0 P0 Ls log_sigma")

parser.add_argument("--sample_proportions", type=str,           default="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
parser.add_argument("--repeats",            type=int,           default=100)
parser.add_argument("--random_state",       type=int,           default=0)

args = parser.parse_args()

def percentiles(x, q, nsamples, repeats):
    perce = []
    for _ in range(repeats):
        rnd_x = np.random.choice(x, size=nsamples, replace=True)
        p = np.percentile(rnd_x, q)
        perce.append(p)

    perce = np.array(perce)
    p_mean = perce.mean(axis=0)
    p_err = perce.std(axis=0)

    return p_mean, p_err


def print_percentiles(p_mean, p_err):
    if isinstance(p_mean, float) and isinstance(p_err, float):
        return "%12.5f%12.5f" % (p_mean, p_err)
    else:
        p_str = "".join(["%12.5f%12.5f" % (p_m, p_e) for p_m, p_e in zip(p_mean, p_err)])
        return p_str


np.random.seed(args.random_state)

experiments = args.experiments.split()
print("experiments:", experiments)

qs = [float(s) for s in args.percentiles.split()]
qs_str = "".join(["%10.1f-th %10.1f-error " % (q, q) for q in qs])
print("qs:", qs_str)

vars = args.vars.split()
print("vars:", vars)

sample_proportions = [float(s) for s in args.sample_proportions.split()]
print("sample_proportions:", sample_proportions)

os.chdir(args.out_dir)

for exper in experiments:
    print("Calculating CIs for " + exper)

    if len(args.file_name)==0:
        trace_file = os.path.join(args.data_dir, exper, 'traces.pickle')
    else:
        trace_file = os.path.join(args.data_dir, args.file_name, exper, 'traces.pickle')

    if os.path.isfile(trace_file): 
        print("Loading " + trace_file)
        sample = pickle.load(open(trace_file, 'rb'))

        all_vars = sample.keys()
        for v in vars:
            if v not in all_vars:
                raise KeyError(v + " not a valid var name.")

        for var in vars:

            x = sample[var]
            nsamples = len(x)

            if var == 'DeltaH_0':
                x = x*1E6

            out_file_handle = open(exper + "_" + var + ".dat", "w")
            out_file_handle.write("proportion   nsamples" + qs_str + "\n")

            for samp_pro in sample_proportions:
                nsamp_pro = int(nsamples * samp_pro)
                p_mean, p_err = percentiles(x, qs, nsamp_pro, args.repeats)

                out_str = "%10.5f%10d" % (samp_pro, nsamp_pro) + print_percentiles(p_mean, p_err) + "\n"

                out_file_handle.write(out_str)

            out_file_handle.close()
    else:
        print(trace_file, "doesn't exist.")

print("DONE")
