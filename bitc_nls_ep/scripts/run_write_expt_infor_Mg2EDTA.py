import sys
import logging
import os
from os.path import basename, splitext
import pickle

import numpy
import glob
import argparse

from _write_experimental_information import input_to_experiment, _pickle_experimental_info

parser = argparse.ArgumentParser()

parser.add_argument( "--out_dir",                       type=str, default="")
parser.add_argument( "--dat_file_dir",                  type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--heat_file_dir",                 type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--heat_file_suffix",              type=str, default =".DAT")

parser.add_argument( "--exclude_experiments",           type=str, default="")
parser.add_argument( "--instrument",                    type=str, default="")
parser.add_argument( "--dummy_itc_file",                action="store_true", default=False)

args = parser.parse_args()

itc_data_files = glob.glob(os.path.join(args.heat_file_dir, "*.itc"))
itc_data_files = [os.path.basename(f) for f in itc_data_files]

exper_names = [f.split(".itc")[0] for f in itc_data_files]
for name in exper_names:
    if not os.path.isfile( os.path.join(args.heat_file_dir, name + args.heat_file_suffix) ):
        print("WARNING: Integrated heat file for " + name + " does not exist")
exper_names = [name for name in exper_names if os.path.isfile( os.path.join(args.heat_file_dir, name + args.heat_file_suffix) ) ]

exclude_experiments = args.exclude_experiments.split()
exper_names = [name for name in exper_names if name not in exclude_experiments]

for name in exper_names: 
    print("Running", name) 
    dat_file = os.path.join(args.dat_file_dir, name+".itc") # .itc file to process
    heat_file = os.path.join(args.heat_file_dir, name+".DAT")

    expt = input_to_experiment(dat_file, heat_file, name, args.instrument, args.dummy_itc_file)
    _pickle_experimental_info(expt, out=os.path.join(os.path.join(args.out_dir,name+'.pickle')))

print('DONE')