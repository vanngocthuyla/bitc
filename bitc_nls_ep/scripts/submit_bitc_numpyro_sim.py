import os
import glob
import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument( "--experimental_design_parameters_dir",  type=str, default="")
parser.add_argument( "--itc_data_dir",                  type=str, default="1.itc_origin_heat_files")
parser.add_argument( "--heat_data_dir",                 type=str, default="1.itc_origin_heat_files")

parser.add_argument( "--exclude_experiments",           type=str, default="")
parser.add_argument( "--split_by",                      type=int, default=10)

parser.add_argument( "--script",                        type=str, default="/Users/seneysophie/Work/Python/Local/bitc_sim_mcmc_nls_ep/scripts/run_numpyro_sim.py")
parser.add_argument( "--heat_file_suffix",              type=str, default=".DAT")

parser.add_argument( "--dc",                            type=float, default=0.1)      # cell concentration relative uncertainty
parser.add_argument( "--ds",                            type=float, default=0.1)      # syringe concentration relative uncertainty

parser.add_argument( "--dummy_itc_file",                action="store_true", default=False)

parser.add_argument( "--uniform_cell_concentration",    action="store_true", default=False)
parser.add_argument( "--uniform_syringe_concentration", action="store_true", default=False)
parser.add_argument( "--concentration_range_factor",    type=float, default=10.)

parser.add_argument( "--niters",                        type=int, default=10000)
parser.add_argument( "--nburn",                         type=int, default=1000)
parser.add_argument( "--nthin",                         type=int, default=2)
parser.add_argument( "--nchain",                        type=int, default=4)

args = parser.parse_args()
out_dir = os.getcwd()

itc_data_files = glob.glob(os.path.join(args.itc_data_dir, "*.itc"))
itc_data_files = [os.path.basename(f) for f in itc_data_files]

exper_names = [f.split(".itc")[0] for f in itc_data_files]
for name in exper_names:
    if not os.path.isfile( os.path.join(args.heat_data_dir, name + args.heat_file_suffix) ):
        print("WARNING: Integrated heat file for " + name + " does not exist")
exper_names = [name for name in exper_names if os.path.isfile( os.path.join(args.heat_data_dir, name + args.heat_file_suffix) ) ]

exclude_experiments = args.exclude_experiments.split()
exper_names = [name for name in exper_names if name not in exclude_experiments]

assert (len(exper_names)%args.split_by)==0, "The split_by number need to be the factor of number of experiments."
if not args.split_by is 0:
    exper_names_list = np.split(np.array(exper_names), args.split_by)
else: 
    exper_names_list = np.array(exper_names)

for i in range(len(exper_names_list)):
    exper_list = str(exper_names_list[i]).replace("[","").replace("]","").replace("'","")
    
    qsub_file = os.path.join(out_dir, ''.join([str(i+1), "_numpyro.job"]))
    log_file  = os.path.join(out_dir, ''.join([str(i+1), "_numpyro.log"]))

    if args.dummy_itc_file:
        dummy_itc_file = ''' --dummy_itc_file '''
    else:
        dummy_itc_file = ''' '''

    if args.uniform_cell_concentration:
        uniform_cell_concentration = ''' --uniform_cell_concentration '''
    else:
        uniform_cell_concentration = ''' '''

    if args.uniform_syringe_concentration:
        uniform_syringe_concentration = ''' --uniform_syringe_concentration '''
    else:
        uniform_syringe_concentration = ''' '''

    qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s '''%log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,mem=2048mb,walltime=72:00:00

module load miniconda/3
source activate bitc
cd ''' + out_dir + '''\n''' + \
    '''date\n''' + \
    '''python ''' + args.script + \
    ''' --itc_data_dir ''' + args.itc_data_dir + \
    ''' --heat_data_dir ''' + args.heat_data_dir + \
    ''' --experimental_design_parameters_dir ''' + args.experimental_design_parameters_dir + \
    ''' --experiments "%s" '''%exper_list + \
    ''' --dc %f '''%args.dc + \
    ''' --ds %f '''%args.ds + \
    dummy_itc_file + uniform_cell_concentration + uniform_syringe_concentration + \
    ''' --concentration_range_factor %f '''%args.concentration_range_factor + \
    ''' --niters %d '''%args.niters + \
    ''' --nburn %d '''%args.nburn + \
    ''' --nthin %d '''%args.nthin + \
    ''' --nchain %d '''%args.nchain + \
    '''\ndate \n'''

    print("Submitting " + qsub_file)
    open(qsub_file, "w").write(qsub_script)
    os.system("qsub %s"%qsub_file)

