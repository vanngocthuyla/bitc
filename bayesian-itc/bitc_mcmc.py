#!/usr/bin/python
"""
copied from https://github.com/nguyentrunghai/bayesian-itc
"""

import logging
import os
from os.path import basename, splitext
import pickle

import numpy
import pymc
import traceback
from bitc.experiments import ExperimentMicroCal, ExperimentMicroCalWithDummyITC, ExperimentYaml
from bitc.instruments import known_instruments, Instrument
from bitc.models import TwoComponentBindingModel, CompetitiveBindingModel
from bitc.models import RacemicMixtureBindingModel, EnantiomerBindingModel
from bitc.parser import bitc_mcmc_parser
from bitc.units import Quantity

try:
    import seaborn
except ImportError:
    pass


def compute_normal_statistics(x_t):

    # Compute mean.
    x = x_t.mean()

    # Compute stddev.
    dx = x_t.std()

    # Compute 95% confidence interval.
    ci = 0.95
    N = x_t.size
    x_sorted = numpy.sort(x_t)
    low_index = round((0.5-ci/2.0)*N)
    high_index = round((0.5+ci/2.0)*N)
    xlow = x_sorted[low_index]
    xhigh = x_sorted[high_index]

    return [x, dx, xlow, xhigh]


def plot_two_component_model_results(model):
    if sum(model.experiment.cell_concentration.values()) > Quantity('0.0 molar'):
        pymc.Matplot.plot(model.mcmc.trace('P0')[:], '%s-P0' % model.experiment.name)
    if sum(model.experiment.syringe_concentration.values()) > Quantity('0.0 molar'):
        pymc.Matplot.plot(model.mcmc.trace('Ls')[:], '%s-Ls' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaG')[:], '%s-DeltaG' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaH')[:], '%s-DeltaH' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaH_0')[:], '%s-DeltaH_0' % model.experiment.name)
    pymc.Matplot.plot(numpy.exp(model.mcmc.trace('log_sigma')[:]), '%s-sigma' % model.experiment.name)
    #  TODO: Plot fits to enthalpogram.
    # experiment.plot(model=model, filename='%s-enthalpogram.png' %  experiment_name) # todo fix this
    # Compute confidence intervals in thermodynamic parameters.
    outfile = open('%s.confidence-intervals.out' % model.experiment.name, 'a+')
    outfile.write('%s\n' % model.experiment.name)
    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaG')[:])
    outfile.write('DG:     %8.2f +- %8.2f kcal/mol     [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))
    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH')[:])
    outfile.write('DH:     %8.2f +- %8.2f kcal/mol     [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))
    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH_0')[:])
    outfile.write('DH0:    %8.2f +- %8.2f ucal         [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))
    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('Ls')[:])
    outfile.write('Ls:     %8.2f +- %8.2f uM           [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))
    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('P0')[:])
    outfile.write('P0:     %8.2f +- %8.2f uM           [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))
    [x, dx, xlow, xhigh] = compute_normal_statistics(numpy.exp(model.mcmc.trace('log_sigma')[:]))
    outfile.write('sigma:  %8.5f +- %8.5f ucal/s^(1/2) [%8.5f, %8.5f] \n' % (x, dx, xlow, xhigh))
    outfile.write('\n')
    outfile.close()



def plot_racemicmixture_model_results(model):
    pymc.Matplot.plot(model.mcmc.trace('P0')[:], '%s-P0' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('Ls')[:], '%s-Ls' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('DeltaH1')[:], '%s-DeltaH1' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaH2')[:], '%s-DeltaH2' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('DeltaG1')[:], '%s-DeltaG1' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaDeltaG')[:], '%s-DeltaDeltaG' % model.experiment.name)
    DeltaG2 = model.mcmc.trace('DeltaG1')[:] + model.mcmc.trace('DeltaDeltaG')[:]
    pymc.Matplot.plot(DeltaG2, '%s-DeltaG2' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('DeltaH_0')[:], '%s-DeltaH_0' % model.experiment.name)
    pymc.Matplot.plot(numpy.exp(model.mcmc.trace('log_sigma')[:]), '%s-sigma' % model.experiment.name)

    #  TODO: Plot fits to enthalpogram.
    # experiment.plot(model=model, filename='%s-enthalpogram.png' %  experiment_name) # todo fix this
    # Compute confidence intervals in thermodynamic parameters.
    outfile = open('%s.confidence-intervals.out' % model.experiment.name, 'a+')
    outfile.write('%s\n' % model.experiment.name)

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('P0')[:])
    outfile.write('P0:     %8.2f +- %8.2f uM     [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('Ls')[:])
    outfile.write('Ls:     %8.2f +- %8.2f uM     [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH1')[:])
    outfile.write('DH1:     %8.2f +- %8.2f kcal/mol   [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH2')[:])
    outfile.write('DH2:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaG1')[:])
    outfile.write('DG1:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaDeltaG')[:])
    outfile.write('DDG:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(DeltaG2)
    outfile.write('DG2:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH_0')[:])
    outfile.write('DH0:    %8.2f +- %8.2f ucal         [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(numpy.exp(model.mcmc.trace('log_sigma')[:]))
    outfile.write('sigma:  %8.5f +- %8.5f ucal/s^(1/2) [%8.5f, %8.5f] \n' % (x, dx, xlow, xhigh))
    outfile.write('\n')
    outfile.close()



def plot_enantiomer_model_results(model):
    pymc.Matplot.plot(model.mcmc.trace('P0')[:], '%s-P0' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('Ls')[:], '%s-Ls' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('rho')[:], '%s-rho' % model.experiment.name)

    Ls1 = model.mcmc.trace('Ls')[:] * model.mcmc.trace('rho')[:]
    pymc.Matplot.plot(Ls1, '%s-Ls1' % model.experiment.name)

    Ls2 = model.mcmc.trace('Ls')[:] * (1. - model.mcmc.trace('rho')[:])
    pymc.Matplot.plot(Ls2, '%s-Ls2' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('DeltaH1')[:], '%s-DeltaH1' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaH2')[:], '%s-DeltaH2' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('DeltaG1')[:], '%s-DeltaG1' % model.experiment.name)
    pymc.Matplot.plot(model.mcmc.trace('DeltaDeltaG')[:], '%s-DeltaDeltaG' % model.experiment.name)
    DeltaG2 = model.mcmc.trace('DeltaG1')[:] + model.mcmc.trace('DeltaDeltaG')[:]
    pymc.Matplot.plot(DeltaG2, '%s-DeltaG2' % model.experiment.name)

    pymc.Matplot.plot(model.mcmc.trace('DeltaH_0')[:], '%s-DeltaH_0' % model.experiment.name)
    pymc.Matplot.plot(numpy.exp(model.mcmc.trace('log_sigma')[:]), '%s-sigma' % model.experiment.name)

    #  TODO: Plot fits to enthalpogram.
    # experiment.plot(model=model, filename='%s-enthalpogram.png' %  experiment_name) # todo fix this
    # Compute confidence intervals in thermodynamic parameters.
    outfile = open('%s.confidence-intervals.out' % model.experiment.name, 'a+')
    outfile.write('%s\n' % model.experiment.name)

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('P0')[:])
    outfile.write('P0:     %8.2f +- %8.2f uM     [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('Ls')[:])
    outfile.write('Ls:     %8.2f +- %8.2f uM     [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('rho')[:])
    outfile.write('rho:     %8.2f +- %8.2f       [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(Ls1)
    outfile.write('Ls1:     %8.2f +- %8.2f uM    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(Ls2)
    outfile.write('Ls2:     %8.2f +- %8.2f uM    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH1')[:])
    outfile.write('DH1:     %8.2f +- %8.2f kcal/mol   [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH2')[:])
    outfile.write('DH2:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaG1')[:])
    outfile.write('DG1:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaDeltaG')[:])
    outfile.write('DDG:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(DeltaG2)
    outfile.write('DG2:     %8.2f +- %8.2f kcal/mol    [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(model.mcmc.trace('DeltaH_0')[:])
    outfile.write('DH0:    %8.2f +- %8.2f ucal         [%8.2f, %8.2f] \n' % (x, dx, xlow, xhigh))

    [x, dx, xlow, xhigh] = compute_normal_statistics(numpy.exp(model.mcmc.trace('log_sigma')[:]))
    outfile.write('sigma:  %8.5f +- %8.5f ucal/s^(1/2) [%8.5f, %8.5f] \n' % (x, dx, xlow, xhigh))
    outfile.write('\n')
    outfile.close()


user_input = bitc_mcmc_parser()

# Process the arguments
working_directory = user_input['--workdir']

if not os.path.exists(working_directory):
    os.mkdir(working_directory)

os.chdir(working_directory)

# Set the logfile
if user_input['--log']:
    logfile = '%(--log)s' % user_input
else:
    logfile = None

# Level of verbosity in log
if user_input['-v'] == 3:
    loglevel = logging.DEBUG
elif user_input['-v'] == 2:
    loglevel = logging.INFO
elif user_input['-v'] == 1:
    loglevel = logging.WARNING
else:
    loglevel = logging.ERROR

# Set up the logger
logging.basicConfig(format='%(levelname)s::%(module)s:L%(lineno)s\n%(message)s', level=loglevel, filename=logfile)

logging.debug("Received this input from the user:")
logging.debug(str(user_input))

# Check if model has all necessary info
if user_input['competitive']:
    if not user_input['--receptor']:
        raise ValueError('Need to specify a receptor for Competitive model')

# Files for processing
datafiles = user_input['<datafile>']  # .itc file to process
datafile_basename, datafile_extension = splitext(basename(datafiles[0]))

if not user_input['--name']:
    # Name of the experiment, and output files
    experiment_name = datafile_basename
else:
    experiment_name = user_input['--name']

# MCMC settings
nfit = user_input['--nfit']      # number of iterations for maximum a posteriori fit
niters = user_input['--niters']  # number of iterations
nburn = user_input['--nburn']    # number of burn-in iterations
nthin = user_input['--nthin']    # thinning period

if user_input['--instrument']:
    # Use an instrument from the brochure
    instrument = known_instruments[user_input['--instrument']]
else:
    # Read instrument properties from the .itc or yml file
    if datafile_extension in ['.yaml', '.yml']:
        import yaml

        with open(datafiles[0], 'r') as yamlfile:
            yamldict = yaml.load(yamlfile)
            instrument_name = yamldict['instrument']
            if instrument_name in known_instruments.keys():
                instrument = known_instruments[instrument_name]
            else:
                raise ValueError("Unknown instrument {} specified in {}".format(instrument_name, datafiles[0]))
    elif datafile_extension in ['.itc']:
        instrument = Instrument(itcfile=datafiles[0])
    else:
        raise ValueError("The instrument needs to be specified on the commandline for non-standard files")


logging.debug("Current state:")
logging.debug(str(locals()))

# Close all figure windows.
import pylab
pylab.close('all')
logging.info("Reading ITC data from %s" % datafiles)


def input_to_experiment(datafile, heatsfile):
    """
    Create an Experiment object from the datafile and the heats file
    :param datafile:
    :param heatsfile:
    :return:
    """
    datafile_basename, datafile_extension = splitext(basename(datafile))
    if datafile_extension in ['.yaml', '.yml']:
        experiment = ExperimentYaml(datafile, experiment_name, instrument)
    elif datafile_extension in ['.itc']:
        if user_input["--dummy_itc_file"]:
            logging.info("Dummy itc file is used")
            experiment = ExperimentMicroCalWithDummyITC(datafile, experiment_name, instrument)
        else:
            experiment = ExperimentMicroCal(datafile, experiment_name, instrument)
    else:
        raise ValueError('Unknown file type. Check your file extension')

    # Read the integrated heats
    experiment.read_integrated_heats(heatsfile)

    _pickle_experimental_info(experiment)
    return experiment


def _pickle_experimental_info(experiment, out="experimental_information.pickle"):
    """
    """
    data_keys = ["data_filename", "instrument", "number_of_injections", "equilibration_time", "stir_speed", 
                "reference_power", "cell_volume", "injections", "filter_period_end_time", "filter_period_midpoint_time", 
                "differential_power", "name", "data_source", "number_of_injections", "target_temperature", "stir_rate", 
                "syringe_concentration", "cell_concentration", "time", "heat", "temperature", "filter_period_end_time", 
                "differential_power", "cell_temperature", "jacket_temperature"]

    exper_info = {key : getattr(experiment, key, None) for key in data_keys}
    pickle.dump(exper_info, open(out, "wb"))
    
    return None


# Construct a Model from Experiment object.

if user_input['twocomponent']:
    experiment = input_to_experiment(user_input['<datafile>'][0], user_input['<heatsfile>'][0])

    models = list()
    try:
        model = TwoComponentBindingModel(experiment, cell_concentration=user_input['--cc'],
                                         syringe_concentration=user_input['--cs'],
                                         dcell=user_input['--dc'], dsyringe=user_input['--ds'],
                                         uniform_cell_concentration=user_input['--uniform_cell_concentration'],
                                         uniform_syringe_concentration=user_input['--uniform_syringe_concentration'],
                                         concentration_range_factor=user_input['--concentration_range_factor'])

    except Exception as e:
            logging.error(str(e))
            logging.error(traceback.format_exc())
            raise Exception("MCMC model could not be constructed!\n" + str(e))

elif user_input['racemicmixture']:
    experiment = input_to_experiment(user_input['<datafile>'][0], user_input['<heatsfile>'][0])

    models = list()
    try:
        model = RacemicMixtureBindingModel(experiment, cell_concentration=user_input['--cc'],
                                           syringe_concentration=user_input['--cs'],
                                           dcell=user_input['--dc'], dsyringe=user_input['--ds'],
                                           uniform_cell_concentration=user_input['--uniform_cell_concentration'],
                                           uniform_syringe_concentration=user_input['--uniform_syringe_concentration'],
                                           concentration_range_factor=user_input['--concentration_range_factor']
                                           )

    except Exception as e:
        logging.error(str(e))
        logging.error(traceback.format_exc())
        raise Exception("MCMC model could not be constructed!\n" + str(e))

elif user_input['enantiomer']:
    experiment = input_to_experiment(user_input['<datafile>'][0], user_input['<heatsfile>'][0])

    models = list()
    try:
        model = EnantiomerBindingModel(experiment, cell_concentration=user_input['--cc'],
                                       syringe_concentration=user_input['--cs'],
                                       dcell=user_input['--dc'], dsyringe=user_input['--ds'],
                                       uniform_cell_concentration=user_input['--uniform_cell_concentration'],
                                       uniform_syringe_concentration=user_input['--uniform_syringe_concentration'],
                                       concentration_range_factor=user_input['--concentration_range_factor']
                                       )

    except Exception as e:
        logging.error(str(e))
        logging.error(traceback.format_exc())
        raise Exception("MCMC model could not be constructed!\n" + str(e))

elif user_input['competitive']:
    experiments = list()
    for datafile, heatsfile in zip(user_input['<datafile>'], user_input['<heatsfile>']):
        experiments.append(input_to_experiment(datafile, heatsfile))
    receptor = user_input['--receptor']
    try:
        model = CompetitiveBindingModel(experiments, receptor)
    except Exception as e:
        logging.error(str(e))
        logging.error(traceback.format_exc())
        raise Exception("MCMC model could not be constructed!\n" + str(e))

else:
    raise ValueError("Something has gone wrong with the selection of a model. Contact your developer.")

# First fit the model.
logging.info("Initializing the model with a maximum a posteriori fit...")
map = pymc.MAP(model)
map.fit(iterlim=nfit)
logging.info(map)
logging.info("Commence MCMC sampling...")
# Less verbosity, unless user specifies extreme verbosity
model.mcmc.sample(iter=niters, burn=nburn, thin=nthin, progress_bar=True, verbose=user_input['-v'] -3)


if user_input['twocomponent']:    # Plot individual terms.
    plot_two_component_model_results(model)

if user_input['racemicmixture']:
    plot_racemicmixture_model_results(model)

if user_input['enantiomer']:
    plot_enantiomer_model_results(model)



print("pickling mcmc traces")
traces = {}
for s in model.mcmc.stochastics:
    traces[s.__name__] = s.trace(chain=None)

pickle.dump(traces, open("traces.pickle", "wb"))

