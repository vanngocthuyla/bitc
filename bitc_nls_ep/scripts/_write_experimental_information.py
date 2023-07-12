"""
Write an Experiment object from the datafile and the heats file
"""

import sys
import logging
import os
from os.path import basename, splitext

import pickle
import numpy

sys.path.append('/Users/seneysophie/Work/Python/Local/bitc_nls_ep/bayesian-itc')

from bitc.experiments import ExperimentMicroCal, ExperimentMicroCalWithDummyITC, ExperimentYaml
from bitc.instruments import known_instruments, Instrument
from bitc.units import Quantity

def input_to_experiment(datafile, heatsfile, experiment_name, instrument_infor, dummy_itc_file=False):
    """
    Create an Experiment object from the datafile and the heats file
    :param datafile:
    :param heatsfile:
    :return:
    """
    datafile_basename, datafile_extension = splitext(basename(datafile))
    
    if len(instrument_infor)>0:
        # Use an instrument from the brochure
        instrument = known_instruments[instrument_infor]
    else:
        # Read instrument properties from the .itc or yml file
        if datafile_extension in ['.yaml', '.yml']:
            import yaml

            with open(datafile, 'r') as yamlfile:
                yamldict = yaml.load(yamlfile)
                instrument_name = yamldict['instrument']
                if instrument_name in known_instruments.keys():
                    instrument = known_instruments[instrument_name]
                else:
                    raise ValueError("Unknown instrument {} specified in {}".format(instrument_name, datafile))
        elif datafile_extension in ['.itc']:
            instrument = Instrument(itcfile=datafile)
        else:
            raise ValueError("The instrument needs to be specified on the commandline for non-standard files")
    
    if datafile_extension in ['.yaml', '.yml']:
        experiment = ExperimentYaml(datafile, experiment_name, instrument)
    elif datafile_extension in ['.itc']:
        if dummy_itc_file:
            logging.info("Dummy itc file is used")
            experiment = ExperimentMicroCalWithDummyITC(datafile, experiment_name, instrument)
        else:
            experiment = ExperimentMicroCal(datafile, experiment_name, instrument)
    else:
        raise ValueError('Unknown file type. Check your file extension')

    # Read the integrated heats
    experiment.read_integrated_heats(heatsfile)

    # _pickle_experimental_info(experiment)
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