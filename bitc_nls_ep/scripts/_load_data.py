import numpy as np

def load_heat_micro_cal(origin_heat_file):
    """
    :param origin_heat_file: str, name of heat file
    :return: 1d ndarray, heats in micro calorie
    """
    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6:
                heats.append(np.float(line.split()[0]))
    return np.array(heats)


def _number_of_lines(csv_file):
    nlines = np.loadtxt(csv_file, delimiter=",").shape[0]
    return nlines


def _read_digitized_heat(csv_file):
    heats = np.loadtxt(csv_file, delimiter=",")[:,-1]
    return heats


def _kcal_per_mol_of_injectant_2_micro_cal(injection_heat, injection_volume, syringe_concentration):
    """
    injection_heat : float, in units of kcal/mol of injectant
    injection_volume : float, in units of micro L
    syringe_concentration : in units of mili M
    """
    vol_in_liter = injection_volume * 10.**(-6)
    concen_in_M  = syringe_concentration * 10.**(-3)
    number_of_mol = concen_in_M * vol_in_liter
    
    heat_in_kcal_per_mol = injection_heat * number_of_mol
    heat_in_micro_cal    = heat_in_kcal_per_mol * 10.**(9)

    return heat_in_micro_cal