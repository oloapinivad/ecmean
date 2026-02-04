#!/usr/bin/env python3
'''
Shared functions for XArray ECmean4
'''

import os
import logging
import platform
import math
import multiprocessing
import pandas as pd
import numpy as np


##################
# HELP FUNCTIONS #
##################


loggy = logging.getLogger(__name__)


def set_multiprocessing_start_method():
    """Function to set the multiprocessing spawn method to fork"""
    plat = platform.system()
    if plat == 'Windows':
        raise OSError("Windows does not support 'fork' start method.")
    if plat in ['Darwin', 'Linux']:
        multiprocessing.set_start_method('fork', force=True)
    else:
        raise OSError(f"Unsupported operative system {plat}")
    loggy.debug('Multiprocessing start method is %s', multiprocessing.get_start_method())
    return plat, multiprocessing.get_start_method()


def check_time_axis(xtime, years):
    """Check if we have 12 months per year and if the required years
    have been found in the NetCDF files. """

    # unique, counts = np.unique(xtime.dt.month, return_counts=True)
    # unique, counts = np.unique(xtime.time.resample(time='1M').mean(), return_counts=True)
    unique, counts = np.unique(xtime.time.dt.month, return_counts=True)
    if len(unique) != 12 or not all(counts == counts[0]):
        loggy.warning('Check your data: some months might be missing...')
        loggy.warning('Month counts: %s', counts)

    # apparently this is also satisfied by the file browsing
    set1 = set(years)
    set2 = set(xtime.dt.year.values)
    missing = list(set1.difference(set2))
    if missing:
        loggy.warning('Check your data: some years are missing')
        loggy.warning('Year missing: %s', missing)


# def chunks(iterable, num):
#     """Generate num adjacent chunks of data from a list iterable
#     Split lists in a convenient way for a parallel process"""

#     size = int(np.ceil(len(iterable) / num))
#     it = iter(iterable)
#     return iter(lambda: tuple(itertools.islice(it, size)), ())

# def split(iterable, num):
#     k, m = divmod(len(iterable), num)
#     return (iterable[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num))

def runtime_weights(varlist):
    """
    Define the weights to estimate the best repartition of the cores
    This is done a-priori, considering that
    1) compound variables are more difficult to compute
    2) 3d variables requires more evaluation
    """

    w = {}
    for k in varlist:
        if k in ['ua', 'ta', 'va', 'hus']:
            t = 8
        elif k in ['pme', 'net_sfc_nosn', 'net_sfc', 'toamsfc_nosn', 'toamsfc',
                   'pr_oce', 'pme_oce', 'pr_land', 'pme_land', 'net_toa']:
            t = 3
        else:
            t = 1
        w[k] = t

    return w


def weight_split(a, n):
    """
    Use the weights by runtime_weights to provide a number of n chunks based on the
    minimum possible value of each chunk computing time. Then, provide the list of variables
    to be used in the multiprocessing routine
    """

    ws = sorted(runtime_weights(a).items(), key=lambda x: x[1], reverse=True)
    ordered = dict(ws)

    elists = [[0] for _ in range(n)]
    olists = [[] for _ in range(n)]

    count = 0
    for fff, value in ordered.items():
        elists[count].append(value)
        olists[count].append(fff)
        count = elists.index(min(elists))

    return olists


def check_var_interface(var, face):
    """Check if a var is defined in the interface file"""

    if var in face['variables']:
        return True

    loggy.warning('%s is not defined in the interface file, skipping it!', var)
    return False


def check_var_climatology(varlist, reference):
    """Check if a var is defined in the climatology/reference file"""

    missing = [element for element in varlist if element not in reference]
    if len(missing) > 0:
        raise KeyError(f'Variable/Variables {missing} is/are not defined in the climatology, aborting!')


def get_domain(var, face):
    """Given a variable var extract its domain (oce or atm) from the interface.
    To do so it creates a dictionary providing the domain associated with a component.
    (the interface file specifies the component for each domain instead)"""

    comp = face['filetype'][face['variables'][var]['filetype']]['component']
    d = face['model']['component']
    domain = dict(zip([list(d.values())[x]
                  for x in range(len(d.values()))], d.keys()))
    return domain[comp]


####################
# OUTPUT FUNCTIONS #
####################

def dict_to_dataframe(varstat, allowed = ['ALL', 'DJF', 'JJA', 'MAM', 'SON']):
    """
    Converts a nested 3-level dictionary to a pandas DataFrame.

    Parameters:
    varstat (dict): Nested dictionary with 3 levels.
    allowed (list): List of keys to be included in the DataFrame at the lower level.

    Returns:
    pd.DataFrame: Transformed DataFrame with hierarchical keys.
    """
    data_table = {}
    for i in varstat.keys():
        pippo = {}
        for outerkey, innerdict in varstat[i].items():
            if outerkey in allowed:
                for innerkey, values in innerdict.items():
                    pippo[(outerkey, innerkey)] = values
        data_table[i] = pippo
    data_table = pd.DataFrame(data_table).T
    return data_table


def write_tuning_table(linefile, varmean, var_table, diag, ref):
    """Write results appending one line to a text file.
       Write a tuning table: need to fix reference to face/ref"""

    if not os.path.isfile(linefile):
        with open(linefile, 'w', encoding='utf-8') as f:
            print('%model  ens  exp from   to ', end='', file=f)
            for var in var_table:
                print('{:>12s}'.format(var), end=' ', file=f)
            print('\n%                         ', end=' ', file=f)
            for var in var_table:
                print('{:>12s}'.format(ref[var]['units']), end=' ', file=f)
            print(file=f)

    with open(linefile, 'a', encoding='utf-8') as f:
        print(f'{diag.modelname} {diag.ensemble} {diag.expname}',
              '{:4d} {:4d} '.format(diag.year1, diag.year2), end='', file=f)
        for var in var_table:
            print(
                '{:12.5f}'.format(
                    varmean[var]['ALL']['Global'] *
                    ref[var].get(
                        'factor',
                        1)),
                end=' ',
                file=f)
        print(file=f)


def init_mydict(one, two):
    """Initialize a two level dictionary"""
    dd = {}
    for o in one:
        dd[o] = {}
        for t in two:
            dd[o][t] = float('NaN')

    return dd


def are_dicts_equal(dict1, dict2, tolerance=1e-6):
    """
    Recursively compare two dictionaries with a given tolerance for float comparisons.
    To be used for testing purposes
    """
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        if keys1 != keys2:
            return False

        for key in keys1:
            if not are_dicts_equal(dict1[key], dict2[key], tolerance):
                return False
        return True

    try:
        dict1 = float(dict1)
        dict2 = float(dict2)
        if math.isnan(dict1) and math.isnan(dict2):
            return True
        return math.isclose(dict1, dict2, rel_tol=tolerance)
    except ValueError:
        return dict1 == dict2
