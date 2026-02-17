#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
    Global mean evaluation of CMIP6 climatology
    It runs and then reads the output of the cmip6 models to provide an assessment of the
    climatology for CMIP6 models to be stored
"""

__author__ = "Paolo Davini (p.davini@isac.cnr.it), Sep 2022."

import os
import warnings
import glob
from collections import defaultdict
import yaml
import copy
import numpy as np
from ecmean.performance_indices import performance_indices
from ecmean.libs.files import load_yaml
warnings.simplefilter("ignore")

# the 30-year climatological window to be used as a baseline
year1 = 2000
year2 = 2024
expname = 'historical'
refclim = 'PDAY26'
nprocs = 4
do_compute = True
do_create_clim = True
do_definitive = True
config_file = 'config_create_clim_new.yml'
climdir = '../climatology/'

#models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'TaiESM1', 'CanESM5', 'CESM2',
#          'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR', 'CMCC-CM2-SR5', 'NorESM2-MM', 'GFDL-CM4']
models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'CanESM5', 'CESM2', 'CNRM-CM6-1',
          'GISS-E2-1-G', 'ACCESS-CM2', 'CNRM-CM6-1', 'SAM0-UNICON', 'UKESM1-0-LL',
          'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR', 'NorESM2-MM', 'GFDL-CM4']
#models = ['EC-Earth3']

# models currently missing on the ESGF
# models= ['CMCC-CM2-SR5', 'TaiESM1']



# call the loop of global mean on all the models
if do_compute:

    defaultconfig = load_yaml(config_file)

    for model in sorted(models):
        print(model)

        if model in ['CNRM-CM6-1', 'UKESM1-0-LL']:
            ENSEMBLE = "r1i1p1f2"
        else:
            ENSEMBLE = "r1i1p1f1"

        model_config = copy.deepcopy(defaultconfig)

        # to possibly drop biased figures
        drop = []
        if drop:
            for var in drop:
                for kind in ['2d_vars', '3d_vars', 'oce_vars', 'ice_vars']:
                    if var in model_config['PI'][kind]['field']:
                        model_config['PI'][kind]['field'].remove(var)

        performance_indices(expname, year1, year2, config=model_config, model=model,
                            ensemble=ENSEMBLE, numproc=nprocs, climatology=refclim,
                            loglevel='debug')

if do_create_clim:

    cfg = load_yaml(config_file)

    # dictionary with all elements
    full = {}
    for model in models:
        print(model)
        filein = glob.glob(os.path.join(cfg['dirs']['tab'], f'PI4_{refclim}_{expname}_{model}_r1i1p1f*_{year1}_{year2}.yml'))
        full[model] = load_yaml(filein[0])

    # idiot averaging
    M0 = models[0]
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for var in full[M0].keys():
        for season in full[M0][var].keys():
            for region in full[M0][var][season].keys():
                element = []
                for model, model_data in full.items():
                    if var in model_data:
                        if not np.isnan(model_data[var][season][region]):
                            element.append(model_data[var][season][region])
                out[var][season][region] = float(round(np.mean(element), 3))

    # clumsy way to get the models for each var
    mout = {}
    for var in full[M0].keys():
        melement = []
        for model, model_data in full.items():
            if var in model_data:
                if not np.isnan(model_data[var]['ALL']['Global']):
                    melement.append(model)
        mout[var] = melement

    # clim files
    pifile = os.path.join(climdir, refclim, f'pi_climatology_{refclim}.yml')
    if not do_definitive:
        update_pifile = os.path.join(climdir, refclim, f'pi_climatology_{refclim}_test.yml')
    else:
        update_pifile = os.path.join(climdir, refclim, f'pi_climatology_{refclim}.yml')
    piclim = load_yaml(pifile)

    # Update the climatology
    for var, season_data in out.items():
        piclim[var]['cmip6'] = {}
        for season, region_data in season_data.items():
            piclim[var]['cmip6'][season] = {}
            for region, value in region_data.items():
                piclim[var]['cmip6'][season][region] = float(value)
        piclim[var]['cmip6']['models'] = mout[var]
        piclim[var]['cmip6']['nmodels'] = len(mout[var])
        piclim[var]['cmip6']['year1'] = year1
        piclim[var]['cmip6']['year2'] = year2

    # dump the new file
    with open(update_pifile, 'w', encoding='utf8') as file:
        yaml.safe_dump(piclim, file, sort_keys=False)
