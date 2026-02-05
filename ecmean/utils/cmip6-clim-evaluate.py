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

#refclim = 'EC24'
refclim = 'HM25'

nprocs = 1
do_compute = True
do_create_clim = False
do_definitive = False

climdir = '../climatology/'

#models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'TaiESM1', 'CanESM5', 'CESM2',
#          'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR', 'CMCC-CM2-SR5', 'NorESM2-MM', 'GFDL-CM4']
if refclim == 'EC24':
    expname = 'historical'
    models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'CanESM5', 'CESM2', 'CNRM-CM6-1',
            'GISS-E2-1-G', 'ACCESS-CM2', 'CNRM-CM6-1', 'SAM0-UNICON', 'UKESM1-0-LL',
            'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR', 'NorESM2-MM', 'GFDL-CM4']
    mip = 'CMIP'
    year1 = 1985
    year2 = 2014
    # models currently missing on the ESGF
# models= ['CMCC-CM2-SR5', 'TaiESM1']
elif refclim == 'HM25':
    models = [
        'EC-Earth3P-HR', 'AWI-CM-1-1-HR', 'BCC-CSM2-HR', 'CMCC-CM2-VHR4', 
        'HadGEM3-GC31-HH', 'INM-CM5-H', 'CNRM-CM6-1-HR', 'ECMWF-IFS-HR']
    #models = ['CESM1-CAM5-SE-HR'] # crashes with the current code
    #models = ['CNRM-CM6-1-HR'] # fake EC-Earth3P-HR Ofx
    #models = ['ECMWF-IFS-HR'] # missing Ofx
    #models = ['AWI-CM-1-1-HR']
    expname ='hist-1950'
    mip = 'HighResMIP'
    year1 = 1985
    year2 = 2014
else:
    raise ValueError(f"Unknown climatology {refclim}.")

config_file = f'config-create-clim-{refclim}.yml'

def cfg_ensemble(model):
    
    if model in ['CNRM-CM6-1', 'UKESM1-0-LL', 'AWI-CM-1-1-HR', 'CNRM-CM6-1-HR']:
        return "r1i1p1f2"
    if model in ['EC-Earth3P-HR']:
        return  "r1i1p2f1"
    return "r1i1p1f1"

def cfg_consortium(model):
    if model == 'HadGEM3-GC31-HM':
        return 'NERC'
    return '*'



# call the loop of global mean on all the models
if do_compute:

    defaultconfig = load_yaml(config_file)

    for model in sorted(models):
        print(model)
        ensemble = cfg_ensemble(model)
        consortium = cfg_consortium(model)
        model_config = copy.deepcopy(defaultconfig)

        # to possibly drop biased figures
        drop = []
        if drop:
            for var in drop:
                for kind in ['atm2d', 'atm3d', 'oce', 'ice']:
                    if var in model_config['performance_indices']['variables'][kind]:
                        model_config['performance_indices']['variables'][kind].remove(var)

        performance_indices(expname, year1, year2, config=model_config, model=model,
                            ensemble=ensemble, consortium=consortium, mip=mip, numproc=nprocs, climatology=refclim,
                            loglevel='debug')

if do_create_clim:

    cfg = load_yaml(config_file)

    # dictionary with all elements
    full = {}
    for model in models:
        ensemble = cfg_ensemble(model)
        filename = os.path.join(cfg['dirs']['tab'], f'PI4_{refclim}_{expname}_{model}_{ensemble}_{year1}_{year2}.yml')
        if not os.path.exists(filename):
            raise ValueError(f'File {filename} does not exist, please run the performance indices first!')
        print(filename)
        filein = glob.glob(filename)
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
        print(f'Updating {var} climatology for {refclim}...')
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
