#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
    Tool to create a new ECmean4 reference climatology for global mean.
    It requires to have cdo and cdo-bindings installed
    The reference file (gm_reference.yml) specifies all the details for each dataset
"""

__author__ = "Paolo Davini (p.davini@isac.cnr.it), Feb 2023." \
" Modified by Marianna Albanese (marianna.albanese@polito.it), Jan 2026."


import logging
import yaml
import numpy as np
import xarray as xr
import os
from cdo import *
from glob import glob

from ecmean.libs.files import load_yaml
from ecmean.libs.units import units_extra_definition, UnitsHandler
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.masks import masked_meansum, select_region
from ecmean.libs.areas import AreaCalculator
from ecmean.libs.support import identify_grid
from ecmean.libs.formula import _eval_formula

cdo = Cdo()

# === DEBUG MODE ===
DEBUG = False

# set default logging
logging.basicConfig(level=logging.INFO)

# variable list
atm_vars = ['tas_land', 'tas', 'psl', 'pr', 'evspsbl', 'pme', 'clt', 'cll', 'clm', 'clh',
            'pr_oce', 'pme_oce', 'pr_land', 'pme_land']
rad_vars = ['net_toa', 'rsnt', 'rlnt', 'rsntcs', 'rlntcs', 'swcf', 'lwcf',
            'rsns', 'rlns', 'hfss', 'hfls', 'net_sfc_nosn', 'net_sfc',
            'toamsfc_nosn', 'toamsfc']
oce_vars = ['tos', 'sos', 'zos', 'wfo']
ice_vars = ['siconc', 'siconc_north', 'siconc_south']

# put them together
variables = atm_vars + rad_vars + oce_vars + ice_vars

# to set: time period (default, can be shorter if data are missing)
year1 = 2000
year2 = 2024

if DEBUG:
    variables = ['pme']   
    year1 = 2000
    year2 = 2024       

# climatology yml output
CLIMNAME = 'EC26_2000_2024'
clim_info = f'create-reference-wilma-{CLIMNAME}.yml'
# yml file to get information on dataset on some machine
clim_file = os.path.join('../reference', f'gm_reference_{CLIMNAME}.yml')

# add other units
units_extra_definition()

# open the clim info file
info = load_yaml(clim_info)
years = list(range(year1, year2 + 1))

# skip NaN: if False, yearly/season average require that all the points are defined in the correspondent
# time window.
nanskipper = False

# set the land-sea mask (ERA5)
maskfile = info['mask']

# function to expand the path


def expand_filedata(directory, var, info):

    return os.path.expandvars(directory).format(datadir=info['dirs']['datadir'], mswepdir=info['dirs']['mswepdir'],
                                                eradir=info['dirs']['eradir'], esadir=info['dirs']['esadir'],
                                                dataset=info[var]['dataset'])

 # loop on variables to be processed
for var in variables:

    # get infos on domain, operation and masks
    mask_type = info[var].get('mask', 'global')
    domain = info[var].get('domain', 'atm')
    operation = info[var].get('operation', 'mean')

    # if outvalue exists, the variable is predifined and has not gridded dataset
    if 'outvalue' in info[var].keys():
        mf = info[var]['outvalue']
        real_year1 = ''
        real_year2 = ''

    else:

        print(var)
        # get the directory (specific treatment if a list, use glob to expand)
        if isinstance(info[var]['dir'], list):
            temp_list = [glob(expand_filedata(ss, var, info)) for ss in info[var]['dir']]
            filedata = [item for sublist in temp_list for item in sublist]
        # use wildcards from xarray
        else:
            filedata = glob(expand_filedata(info[var]['dir'], var, info))
        logging.warning(filedata)

        # load data and time select
        print("Loading multiple files...")
        xfield = xr.open_mfdataset(filedata, chunks='auto', preprocess=xr_preproc, engine='netcdf4')
        
        # if derived, use the formula skill (or just rename)
        cmd = info[var]['derived']
        xfield = _eval_formula(cmd, xfield).to_dataset(name=var)

        # select time
        valid_time = xfield[var].dropna("time", how="all").time
        avail_years = np.unique(valid_time.dt.year.values)
        year1_eff = max(year1, int(avail_years.min()))
        year2_eff = min(year2, int(avail_years.max()))
        if year1_eff > year2_eff:
            raise ValueError(
                f"{var}: requested period {year1}-{year2} "
                f"not compatible with data availability "
                f"{avail_years.min()}-{avail_years.max()}"
            )
        years_eff = list(range(year1_eff, year2_eff + 1))
        # select time
        cfield = xfield[var].sel(time=xfield.time.dt.year.isin(years_eff))

        real_year1 = year1_eff
        real_year2 = year2_eff
        # tell us that we do not have the full data window
        if (real_year1 != year1):
            logging.warning("Initial year different from what expected: " + str(real_year1))
        if (real_year2 != year2):
            logging.warning("Final year different from what expected: " + str(real_year2))

        # get a single record and exploit of ecmean function to estimate areas
        print("Compute cell area for weights...")
        first = cfield.to_dataset(name=var)
        gg = identify_grid(first)
        calc = AreaCalculator()
        weights = calc.calculate_area(first, gridtype=gg).load()

        # compute land sea mask
        if mask_type != 'global':
            print("Mask...")
            xmask = cdo.remapbil(filedata[0], input=maskfile, returnXArray='lsm')
            # elimina la dimensione time
            if 'time' in xmask.dims:
                xmask = xmask.isel(time=0, drop=True)
            # binarizza SEMPRE
            xmask = (xmask >= 0.5)
            xmask.load()
        else:
            xmask = 0.

        # yearly and season averages
        print("Time averages...")
        gfield1 = cfield.resample(time='YS', skipna=nanskipper).mean('time', skipna=nanskipper).load()
        gfield2 = cfield.resample(time='QE-NOV', skipna=nanskipper).mean('time', skipna=nanskipper).load()

        print("Season loop...")
        mf = {}
        for season in ['ALL', 'DJF', 'MAM', 'JJA', 'SON']:
            print(f"Season loop... {season}")
        # select the season
            if season == 'ALL':
                gfield = gfield1
            else:
                gfield = gfield2.sel(time=gfield2.time.dt.season.isin(season))
            # for winter, we drop first and last to have only complete season.
            # this reduces the sample by one but it is safer for variance
            if season == 'DJF':
                gfield = gfield.drop_isel(time=[0, gfield.sizes['time'] - 1])

            mf[season] = {}
            # print("Region loop...")
            for region in ['Global', 'Tropical', 'North Midlat', 'South Midlat', 'NH', 'SH', 'Equatorial', 'North Pole', 'South Pole']:

                # slice everything
                slicefield = select_region(gfield, region)
                sliceweights = select_region(weights, region)
                if mask_type != 'global':
                    slicemask = select_region(xmask, region)
                else:
                    slicemask = 0.

                # get the masked-mean-sum
                out = masked_meansum(xfield=slicefield, weights=sliceweights,
                                     mask=slicemask, operation=operation,
                                     mask_type=mask_type, domain=domain)

                # set the units
                units_handler = UnitsHandler(var,
                                             org_units=info[var]['org_units'],
                                             tgt_units=info[var]['tgt_units'],
                                             operation=info[var].get('operation', 'mean'),
                                             org_direction=info[var].get('direction', 'down')
                                             )
                offset, factor = units_handler.units_converter()
                # new_units = _units_are_integrals(info[var]['org_units'], info[var])
                # offset, factor = units_converter(new_units, info[var]['tgt_units'])
                # down = {'direction': 'down'}
                # factor = factor * directions_match(info[var], down)
                final = out * factor + offset

                omean = np.mean(final)
                ostd = np.std(final)
                mf[season][region] = {}
                mf[season][region]['mean'] = float(str(round(omean, 3)))
                mf[season][region]['std'] = float(str(round(ostd, 3)))
                if season == 'ALL' and region == 'Global':
                    logging.warning('{} {} {} mean is: {:.2f} +- {:.2f}'.format(var, season, region, omean, ostd))
                else:
                    logging.info('{} {} {} mean is: {:.2f} +- {:.2f}'.format(var, season, region, omean, ostd))

    # log output
    logging.info(mf)

    # preparing clim file
    if os.path.isfile(clim_file):
        dclim = load_yaml(clim_file)
    else:
        dclim = {}

    # initialize variable if not exists
    if var not in dclim:
        dclim[var] = {}

    # create the new dictonary
    dclim[var]['longname'] = info[var]['longname']
    dclim[var]['dataset'] = info[var]['dataset']
    if mask_type != 'global':
        dclim[var]['mask'] = mask_type
    if operation != 'mean':
        dclim[var]['operation'] = operation
    if operation != 'atm':
        dclim[var]['domain'] = domain
    dclim[var]['units'] = info[var]['tgt_units']
    dclim[var]['year1'] = real_year1
    dclim[var]['year2'] = real_year2
    if 'notes' in info[var].keys():
        dclim[var]['notes'] = info[var]['notes']
    dclim[var]['obs'] = mf

    # dump the yaml file
    with open(clim_file, 'w', encoding='utf-8') as file:
        yaml.safe_dump(dclim, file, sort_keys=False)
