#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
    Tool to create a new ecmean reference climatology for global mean.
    It requires to have cdo and cdo-bindings installed
    The reference file (gm_reference.yml) specifies all the details for each dataset
"""

__author__ = "Paolo Davini (paolo.davini@cnr.it), Feb 2023." \
" Modified by Marianna Albanese (marianna.albanese@polito.it), Jan 2026."


import os
from glob import glob
from time import time

import logging
import yaml
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

from ecmean.libs.files import load_yaml
from ecmean.libs.units import units_extra_definition, UnitsHandler
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.masks import masked_meansum, select_region
from ecmean.libs.areas import AreaCalculator
from ecmean.libs.support import identify_grid
from ecmean.libs.formula import _eval_formula

from ecmean.utils.utils import timeframe_years, expand_filedata, \
                            select_time_period, parse_create_args, \
                            select_time_data

from cdo import *
cdo = Cdo()

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

# skip NaN: if False, yearly/season average require that all
# the points are defined in the correspondent time window.
NANSKIP = False

# add other units
units_extra_definition()


def main(climdata='EC26', timeframe='HIST', machine='wilma'):
    """Main function to create the reference climatology.
    
    Parameters
    ----------
    climdata : str
        Climate dataset name (default: 'EC26')
    timeframe : str
        Time period - HIST, PDAY, or CMIP (default: 'HIST')
    machine : str
        Machine name for data paths (default: 'wilma')
    """
    # define the years of the reference period
    year1, year2 = timeframe_years(timeframe)
    climname = f'{climdata}-{timeframe}'

    # climatology yml output
    clim_file = os.path.join('..', 'reference', f'gm_reference_{climname}.yml')
    # yml file to get information on dataset on some machine
    clim_info = f'create-reference-{machine}-{climdata}.yml'

    # open the clim info file
    info = load_yaml(clim_info)

    # set the land-sea mask (ERA5)
    maskfile = info['mask']

    # loop on variables to be processed
    for var in variables:

        logging.warning("Processing variable %s", var)
        tic = time()

        # get infos on domain, operation and masks
        mask_type = info[var].get('mask', 'global')
        domain = info[var].get('domain', 'atm')
        operation = info[var].get('operation', 'mean')

        # if outvalue exists, the variable is predifined and has not gridded dataset
        if 'outvalue' in info[var]:
            mf = info[var]['outvalue']
            real_year1 = ''
            real_year2 = ''

        else:

            # get the directory (specific treatment if a list, use glob to expand)
            if isinstance(info[var]['dir'], list):
                temp_list = [glob(expand_filedata(ss, var, info)) for ss in info[var]['dir']]
                filedata = [item for sublist in temp_list for item in sublist]
            # use wildcards from xarray
            else:
                filedata = glob(expand_filedata(info[var]['dir'], var, info))
            logging.debug(filedata)

            # load data and time select
            logging.info("Loading multiple files...")
            xfield = xr.open_mfdataset(filedata, chunks='auto', preprocess=xr_preproc,
                                    engine='netcdf4', join='outer', data_vars='all')
            
            # if derived, use the formula skill (or just rename)
            cmd = info[var]['derived']
            xfield = _eval_formula(cmd, xfield).to_dataset(name=var)

            # select time based on data availability
            cfield, real_year1, real_year2 = select_time_period(xfield, var, year1, year2)

            # get a single record and exploit of ecmean function to estimate areas
            logging.info("Compute cell area for weights...")
            first = cfield.to_dataset(name=var)
            gg = identify_grid(first)
            calc = AreaCalculator()
            weights = calc.calculate_area(first, gridtype=gg).load()

            # compute land sea mask
            if mask_type != 'global':
                logging.info("Masking...")
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
            logging.info("Time averages...")
            gfield1 = cfield.resample(time='YS', skipna=NANSKIP).mean('time', skipna=NANSKIP).load()
            gfield2 = cfield.resample(time='QE-NOV', skipna=NANSKIP).mean('time', skipna=NANSKIP).load()

            logging.info("Season loop...")
            mf = {}
            for season in ['ALL', 'DJF', 'MAM', 'JJA', 'SON']:
                logging.info("Processing season %s", season)
                # select the season


                mf[season] = {}
                # print("Region loop...")
                for region in ['Global', 'Tropical', 'North Midlat', 'South Midlat',
                            'NH', 'SH', 'Equatorial', 'North Pole', 'South Pole']:
                    
                    gfield = select_time_data(gfield1, gfield2, season)

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
                        logging.warning('%s %s %s mean is: %.2f +- %.2f', var, season, region, omean, ostd)
                    else:
                        logging.info('%s %s %s mean is: %.2f +- %.2f', var, season, region, omean, ostd)

        # log output
        logging.info(mf)

        toc = time()
        logging.warning('Processing in {:.4f} seconds'.format(toc - tic))

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

# setting up dask
if __name__ == "__main__":
    args = parse_create_args().parse_args()

    logging.getLogger().setLevel(args.loglevel.upper())

    # set up clusters
    if args.cores > 1:
        workers = args.cores
        cluster = LocalCluster(threads_per_worker=1, n_workers=workers)
        client = Client(cluster)
        logging.warning(client)

    main(args.climdata, args.timeframe, args.machine)

    if args.cores > 1:
        client.close()
        cluster.close()
