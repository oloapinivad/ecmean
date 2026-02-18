#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
    Tool to create a new ECmean4 climatology.
    It requires to have cdo and cdo-bindings installed
"""

__author__ = "Paolo Davini (p.davini@isac.cnr.it), Sep 2022."

import logging
import os
import sys
import tempfile
from time import time
from dask.distributed import Client, LocalCluster

import matplotlib
import pandas as pd
import xarray as xr
import yaml
from cdo import *
#from dask.distributed import Client, LocalCluster, progress

from ecmean.libs.climatology import check_histogram, full_histogram, mask_from_field, variance_threshold
from ecmean.libs.files import load_yaml
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.units import units_extra_definition
from ecmean.utils.utils import select_time_period, timeframe_years, \
    parse_create_args, select_time_data

# activate CDO class
cdo = Cdo(logging=True)

# output for matplot lib
matplotlib.use('Agg')

# set default logging
logging.basicConfig(level=logging.INFO)

# variable list
variables = ['tas', 'pr', 'net_sfc', 'tauu', 'tauv', 'psl',
             'ua', 'va', 'ta', 'hus', 'tos', 'sos', 'siconc']

# targets resolution
grids = ['r360x180']

# skip NaN: if False, yearly/season average require that all
# the points are defined in the correspondent time window.
NANSKIP = False

# some dataset show very low variance in some grid point: this might create
# irrealistic high values of PI due to the  division by variance performend
# a hack is to use 5 sigma from the mean of the log10 distribution of variance
# define a couple of threshold to remove variance outliers
FIGDIR = '/work/users/malbanes/figures/ecmean-py-variances/'

# add other units
units_extra_definition()


def main(climdata='EC26', timeframe='HIST', machine='wilma', do_figures=False):
    """Main function to create the climatology.
    
    Parameters
    ----------
    climdata : str
        Climate dataset name (default: 'EC26')
    timeframe : str
        Time period - HIST, PDAY, or CMIP (default: 'HIST')
    machine : str
        Machine name for data paths (default: 'wilma')
    do_figures : bool
        Generate diagnostic figures (default: False)
    """
    # define the years from timeframe
    year1, year2 = timeframe_years(timeframe)
    climname = f'{climdata}-{timeframe}'

    # climatology yaml output
    tgtdir = os.path.join('..', 'climatology', climname)
    os.makedirs(tgtdir, exist_ok=True)
    clim_file = os.path.join(tgtdir, 'pi_climatology_' + climname + '.yml')
    # yml file to get information on dataset on some machine
    clim_info = f'create-clim-{machine}-{climdata}.yml'

    # figures directory
    if do_figures:
        figdir = os.path.join(FIGDIR, climname)
        os.makedirs(figdir, exist_ok=True)

    # always keep the attributes along the xarray
    xr.set_options(keep_attrs=True)

    # open the clim info file
    info = load_yaml(clim_info)

    # directory definitions and creations
    datadir = info['dirs']['datadir']
    archivedir = info['dirs']['archivedir']

    # loop on variables to be processed
    for var in variables:

        logging.warning('Processing variable: %s', var)
        tic = time()
        # get the directory
        filedata = str(os.path.expandvars(info[var]['dir'])).format(
            datadir=datadir, archivedir=archivedir,
            dataset=info[var]['dataset'],
            varname=info[var]['varname'])
        logging.debug(filedata)

        # load data and time select
        print("Loading multiple files...")
        # unable to operate with Parallel=True
        xfield = xr.open_mfdataset(filedata, chunks='auto',
                                   parallel=False, preprocess=xr_preproc, engine='netcdf4',
                                   data_vars='all', join='outer', compat='no_conflicts')
        xfield = xfield.rename({info[var]['varname']: var})
        
        # select time based on data availability
        cfield, real_year1, real_year2 = select_time_period(xfield, var, year1, year2)

        # check existence of unit, then apply from file
        if 'units' in info[var]:
            cfield.attrs['units'] = info[var]['units']
        elif not hasattr(cfield, 'units'):
            raise ValueError('no unit found or defined!')

        # cleaning
        # cfield = fix_specific_dataset(var, info[var]['dataset'], cfield)
        logging.debug(cfield)

        # monthly average using resample/pandas
        logging.info("resampling...")
        zfield = cfield.resample(time='1MS', skipna=NANSKIP).mean('time', skipna=NANSKIP)
        #zfield = zfield.persist()
        #progress(zfield)
        zfield.compute()

        if do_figures:
            logging.debug("Full histogram...")
            figname = f'values_{var}_{info[var]["dataset"]}_{real_year1}_{real_year2}_full.pdf'
            file = os.path.join(figdir, var, figname)
            full_histogram(zfield, file)

        # dump the netcdf file to disk
        logging.info("new file...")
        tmpfile = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)

        # preserve dtype for numerical reasons
        codes = ['dtype', '_FillValue', 'scale_factor', 'add_offset', 'missing_value']
        ftype = {k: v for k, v in cfield.encoding.items() if k in codes}
        logging.info(ftype)
        zfield.to_netcdf(tmpfile.name, encoding={var: ftype})

        # loop on grids
        for grid in grids:

            # create target directory
            os.makedirs(os.path.join(tgtdir, grid), exist_ok=True)

            # use cdo to interpolate: call to attribute to exploit different interpolation
            logging.info("interpolation..")
            interpolator = getattr(cdo, info[var]['remap'])
            yfield = interpolator(grid, input=tmpfile.name, returnXArray=var)
    
            # create empty lists
            d1 = []
            d2 = []

            # compute the yearly mean and the season mean
            logging.info("Averaging...")
            gfield1 = yfield.resample(time='YS', skipna=NANSKIP).mean('time', skipna=NANSKIP).load()
            gfield2 = yfield.resample(time='QE-NOV', skipna=NANSKIP).mean('time', skipna=NANSKIP).load()

            # loop on seasons
            for season in ['ALL', 'DJF', 'MAM', 'JJA', 'SON']:
                logging.info(season)

                gfield = select_time_data(gfield1, gfield2, season)

                logging.debug(gfield.shape)

                # zonal averaging for 3D fields
                if 'plev' in gfield.coords:
                    gfield = gfield.mean(dim='lon')
                    # select only up to 10hpa
                    gfield = gfield.sel(plev=slice(100000, 1000))

                # create a reference time (average year, average month of the season)
                timestring = f'{int((year1 + year2) / 2)}-{str(gfield.time.dt.month.values[0])}-15'
                reftime = pd.to_datetime(timestring)

                # compute mean and variance: remove NaN in this case only
                omean = gfield.mean('time', skipna=True, keepdims=True)
                ovar = gfield.var('time', skipna=True, keepdims=True)

                # define the variance threshold
                low, high = variance_threshold(ovar)
                logging.info('Variance threshold: low = %s, high = %s', low, high)

                # clean according to thresholds
                fvar = ovar.where((ovar >= low) & (ovar <= high))
                fmean = omean.where((ovar >= low) & (ovar <= high))

                if do_figures:
                    logging.info("Mean and variance histograms...")
                    figname = f'{var}_{info[var]["dataset"]}_{grid}_{real_year1}_{real_year2}_{season}.pdf'
                    os.makedirs(os.path.join(figdir, var), exist_ok=True)
                    file = os.path.join(figdir, var, figname)
                    check_histogram(omean, ovar, fvar, file)

                # add a reference time
                ymean = fmean.assign_coords({"time": ("time", [reftime])})
                yvar = fvar.assign_coords({"time": ("time", [reftime])})

                # append the dataset in the list
                d1.append(ymean)
                d2.append(yvar)
            
            # cleanup temporary file
            os.remove(tmpfile.name)

            # merge into a single dataarray
            season_mean = xr.concat(d1[1:], dim='time')
            season_variance = xr.concat(d2[1:], dim='time')
            full_mean = d1[0]
            full_variance = d2[0]

            # define compression and dtype for time, keep original dtype
            ftype["zlib"] = True
            compression = {var: ftype, 'time': {'dtype': 'f8'}}

            # define file suffix
            suffix = f'{var}_{info[var]["dataset"]}_{grid}_{real_year1}-{real_year2}.nc'

            # save full - standard format
            full_variance.to_netcdf(os.path.join(tgtdir, grid, 'climate_variance_' + suffix), encoding=compression)
            full_mean.to_netcdf(os.path.join(tgtdir, grid, 'climate_average_' + suffix), encoding=compression)

            # save season - 4 season format
            season_variance.to_netcdf(os.path.join(tgtdir, grid, 'seasons_variance_' + suffix), encoding=compression)
            season_mean.to_netcdf(os.path.join(tgtdir, grid, 'seasons_average_' + suffix), encoding=compression)

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

            # assign to the dictionary the required info
            dclim[var]['dataset'] = info[var]['dataset']
            dclim[var]['description'] = info[var]['description']
            dclim[var]['longname'] = info[var]['longname']
            # dclim[var]['dataname'] = info[var]['varname']
            dclim[var]['remap'] = info[var]['remap']
            dclim[var]['mask'] = mask_from_field(full_mean)
            dclim[var]['units'] = full_mean.attrs['units']
            dclim[var]['year1'] = int(real_year1)
            dclim[var]['year2'] = int(real_year2)

            # dump the yaml file
            with open(clim_file, 'w', encoding='utf8') as file:
                yaml.safe_dump(dclim, file, sort_keys=False)

            logging.debug(dclim)


# setting up dask
if __name__ == "__main__":
    args = parse_create_args()

    logging.getLogger().setLevel(args.loglevel.upper())

    if args.cores > 1:
        workers = args.cores
        cluster = LocalCluster(threads_per_worker=1, n_workers=workers)
        client = Client(cluster)
        logging.warning(client)

    main(args.climdata, args.timeframe, args.machine, args.figures)

    if args.cores > 1:
        client.close()
        cluster.close()