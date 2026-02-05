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

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from cdo import *
#from dask.distributed import Client, LocalCluster, progress

from ecmean.libs.climatology import check_histogram, full_histogram, mask_from_field, variance_threshold
from ecmean.libs.files import load_yaml
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.units import units_extra_definition

# activate CDO class
cdo = Cdo(logging=True)

# output for matplot lib
matplotlib.use('Agg')

# set default logging
logging.basicConfig(level=logging.INFO)

# variable list
variables = ['tas', 'pr', 'net_sfc', 'tauu', 'tauv', 'psl',
             'ua', 'va', 'ta', 'hus', 'tos', 'sos', 'siconc']

# to set: time period (default, can be shorter if data are missing) #WARNING MISSING
year1 = 1980
year2 = 2010

# yml file to get information on dataset on some machine
CLIM = 'HIST26'
clim_info = f'/work/users/malbanes/ECmean4/ecmean/utils/create-clim-wilma-{CLIM}.yml'

# figures : some diagnostic figures can be saved to check the consistency of mean and variance fields
do_figures = True
figdir = '/work/users/malbanes/figures/ecmean-py-variances/'

# targets resolutio
grids = ['r360x180']

# number of dask workers and threads
workers = 1
threads = 1

# skip NaN: if False, yearly/season average require that all the points are defined in the correspondent
# time window.
nanskipper = False

# some dataset show very low variance in some grid point: this might create
# irrealistic high values of PI due to the  division by variance performend
# a hack is to use 5 sigma from the mean of the log10 distribution of variance
# define a couple of threshold to remove variance outliers


# add other units
units_extra_definition()

# to exploit of dask we need a main function


def main():

    # always keep the attributes along the xarray
    xr.set_options(keep_attrs=True)

    # open the clim info file
    info = load_yaml(clim_info)

    # set few parameters
    clim_name = info['clim']
    years = list(range(year1, year2 + 1))

    # directory definitions and creations
    tmpdir = info['dirs']['tmpdir']
    tgtdir = info['dirs']['tgtdir'].format(clim=clim_name)
    datadir = info['dirs']['datadir']
    archivedir = info['dirs']['archivedir']
    for directory in [tmpdir, tgtdir, datadir]:
        os.makedirs(directory, exist_ok=True)

    # climatology yaml output
    clim_file = os.path.join(tgtdir, 'pi_climatology_' + clim_name + '.yml')

    # loop on variables to be processed
    for var in variables:

        logging.info(var)
        tic = time()
        # get the directory
        filedata = str(os.path.expandvars(info[var]['dir'])).format(
            datadir=datadir, archivedir=archivedir,
            dataset=info[var]['dataset'],
            varname=info[var]['varname'])
        logging.info(filedata)

        # load data and time select
        print("Loading multiple files...")
        # unable to operate with Parallel=True
        xfield = xr.open_mfdataset(filedata, chunks='auto',
                                   parallel=False, preprocess=xr_preproc, engine='netcdf4')
        xfield = xfield.rename({info[var]['varname']: var})
        cfield = xfield[var].sel(time=xfield.time.dt.year.isin(years))

        # keep original dtype
        # cfield = cfield.astype(cfield.encoding['dtype'])

        real_year1 = np.min(cfield.time.dt.year.values)
        real_year2 = np.max(cfield.time.dt.year.values)
        if (real_year1 != year1):
            logging.warning("Initial year different from what expected: %s", real_year1)
        if (real_year2 != year2):
            logging.warning("Final year different from what expected: %s", real_year2)

        # check existence of unit, then apply from file
        if 'units' in info[var]:
            cfield.attrs['units'] = info[var]['units']
        elif not hasattr(cfield, 'units'):
            sys.exit('ERROR: no unit found or defined!')

        # cleaning
        # cfield = fix_specific_dataset(var, info[var]['dataset'], cfield)
        logging.info(cfield)

        # monthly average using resample/pandas
        logging.info("resampling...")
        zfield = cfield.resample(time='1MS', skipna=nanskipper).mean('time', skipna=nanskipper)
        #zfield = zfield.persist()
        #progress(zfield)
        zfield.compute()

        if do_figures:
            print("Full histogram...")
            figname = 'values_' + var + '_' + info[var]['dataset'] + '_' + str(real_year1) + '_' + str(real_year2) + '.pdf'
            os.makedirs(os.path.join(figdir, var), exist_ok=True)
            file = os.path.join(figdir, var, figname)
            full_histogram(zfield, file)

        # dump the netcdf file to disk
        logging.info("new file...")
        temp_name = next(tempfile._get_candidate_names()) + '.nc'
        tmpout = os.path.join(tmpdir, temp_name)

        # preserve dtype for numerical reasons
        codes = ['dtype', '_FillValue', 'scale_factor', 'add_offset', 'missing_value']
        ftype = {k: v for k, v in cfield.encoding.items() if k in codes}
        logging.info(ftype)
        zfield.to_netcdf(tmpout, encoding={var: ftype})

        # loop on grids
        for grid in grids:

            # create target directory
            os.makedirs(os.path.join(tgtdir, grid), exist_ok=True)

            # use cdo to interpolate: call to attribute to exploit different interpolation
            logging.info("interpolation..")
            interpolator = getattr(cdo, info[var]['remap'])
            yfield = interpolator(grid, input=tmpout, returnXArray=var)

            # keep original dtype
            # yfield = yfield.astype(yfield.encoding['dtype'])
            # print(yfield)
            os.remove(tmpout)

            # create empty lists
            d1 = []
            d2 = []

            # compute the yearly mean and the season mean
            logging.info("Averaging...")
            gfield1 = yfield.resample(time='YS', skipna=nanskipper).mean('time', skipna=nanskipper)
            gfield2 = yfield.resample(time='QE-NOV', skipna=nanskipper).mean('time', skipna=nanskipper)

            # loop on seasons
            for season in ['ALL', 'DJF', 'MAM', 'JJA', 'SON']:
                logging.info(season)

                # select the season
                if season == 'ALL':
                    gfield = gfield1
                else:
                    gfield = gfield2.sel(time=gfield2.time.dt.season.isin(season))
                    # for winter, we drop first and last to have only complete season.
                    # this reduces the sample by one but it is safer for variance
                    if season == 'DJF':
                        gfield = gfield.drop_isel(time=[0, gfield.sizes['time'] - 1])

                logging.debug(gfield.shape)

                # zonal averaging for 3D fields
                if 'plev' in gfield.coords:
                    gfield = gfield.mean(dim='lon')
                    # select only up to 10hpa
                    gfield = gfield.sel(plev=slice(100000, 1000))

                # create a reference time (average year, average month of the season)
                reftime = pd.to_datetime(str(int((year1 + year2) / 2)) + '-' +
                                         str(gfield.time.dt.month.values[0]) + '-15')

                # compute mean and variance: remove NaN in this case only
                omean = gfield.mean('time', skipna=True, keepdims=True)
                ovar = gfield.var('time', skipna=True, keepdims=True)

                # special fix
                # ovar = fix_specific_dataset(var, info[var]['dataset'], ovar)

                # define the variance threshold
                low, high = variance_threshold(ovar)
                logging.debug(low)
                logging.debug(high)

                # clean according to thresholds
                fvar = ovar.where((ovar >= low) & (ovar <= high))
                fmean = omean.where((ovar >= low) & (ovar <= high))

                #print(fvar)

                if do_figures:
                    logging.info("Mean and variance histograms...")
                    figname = var + '_' + info[var]['dataset'] + '_' + grid + '_' + \
                        str(real_year1) + '_' + str(real_year2) + '_' + season + '.pdf'
                    os.makedirs(os.path.join(figdir, var), exist_ok=True)
                    file = os.path.join(figdir, var, figname)
                    check_histogram(omean, ovar, fvar, file)

                # add a reference time
                ymean = fmean.assign_coords({"time": ("time", [reftime])})
                yvar = fvar.assign_coords({"time": ("time", [reftime])})

                # append the dataset in the list
                d1.append(ymean)
                d2.append(yvar)

            # merge into a single dataarray
            season_mean = xr.concat(d1[1:], dim='time')
            season_variance = xr.concat(d2[1:], dim='time')
            full_mean = d1[0]
            full_variance = d2[0]

            # define compression and dtype for time, keep original dtype
            ftype["zlib"] = True
            compression = {var: ftype, 'time': {'dtype': 'f8'}}

            # define file suffix
            suffix = var + '_' + info[var]['dataset'] + '_' + grid + '_' + str(real_year1) + '-' + str(real_year2) + '.nc'

            # save full - standard format
            full_variance.to_netcdf(os.path.join(tgtdir, grid, 'climate_variance_' + suffix), encoding=compression)
            full_mean.to_netcdf(os.path.join(tgtdir, grid, 'climate_average_' + suffix), encoding=compression)

            # save season - 4 season format
            season_variance.to_netcdf(os.path.join(tgtdir, grid, 'seasons_variance_' + suffix), encoding=compression)
            season_mean.to_netcdf(os.path.join(tgtdir, grid, 'seasons_average_' + suffix), encoding=compression)

            toc = time()
            logging.info('Processing in {:.4f} seconds'.format(toc - tic))

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

    # set up clusters
    #cluster = LocalCluster(threads_per_worker=threads, n_workers=workers)
    #client = Client(cluster)
    #logging.warning(client)
    main()