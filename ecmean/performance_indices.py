#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 python3 version of ECmean performance indices tool
 Using a reference file from yaml and cdo bindings

 @author Paolo Davini (p.davini@isac.cnr.it), 2022
 @author Jost von Hardenberg (jost.hardenberg@polito.it), 2022
'''

import sys
import os
from time import time
from multiprocessing import Process, Manager
import numpy as np
import xarray as xr
import yaml
import dask
from ecmean import Diagnostic, Supporter, UnitsHandler
from ecmean.libs.general import weight_split, get_domain, \
    check_time_axis, init_mydict, check_var_interface, check_var_climatology, \
    set_multiprocessing_start_method
from ecmean.libs.files import var_is_there, get_inifiles, load_yaml, \
    make_input_filename, get_clim_files, load_output_yaml
from ecmean.libs.formula import formula_wrapper
from ecmean.libs.masks import mask_field, select_region
from ecmean.libs.areas import guess_bounds
from ecmean.libs.units import units_extra_definition
from ecmean.libs.ncfixers import xr_preproc, adjust_clim_file
from ecmean.libs.ecplotter import ECPlotter
from ecmean.libs.parser import parse_arguments
from ecmean.libs.loggy import setup_logger

dask.config.set(scheduler="synchronous")

class PerformanceIndices:
    """
    Class to compute the performance indices for a given experiment and years.

    Attributes:
        exp (str): Experiment name.
        year1 (int): Start year of the experiment.
        year2 (int): End year of the experiment.
        config (str): Path to the configuration file. Default is 'config.yml'.
        loglevel (str): Logging level. Default is 'WARNING'.
        numproc (int): Number of processes to use. Default is 1.
        climatology (str): Climatology to use. Default is 'EC23'.
        interface (str): Path to the interface file.
        model (str): Model name.
        ensemble (str): Ensemble identifier. Default is 'r1i1p1f1'.
        silent (bool): If True, suppress output. Default is None.
        xdataset (xarray.Dataset): Dataset to use.
        outputdir (str): Directory to store output files.
        loggy (logging.Logger): Logger instance.
        diag (Diagnostic): Diagnostic instance.
        face (dict): Interface dictionary.
        piclim (dict): Climatology dictionary.
        util_dictionary (Supporter): Utility dictionary for remapping and masks.
        varstat (dict): Dictionary to store variable statistics.
        funcname (str): Name of the class.
        start_time (float): Start time for performance measurement.
        title (str): Title of the plot, overrides default title.
    Methods:
        toc(message):
            Update the timer and log the elapsed time.
        prepare():
            Prepare the necessary components for performance indices calculation.
        run():
            Run the performance indices calculation.
        store(yamlfile=None):
            Store the performance indices in a yaml file.
        plot(mapfile=None, figformat='pdf'):
            Generate the heatmap for performance indices.
        pi_worker(util, piclim, face, diag, field_3d, varstat, varlist):
            Main parallel diagnostic worker for performance indices.
    """

    def __init__(self, exp, year1, year2, config='config.yml',
                 loglevel='WARNING', numproc=1, climatology=None,
                 interface=None, model=None, ensemble='r1i1p1f1',
                 silent=None, xdataset=None, outputdir=None,
                 extrafigure=False, title=None):
        """Initialize the PerformanceIndices class with the given parameters."""

        self.loglevel = loglevel
        self.loggy = setup_logger(level=self.loglevel)
        self.diag = Diagnostic(exp=exp, year1=year1, year2=year2, config=config,
                               funcname=self.__class__.__name__,
                               numproc=numproc, climatology=climatology,
                               interface=interface,
                               modelname=model, ensemble=ensemble,
                               outputdir=outputdir, xdataset=xdataset)
        self.silent = silent
        self.face = None
        self.piclim = None
        self.util_dictionary = None
        self.varstat = None
        self.extrafigure = extrafigure #special key to be set for manual debugging, producing extra figures: DO NOT USE
        self.outarray = None
        self.start_time = time()
        self.title = title

    def toc(self, message):
        """Update the timer and log the elapsed time."""
        elapsed_time = time() - self.start_time
        self.start_time = time()
        self.loggy.info('%s time: %.2f seconds', message, elapsed_time)

    def prepare(self):
        """Prepare the necessary components for performance indices calculation."""
        # set dask and multiprocessing fork
        plat, mprocmethod = set_multiprocessing_start_method()
        self.loggy.info('Running on %s and multiprocessing method set as "%s"', plat, mprocmethod)

        # initialize the diag class, load the interface and the reference file
        self.face = load_yaml(self.diag.interface)
        self.piclim = load_yaml(self.diag.climfile)

        # check that everything is there
        check_var_climatology(self.diag.field_all, self.piclim.keys())

        # Create missing folders
        os.makedirs(self.diag.tabdir, exist_ok=True)
        os.makedirs(self.diag.figdir, exist_ok=True)

        # new bunch of functions to set grids, create correction command, masks and areas
        comp = self.face['model']['component']  # Get component for each domain

        # all clim have the same grid, read from the first clim available and get target grid
        clim, _ = get_clim_files(self.piclim, 'tas', self.diag, 'ALL')
        target_remap_grid = xr.open_dataset(clim)

        # get file info files
        inifiles = get_inifiles(self.face, self.diag)

        # create remap dictionary with atm and oce interpolators
        self.util_dictionary = Supporter(
            comp, inifiles['atm'], inifiles['oce'],
            areas=False, remap=True, targetgrid=target_remap_grid
        )

        # verify if we can run amip, omip or coupled run
        self.diag.configure_amip_omip_cpld(self.util_dictionary)

        self.toc('Preparation')

    def run(self):
        """Run the performance indices calculation."""
        # main loop: manager is required for shared variables
        mgr = Manager()

        # dictionaries are shared, so they have to be passed as functions
        self.varstat = mgr.dict()
        processes = []

        # special dictionary for extra figures
        if self.extrafigure:
            self.loggy.debug('Extra figures requested, defining arrays')
            self.outarray = mgr.dict()
            for kind in ['bias', 'map']:
                self.outarray[kind] = mgr.dict()
        else:
            self.outarray = False

        # loop on the variables, create the parallel process
        for varlist in weight_split(self.diag.field_all, self.diag.numproc):
            core = Process(target=self.pi_worker, args=(self.util_dictionary, self.piclim,
                                                        self.face, self.diag, self.diag.field_atm3d,
                                                        self.varstat, self.outarray, varlist,
                                                        self.loglevel))
            core.start()
            processes.append(core)

        # wait for the processes to finish
        for proc in processes:
            proc.join()
        self.toc('Computation')

    def store(self, yamlfile=None):
        """Store the performance indices in a yaml file."""

        # order according to the original request the fields in the yaml file
        self.varstat = {var: self.varstat[var] for var in self.diag.field_all}

        # dump the yaml file for PI, including all the seasons (need to copy to avoid mess)
        if yamlfile is None:
            yamlfile = self.diag.filenames('yml')
        self.loggy.info('Storing the performance indices in %s', yamlfile)
        with open(yamlfile, 'w', encoding='utf-8') as file:
            yaml.safe_dump(self.varstat, file, default_flow_style=False, sort_keys=False)
        self.toc('Storing')



    def plot(self, diagname='performance_indices', mapfile=None, figformat='pdf', storefig=True, returnfig=False):     
        """
        Generate the heatmap for performance indices.

        Args:
            diagname (str): Name of the diagnostic. Default is 'performance_indices'.
            mapfile (str): Path to the output file. If None, it will be defined automatically following ECmean syntax.
            storefig (bool): If True, store the figure in the specified file. Default is True.
            returnfig (bool): If True, return the figure object. Default is False.
        """
        plotter = ECPlotter(
            diagnostic=diagname, modelname=self.diag.modelname,
            expname=self.diag.expname, year1=self.diag.year1,
            year2=self.diag.year2, regions=self.diag.regions,
            seasons=self.diag.seasons)
        if self.varstat is None:
            self.varstat = load_output_yaml(self.diag.filenames('yml'))
        if mapfile is None:
            mapfile = self.diag.filenames(figformat)

        fig = plotter.heatmap_plot(
            data=self.varstat, base=self.piclim,
            variables=self.diag.field_all, climatology=self.diag.climatology,
            filename=mapfile, storefig=storefig, title=self.title)
        
        self.toc('Plotting')

        if returnfig:
            self.loggy.info('Returning figure object')
            return fig
        
    # def plot(self, mapfile=None, figformat='pdf', returnfig=False, storefig=True):
    #     """
    #     Generate the heatmap for performance indices.

    #     Args:
    #         mapfile (str): Path to the output file. If None, it will be defined automatically following ECmean syntax.
    #         figformat (str): Format of the output file. Default is 'pdf'.
    #         returnfig (bool): If True, return the figure object. Default is False.
    #         storefig (bool): If True, store the figure in the specified file. Default is True.
    #     """
    #     if self.extrafigure:
    #         self.loggy.debug('Plotting extra results...')
    #         print(self.outarray)
    #         debugfig = os.path.join(self.diag.figdir, f"map_{os.path.basename(self.diag.filenames('png'))}")
    #         plot_xarray(self.outarray['map'], filename=debugfig, log_scale=True, cmap='viridis')
    #         debugfig = os.path.join(self.diag.figdir, f"bias_{os.path.basename(self.diag.filenames('png'))}")
    #         plot_xarray(self.outarray['bias'], filename=debugfig, cmap='seismic', log_scale=False)

    #     # to this date, only EC23/EC24 support comparison with CMIP6 data
    #     if self.diag.climatology in ['EC23', 'EC24']:

    #         # load yaml file if is missing
    #         if not self.varstat:
    #             yamlfile = self.diag.filenames('yml')
    #             self.loggy.info('Loading the stored data from the yaml file %s', yamlfile)
    #             if os.path.isfile(yamlfile):
    #                 with open(yamlfile, 'r', encoding='utf-8') as file:
    #                     self.varstat = yaml.safe_load(file)
    #             else:
    #                 raise FileNotFoundError(f'File {yamlfile} not found.')

    #         # prepare the data for the heatmap from the original yaml dictionaries
    #         self.loggy.debug('%s, %s, %s', self.varstat, self.piclim, self.diag.field_all)
    #         data2plot, cmip6, longnames = prepare_clim_dictionaries_pi(data=self.varstat,
    #                                                                    clim=self.piclim,
    #                                                                    shortnames=self.diag.field_all)

    #         # call the heatmap routine for a plot
    #         if mapfile is None:
    #             mapfile = self.diag.filenames(figformat)

    #         fig = heatmap_comparison_pi(
    #             data_dict=data2plot, cmip6_dict=cmip6, 
    #             diag=self.diag, longnames=longnames, filemap=mapfile,
    #             storefig=storefig)

    #         self.toc('Plotting')
    #         if returnfig:
    #             return fig
        
    #     else:
    #         self.loggy.warning('Only EC23 and EC24 climatology is supported for comparison with CMIP6 data.')


    @staticmethod
    def pi_worker(util, piclim, face, diag, field_3d, varstat, dictarray, varlist, loglevel):
        """
        Main parallel diagnostic worker for performance indices.

        Args:
            util (Supporter): Utility dictionary for remapping and masks.
            piclim (dict): Climatology dictionary.
            face (dict): Interface dictionary.
            diag (Diagnostic): Diagnostic instance.
            field_3d (list): List of 3D fields.
            varstat (dict): Dictionary to store variable statistics.
            dictarray (dict): Dictionary to store the output array.
            varlist (list): List of variables to process.
        """
        loggy = setup_logger(level=loglevel)

        # from python 3.14 this has to be into the worker
        units_extra_definition()

        for var in varlist:
            # store NaN in dict (can't use defaultdict due to multiprocessing)
            result = init_mydict(diag.seasons, diag.regions)

            if check_var_interface(var, face):
                # get domain
                domain = get_domain(var, face)

                # get masks
                domain_mask = getattr(util, domain + 'mask')

                # check if required variables are there: use interface file
                # check into first file, and load also model variable units
                infile = make_input_filename(var, face, diag)

                # check if var is available
                isavail, varunit = var_is_there(infile, var, face)

                # if the variable is available
                if isavail:
                    # perform the unit conversion extracting offset and factor
                    offset, factor = UnitsHandler(var, org_units=varunit,
                                                  clim=piclim, face=face).units_converter()

                    # open file: chunking on time only, might be improved
                    if not isinstance(infile, (xr.DataArray, xr.Dataset)):
                        xfield = xr.open_mfdataset(
                            infile, preprocess=xr_preproc, chunks={'time': 12},
                            data_vars='all', join='outer', compat='no_conflicts')
                    else:
                        xfield = infile

                    # in case of big files with multi year, be sure of having opened the right records
                    xfield = xfield.sel(time=xfield.time.dt.year.isin(diag.years_joined))

                    # check time axis
                    check_time_axis(xfield.time, diag.years_joined)

                    # get the data-array field for the required var
                    outfield = formula_wrapper(var, face, xfield)

                    # mean over time and fixing of the units
                    for season in diag.seasons:
                        loggy.debug(season)

                        # copy of the full field
                        tmean = outfield.copy(deep=True)

                        # get filenames for climatology
                        clim, vvvv = get_clim_files(piclim, var, diag, season)

                        # open climatology files, fix their metadata
                        cfield = adjust_clim_file(xr.open_mfdataset(clim, preprocess=xr_preproc))
                        vfield = adjust_clim_file(xr.open_mfdataset(vvvv, preprocess=xr_preproc), remove_zero=True)

                        # season selection
                        if season != 'ALL':
                            tmean = tmean.sel(time=tmean.time.dt.season.isin(season))
                            cfield = cfield.sel(time=cfield.time.dt.season.isin(season))
                            vfield = vfield.sel(time=vfield.time.dt.season.isin(season))

                        # averaging, applying offset and factor and loading
                        tmean = (tmean.mean(dim='time') * factor + offset).load()

                        # averaging and loading the climatology
                        cfield = cfield.mean(dim='time').load()
                        vfield = vfield.mean(dim='time').load()

                        # apply interpolation, if fixer is available and with different grids
                        fix = getattr(util, f'{domain}fix')
                        remap = getattr(util, f'{domain}remap')

                        if fix:
                            tmean = fix(tmean, keep_attrs=True)
                        try:
                            final = remap(tmean, keep_attrs=True)
                        except ValueError:
                            loggy.error('Cannot interpolate %s with the current weights...', var)
                            continue

                        # vertical interpolation
                        if var in field_3d:
                            # xarray interpolation on plev, forcing to be in Pascal
                            final = final.metpy.convert_coordinate_units('plev', 'Pa')
                            if set(final['plev'].data) != set(cfield['plev'].data):
                                loggy.warning('%s: Need to interpolate vertical levels...', var)
                                final = final.interp(plev=cfield['plev'].data, method='linear')

                                # safety check for missing values
                                sample = final.isel(lon=0, lat=0)
                                if np.sum(np.isnan(sample)) != 0:
                                    loggy.warning(
                                        '%s: You have NaN after the interpolation, this will affect your PIs...', var)
                                    levnan = cfield['plev'].where(np.isnan(sample))
                                    loggy.warning(levnan[~np.isnan(levnan)].data)

                            # zonal mean
                            final = final.mean(dim='lon')

                            # compute PI
                            complete = (final - cfield)**2 / vfield

                            # compute vertical bounds as weights
                            bounds_lev = guess_bounds(complete['plev'], name='plev')
                            bounds = abs(bounds_lev[:, 0] - bounds_lev[:, 1])
                            www = xr.DataArray(bounds, coords=[complete['plev']], dims=['plev'])

                            # vertical mean
                            outarray = complete.weighted(www).mean(dim='plev')

                        # horizontal averaging with land-sea mask
                        else:
                            complete = (final - cfield)**2 / vfield
                            outarray = mask_field(xfield=complete, mask_type=piclim[var]['mask'], dom=domain, mask=domain_mask)

                        # loop on different regions
                        for region in diag.regions:
                            slicearray = select_region(outarray, region)

                            # latitude-based averaging
                            weights = np.cos(np.deg2rad(slicearray.lat))
                            out = slicearray.weighted(weights).mean().data
                            # store the PI
                            result[season][region] = round(float(out), 3)

                            # diagnostic
                            if region == 'Global':
                                loggy.info('PI for %s %s %s %s', region, season, var, result[season][region])

                # debug array for extrafigures
                if not isinstance(dictarray, bool):
                    dictarray['map'][var] = complete if 'complete' in locals() else None
                    dictarray['bias'][var] = final - cfield if 'final' in locals() else None

            # nested dictionary, to be redefined as a dict to remove lambdas
            varstat[var] = result


def pi_entry_point():
    """
    Command line interface to run the performance indices calculation.
    """
    # read arguments from command line
    args = parse_arguments(sys.argv[1:], script='pi')

    performance_indices(exp=args.exp, year1=args.year1, year2=args.year2,
                        numproc=args.numproc, loglevel=args.loglevel,
                        climatology=args.climatology,
                        interface=args.interface, config=args.config,
                        model=args.model, ensemble=args.ensemble, outputdir=args.outputdir)


def performance_indices(exp, year1, year2, config='config.yml', loglevel='WARNING',
                        numproc=1, climatology=None, interface=None, model=None,
                        ensemble='r1i1p1f1', silent=None, xdataset=None, outputdir=None,
                        title=None):
    """
    Wrapper function to compute the performance indices for a given experiment and years.
    """
    pi = PerformanceIndices(exp=exp, year1=year1, year2=year2, config=config,
                            loglevel=loglevel, numproc=numproc, climatology=climatology,
                            interface=interface, model=model, ensemble=ensemble, silent=silent,
                            xdataset=xdataset, outputdir=outputdir, 
                            title=title)
    pi.prepare()
    pi.run()
    pi.store()
    pi.plot()
