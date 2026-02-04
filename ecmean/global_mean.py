#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    python3 version of ECmean global mean tool.
    Using a reference file from yaml and Xarray

    @author Paolo Davini (p.davini@isac.cnr.it), Sep 2022.
    @author Jost von Hardenberg (jost.hardenberg@polito.it), Sep 2022
"""

__author__ = "Paolo Davini (p.davini@isac.cnr.it), Sep 2022."

import os
import sys
from multiprocessing import Process, Manager
from time import time
from tabulate import tabulate
import numpy as np
import xarray as xr
import yaml
import dask

from ecmean import Diagnostic, Supporter, UnitsHandler
from ecmean.libs.general import weight_split, write_tuning_table, get_domain, \
    check_time_axis, init_mydict, \
    check_var_interface, check_var_climatology, set_multiprocessing_start_method
from ecmean.libs.files import var_is_there, get_inifiles, load_yaml, make_input_filename, load_output_yaml
from ecmean.libs.formula import formula_wrapper
from ecmean.libs.masks import masked_meansum, select_region
from ecmean.libs.units import units_extra_definition
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.parser import parse_arguments
from ecmean.libs.ecplotter import ECPlotter
from ecmean.libs.loggy import setup_logger

dask.config.set(scheduler="synchronous")


class GlobalMean:
    """
    Attributes:
        exp (str): Experiment name.
        year1 (int): Start year of the experiment.
        year2 (int): End year of the experiment.
        config (str): Path to the configuration file. Default is 'config.yml'.
        loglevel (str): Logging level. Default is 'WARNING'.
        numproc (int): Number of processes to use. Default is 1.
        interface (str): Path to the interface file. Default is None.
        model (str): Model name. Default is None.
        ensemble (str): Ensemble identifier. Default is 'r1i1p1f1'.
        addnan (bool): Whether to add NaNs. Default is False.
        silent (bool): Whether to suppress output. Default is None.
        trend (bool): Whether to compute trends. Default is None.
        line (str): Line identifier. Default is None.
        outputdir (str): Output directory. Default is None.
        xdataset (str): Path to the xdataset. Default is None.
        loggy (logging.Logger): Logger instance.
        diag (Diagnostic): Diagnostic instance.
        face (dict): Interface dictionary.
        ref (dict): Reference dictionary.
        util_dictionary (Supporter): Supporter instance.
        varmean (dict): Dictionary to store variable means.
        vartrend (dict): Dictionary to store variable trends.
        funcname (str): Name of the class.
        start_time (float): Start time for the timer.
        title (str): Title of the plot, overrides default title.
    Methods:
        toc(message):
            Update the timer and log the elapsed time.
        prepare():
            Prepare the necessary components for the global mean computation.
        run():
            Run the global mean computation using multiprocessing.
        store():
            Store the computed global mean values in a table and YAML file.
        plot(mapfile=None, figformat='pdf'):
        gm_worker(util, ref, face, diag, varmean, vartrend, varlist):
    """

    def __init__(self, exp, year1, year2, config='config.yml', loglevel='WARNING', numproc=1,
                 interface=None, model=None, ensemble='r1i1p1f1', addnan=False, silent=None,
                 trend=None, line=None, outputdir=None, xdataset=None, reference='EC23', title=None):

        self.loglevel = loglevel
        self.loggy = setup_logger(level=self.loglevel)
        self.diag = Diagnostic(exp, year1, year2, config,
                               funcname=self.__class__.__name__,
                               numproc=numproc, ensemble=ensemble, interface=interface,
                               modelname=model, addnan=addnan, silent=silent,
                               trend=trend, line=line, outputdir=outputdir,
                               xdataset=xdataset, reference=reference)
        self.face = None
        self.ref = None
        self.util_dictionary = None
        self.varmean = None
        self.vartrend = None
        self.start_time = time()
        self.title = title

    def toc(self, message):
        """Update the timer and log the elapsed time."""
        elapsed_time = time() - self.start_time
        self.start_time = time()
        self.loggy.info('%s time: %.2f seconds', message, elapsed_time)

    def prepare(self):
        """Prepare the necessary components for the global mean computation."""
        plat, mprocmethod = set_multiprocessing_start_method()
        self.loggy.info('Running on %s and multiprocessing method set as "%s"', plat, mprocmethod)

        self.face = load_yaml(self.diag.interface)
        self.ref = load_yaml(self.diag.reffile)

        check_var_climatology(self.diag.var_all, self.ref.keys())

        os.makedirs(self.diag.tabdir, exist_ok=True)
        os.makedirs(self.diag.figdir, exist_ok=True)

        comp = self.face['model']['component']
        inifiles = get_inifiles(self.face, self.diag)

        self.util_dictionary = Supporter(
            comp, inifiles['atm'], inifiles['oce'], areas=True, remap=False
        )

        # verify if we can run amip, omip or coupled run
        self.diag.configure_amip_omip_cpld(self.util_dictionary)

        self.toc('Preparation')

    def run(self):
        """Run the global mean computaacross all variables on using multiprocessing."""
        mgr = Manager()
        self.varmean = mgr.dict()
        self.vartrend = mgr.dict()
        processes = []

        for varlist in weight_split(self.diag.var_all, self.diag.numproc):
            core = Process(target=self.gm_worker, args=(self.util_dictionary,
                                                        self.ref,
                                                        self.face,
                                                        self.diag,
                                                        self.varmean,
                                                        self.vartrend,
                                                        varlist,
                                                        self.loglevel))
            core.start()
            processes.append(core)

        for proc in processes:
            proc.join()
        self.toc('Computation')

    def store(self, yamlfile=None, tablefile=None):
        """
        Rearrange the data and save the yaml file and the table.
        Args:
            yamlfile: Path to the output YAML file. If None, it will be defined automatically.
            tablefile: Path to the output TXT file. If None, it will be defined automatically.
        """
        global_table = []

        # reorder the data to be stored
        for var in self.diag.var_all:
            gamma = self.ref[var]

            # if dictionary we have more info season, mean and standard deviation
            if isinstance(gamma['obs'], dict):
                tabval = gamma['obs']['ALL']['Global']
                outval = str(tabval['mean']) + '\u00B1' + str(tabval['std'])
            else:
                outval = gamma['obs']

            years = f"{gamma['year1']}-{gamma['year2']}"

            # prepare output sequence using list expansion if trend is requested
            out_sequence = [
                var,
                gamma['longname'],
                gamma['units'],
                self.varmean[var]['ALL']['Global'],
                *([self.vartrend[var]['ALL']['Global']] if self.diag.ftrend else []),
                outval,
                gamma.get('dataset', ''),
                years
            ]

            global_table.append(out_sequence)

        # create header using list expansion if trend is requested
        head = [
            'Variable', 
            'Longname', 
            'Units', 
            self.diag.modelname,
            *(['Trend'] if self.diag.ftrend else []),
            'Obs.', 
            'Dataset', 
            'Years'
        ]

        # save table
        if tablefile is None:
            tablefile = self.diag.filenames('txt')
        
        self.loggy.info('TXT file is: %s', tablefile)
        with open(tablefile, 'w', encoding='utf-8') as out:
            out.write(tabulate(global_table, headers=head, stralign='center', tablefmt='orgtbl'))

        # reorder
        self.varmean = {var: self.varmean[var] for var in self.diag.var_all}

        # save yaml file
        if yamlfile is None:
            yamlfile = self.diag.filenames('yml')

        self.loggy.info('YAML file is: %s', yamlfile)
        with open(yamlfile, 'w', encoding='utf-8') as file:
            yaml.safe_dump(self.varmean, file, default_flow_style=False, sort_keys=False)
        self.toc('Storing')

    def plot(self, diagname="global_mean", mapfile=None, figformat='pdf', storefig=True, returnfig=False, addnan=True):
        """
        Generate the heatmap for global mean.

        Args:
            diagname (str): Name of the diagnostic. Default is 'global_mean'.
            mapfile (str): Path to the output file. If None, it will be defined automatically following ECmean syntax.
            figformat (str): Format of the output file. Default is 'pdf'.
            storefig (bool): If True, store the figure in the specified file. Default is True.
            returnfig (bool): If True, return the figure object. Default is False.
            addnan (bool): If True, add NaN values to the plot. Default is True.
        """
        plotter = ECPlotter(
            diagnostic=diagname, modelname=self.diag.modelname,
            expname=self.diag.expname, year1=self.diag.year1,
            year2=self.diag.year2, regions=self.diag.regions,
            seasons=self.diag.seasons)
        if self.varmean is None:
            self.varmean = load_output_yaml(self.diag.filenames('yml'))
        if mapfile is None:
            mapfile = self.diag.filenames(figformat)

        fig = plotter.heatmap_plot(
            data=self.varmean, base=self.ref,
            variables=self.diag.var_all,
            filename=mapfile, storefig=storefig, addnan=addnan, 
            title=self.title, reference=self.diag.reference
        )
        if self.diag.ftable:
            self.loggy.info('Line file is: %s', self.diag.linefile)
            write_tuning_table(self.diag.linefile, self.varmean, self.diag.var_table, self.diag, self.ref)
        
        self.toc('Plotting')

        if returnfig:
            self.loggy.info('Returning figure object')
            return fig

    # def plot(self, mapfile=None, figformat='pdf'):
    #     """"
    #     Plot the global mean values.
    #     Args:
    #         mapfile: Path to the output file. If None, it will be defined automatically following ECmean syntax
    #         figformat: Format of the output file.
    #     """

    #     # load yaml file if is missing
    #     if not self.varmean:
    #         yamlfile = self.diag.filenames('yml')
    #         self.loggy.info('Loading the stored data from the yaml file %s', yamlfile)
    #         if os.path.isfile(yamlfile):
    #             with open(yamlfile, 'r', encoding='utf-8') as file:
    #                 self.varmean = yaml.safe_load(file)
    #         else:
    #             raise FileNotFoundError(f'YAML file {yamlfile} not found')

    #     # prepare the dictionaries for the plotting
    #     obsmean, obsstd, data2plot, units_list = prepare_clim_dictionaries_gm(self.varmean, self.ref,
    #                                                                           self.diag.var_all, self.diag.seasons,
    #                                                                           self.diag.regions)
    #     if mapfile is None:
    #         mapfile = self.diag.filenames(figformat)
    #     self.loggy.info('Figure file is: %s', mapfile)

    #     # call the heatmap for plottinh
    #     heatmap_comparison_gm(data_dict=data2plot, mean_dict=obsmean, std_dict=obsstd,
    #                           diag=self.diag, units_list=units_list,
    #                           filemap=mapfile, addnan=self.diag.addnan)



    @staticmethod
    def gm_worker(util, ref, face, diag, varmean, vartrend, varlist, loglevel):
        """"
        Workhorse for the global mean computation.

        Args:
            util (Supporter): Utility dictionary for remapping and masks.
            ref (dict): Reference climatology dictionary.
            face (dict): Interface dictionary.
            diag (Diagnostic): Diagnostic instance.
            varmean (dict): Shared dictionary to store variable means.
            vartrend (dict): Shared dictionary to store variable trends.
            varlist (dict): List of variables to process.
            varlist (list): List of variables to process.
        """
        loggy = setup_logger(level=loglevel)

        # from python 3.14 this has to be into the worker
        units_extra_definition()

        # Local accumulation - avoid Manager().dict() overhead during computation
        local_varmean = {}
        local_vartrend = {}

        for var in varlist:
            result = init_mydict(diag.seasons, diag.regions)
            trend = init_mydict(diag.seasons, diag.regions)

            if check_var_interface(var, face):
                domain = get_domain(var, face)
                weights = getattr(util, domain + 'area')
                domain_mask = getattr(util, domain + 'mask')
                infile = make_input_filename(var, face, diag)
                isavail, varunit = var_is_there(infile, var, face)

                if isavail:
                    offset, factor = UnitsHandler(var, org_units=varunit,
                                                  clim=ref, face=face).units_converter()

                    if not isinstance(infile, (xr.DataArray, xr.Dataset)):
                        xfield = xr.open_mfdataset(
                            infile, preprocess=xr_preproc, chunks={'time': 12},
                            data_vars='all', combine='by_coords', 
                            compat='no_conflicts')
                    else:
                        xfield = infile

                    xfield = xfield.sel(time=xfield.time.dt.year.isin(diag.years_joined))
                    check_time_axis(xfield.time, diag.years_joined)

                    # compute here since the operation are built on this
                    cfield = formula_wrapper(var, face, xfield).compute()

                    for season in diag.seasons:
                        tfield = cfield.copy(deep=True)
                        if season != 'ALL':
                            tfield = tfield.sel(time=cfield.time.dt.season.isin(season))

                        if diag.ftrend:
                            tfield = tfield.groupby('time.year').mean('time')
                        else:
                            tfield = tfield.mean(dim='time')

                        for region in diag.regions:
                            slicefield = select_region(tfield, region)
                            sliceweights = select_region(weights, region)
                            if isinstance(domain_mask, xr.DataArray):
                                slicemask = select_region(domain_mask, region)
                            else:
                                slicemask = 0.

                            avg = masked_meansum(
                                xfield=slicefield, weights=sliceweights, mask=slicemask,
                                operation=ref[var].get('operation', 'mean'),
                                mask_type=ref[var].get('mask', 'global'),
                                domain=domain)

                            if isinstance(avg, dask.array.core.Array):
                                avg = avg.compute()

                            result[season][region] = float((np.nanmean(avg) + offset) * factor)

                            if diag.ftrend:
                                if len(avg) == len(diag.years_joined):
                                    trend[season][region] = np.polyfit(diag.years_joined, avg, 1)[0]
                            if season == 'ALL' and region == 'Global':
                                loggy.info('Average: %s %s %s %s', var, season, region, result[season][region])

            local_varmean[var] = result
            local_vartrend[var] = trend

        # OPTIMIZATION: Single bulk write to Manager.dict() at the end
        varmean.update(local_varmean)
        vartrend.update(local_vartrend)


def gm_entry_point():
    """"
    Entry point for the global mean computation.
    """
    args = parse_arguments(sys.argv[1:], script='gm')
    global_mean(exp=args.exp, year1=args.year1, year2=args.year2, numproc=args.numproc,
                trend=args.trend, line=args.line, loglevel=args.loglevel,
                interface=args.interface, config=args.config, model=args.model,
                ensemble=args.ensemble, addnan=args.addnan, outputdir=args.outputdir,
                reference=args.reference)
    print('ECmean4 Global Mean successfully computed!')


def global_mean(exp, year1, year2, config='config.yml', loglevel='WARNING', numproc=1,
                interface=None, model=None, ensemble='r1i1p1f1', addnan=False, silent=None,
                trend=None, line=None, outputdir=None, xdataset=None, reference=None, title=None):
    """Wrapper function to compute the global mean."""
    gm = GlobalMean(exp, year1, year2, config,
                    loglevel=loglevel, numproc=numproc, interface=interface, model=model,
                    ensemble=ensemble, addnan=addnan, silent=silent, trend=trend,
                    line=line, outputdir=outputdir, xdataset=xdataset, reference=reference, title=title)
    gm.prepare()
    gm.run()
    gm.store()
    gm.plot()
