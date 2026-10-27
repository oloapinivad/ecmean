#!/usr/bin/env python3
"""
python3 version of ECmean global mean tool.
Using a reference file from yaml and Xarray

@author Paolo Davini (paolo.davini@cnr.it), Sep 2022.
@author Jost von Hardenberg (jost.hardenberg@polito.it), Sep 2022
"""

__author__ = "Paolo Davini (paolo.davini@cnr.it)"

import logging
import os
import sys
from multiprocessing import Manager, Process

import dask
import numpy as np
import xarray as xr
import yaml
from tabulate import tabulate

from ecmean import Diagnostic, Supporter, UnitsHandler
from ecmean.libs.ecplotter import ECPlotter
from ecmean.libs.files import get_inifiles, load_output_yaml, load_variable, load_yaml
from ecmean.libs.general import (
    Timer,
    check_var_climatology,
    check_workers,
    get_domain,
    init_mydict,
    set_multiprocessing_start_method,
    weight_split,
    write_tuning_table,
)
from ecmean.libs.loggy import setup_logger
from ecmean.libs.masks import masked_meansum, select_region
from ecmean.libs.parser import parse_arguments
from ecmean.libs.units import apply_units_conversion, units_extra_definition

dask.config.set(scheduler="synchronous")

# same logger configured by setup_logger() in the workers
loggy = logging.getLogger("ecmean")


class GlobalMean:
    """
    Class for computing global mean statistics from climate model output.

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
        trend (bool): Whether to compute trends. Default is None.
        line (str): Line identifier. Default is None.
        outputdir (str): Output directory. Default is None.
        xdataset (xarray.Dataset): Dataset to use instead of reading the files. Default is None.
        reference (str): Reference for the comparison (e.g. 'EC26-CMIP', 'EC26-HIST', 'EC26-PDAY').
            If None, the value in the configuration file or the default (EC26-CMIP) is used.
        loggy (logging.Logger): Logger instance.
        diag (Diagnostic): Diagnostic instance.
        face (dict): Interface dictionary.
        ref (dict): Reference dictionary.
        util_dictionary (Supporter): Supporter instance.
        varmean (dict): Dictionary to store variable means.
        vartrend (dict): Dictionary to store variable trends.
        funcname (str): Name of the class.
        timer (Timer): Timer logging the elapsed time.
        title (str): Title of the plot, overrides default title.
    """

    def __init__(
        self,
        exp,
        year1,
        year2,
        config="config.yml",
        loglevel="WARNING",
        numproc=1,
        interface=None,
        model=None,
        ensemble="r1i1p1f1",
        addnan=False,
        trend=None,
        line=None,
        outputdir=None,
        xdataset=None,
        reference=None,
        title=None,
    ):

        self.loglevel = loglevel
        self.loggy = setup_logger(level=self.loglevel)
        self.diag = Diagnostic(
            exp,
            year1,
            year2,
            config,
            funcname=self.__class__.__name__,
            numproc=numproc,
            ensemble=ensemble,
            interface=interface,
            modelname=model,
            addnan=addnan,
            trend=trend,
            line=line,
            outputdir=outputdir,
            xdataset=xdataset,
            reference=reference,
        )
        self.face = None
        self.ref = None
        self.util_dictionary = None
        self.varmean = None
        self.vartrend = None
        self.timer = Timer(self.loggy)
        self.title = title

    def toc(self, message):
        """Update the timer and log the elapsed time."""
        self.timer.toc(message)

    def final_toc(self):
        """Log the total elapsed time since the start."""
        self.timer.final_toc()

    def prepare(self):
        """Prepare the necessary components for the global mean computation."""
        plat, mprocmethod = set_multiprocessing_start_method()
        self.loggy.info('Running on %s and multiprocessing method set as "%s"', plat, mprocmethod)

        self.face = load_yaml(self.diag.interface)
        self.ref = load_yaml(self.diag.reffile)

        check_var_climatology(self.diag.var_all, self.ref.keys())

        os.makedirs(self.diag.tabdir, exist_ok=True)
        os.makedirs(self.diag.figdir, exist_ok=True)

        comp = self.face["model"]["component"]
        inifiles = get_inifiles(self.face, self.diag)

        self.util_dictionary = Supporter(comp, inifiles["atm"], inifiles["oce"], areas=True, remap=False)

        # verify if we can run amip, omip or coupled run
        self.diag.configure_amip_omip_cpld(self.util_dictionary, self.ref)

        self.toc("Preparation")

    def run(self):
        """Run the global mean computaacross all variables on using multiprocessing.

        A variable that fails is logged and stored as NaN, without affecting the others.

        Raises:
            RuntimeError: If a worker process dies without completing its variables.
        """
        mgr = Manager()
        self.varmean = mgr.dict()
        self.vartrend = mgr.dict()
        processes = []

        for varlist in weight_split(self.diag.var_all, self.diag.numproc):
            core = Process(
                target=self.gm_worker,
                args=(
                    self.util_dictionary,
                    self.ref,
                    self.face,
                    self.diag,
                    self.varmean,
                    self.vartrend,
                    varlist,
                    self.loglevel,
                ),
            )
            core.start()
            processes.append((core, varlist))

        for proc, _ in processes:
            proc.join()
        check_workers(processes)
        self.toc("Computation")

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
            if isinstance(gamma["obs"], dict):
                tabval = gamma["obs"]["ALL"]["Global"]
                outval = str(tabval["mean"]) + "\u00b1" + str(tabval["std"])
            else:
                outval = gamma["obs"]

            years = f"{gamma['year1']}-{gamma['year2']}"

            # prepare output sequence using list expansion if trend is requested
            out_sequence = [
                var,
                gamma["longname"],
                gamma["units"],
                self.varmean[var]["ALL"]["Global"],
                *([self.vartrend[var]["ALL"]["Global"]] if self.diag.ftrend else []),
                outval,
                gamma.get("dataset", ""),
                years,
            ]

            global_table.append(out_sequence)

        # create header using list expansion if trend is requested
        head = [
            "Variable",
            "Longname",
            "Units",
            self.diag.modelname,
            *(["Trend"] if self.diag.ftrend else []),
            "Obs.",
            "Dataset",
            "Years",
        ]

        # save table
        if tablefile is None:
            tablefile = self.diag.filenames("txt")

        self.loggy.info("TXT file is: %s", tablefile)
        with open(tablefile, "w", encoding="utf-8") as out:
            out.write(tabulate(global_table, headers=head, stralign="center", tablefmt="orgtbl"))

        # reorder
        self.varmean = {var: self.varmean[var] for var in self.diag.var_all}

        # save yaml file
        if yamlfile is None:
            yamlfile = self.diag.filenames("yml")

        self.loggy.info("YAML file is: %s", yamlfile)
        with open(yamlfile, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.varmean, file, default_flow_style=False, sort_keys=False)
        self.toc("Storing")

    def plot(
        self,
        diagname="global_mean",
        mapfile=None,
        figformat="pdf",
        storefig=True,
        returnfig=False,
        addnan=None,
    ):
        """
        Generate the heatmap for global mean.

        Args:
            diagname (str): Name of the diagnostic. Default is 'global_mean'.
            mapfile (str): Path to the output file. If None, it will be defined automatically following ECmean syntax.
            figformat (str): Format of the output file. Default is 'pdf'.
            storefig (bool): If True, store the figure in the specified file. Default is True.
            returnfig (bool): If True, return the figure object. Default is False.
            addnan (bool): If True, add variables without reference values to the plot.
                If None, use the value set at initialization. Default is None.
        """
        if addnan is None:
            addnan = self.diag.addnan
        plotter = ECPlotter(
            diagnostic=diagname,
            modelname=self.diag.modelname,
            expname=self.diag.expname,
            year1=self.diag.year1,
            year2=self.diag.year2,
            regions=self.diag.regions,
            seasons=self.diag.seasons,
        )
        if self.varmean is None:
            self.varmean = load_output_yaml(self.diag.filenames("yml"))
        if mapfile is None:
            mapfile = self.diag.filenames(figformat)

        fig = plotter.heatmap_plot(
            data=self.varmean,
            base=self.ref,
            variables=self.diag.var_all,
            filename=mapfile,
            storefig=storefig,
            addnan=addnan,
            title=self.title,
            reference=self.diag.reference,
        )
        if self.diag.ftable:
            self.loggy.info("Line file is: %s", self.diag.linefile)
            write_tuning_table(
                self.diag.linefile,
                self.varmean,
                self.diag.var_table,
                self.diag,
                self.ref,
            )

        self.toc("Plotting")

        if returnfig:
            self.loggy.info("Returning figure object")
            return fig

    @staticmethod
    def gm_worker(util, ref, face, diag, varmean, vartrend, varlist, loglevel):
        """ "
        Workhorse for the global mean computation.
        Errors are caught per variable: a failing variable is logged and stored as NaN.

        Args:
            util (Supporter): Utility dictionary for remapping and masks.
            ref (dict): Reference climatology dictionary.
            face (dict): Interface dictionary.
            diag (Diagnostic): Diagnostic instance.
            varmean (dict): Shared dictionary to store variable means.
            vartrend (dict): Shared dictionary to store variable trends.
            varlist (list): List of variables to process.
            loglevel (str): Logging level for the worker.
        """
        loggy = setup_logger(level=loglevel)

        # from python 3.14 this has to be into the worker
        units_extra_definition()

        # Local accumulation - avoid Manager().dict() overhead during computation
        local_varmean = {}
        local_vartrend = {}

        for var in varlist:
            try:
                local_varmean[var], local_vartrend[var] = _gm_variable(var, util, ref, face, diag)
            except Exception:
                # keep the other variables of this worker alive, store NaN for the failing one
                loggy.exception("Failed to compute global mean for %s, results will be NaN", var)
                local_varmean[var] = init_mydict(diag.seasons, diag.regions)
                local_vartrend[var] = init_mydict(diag.seasons, diag.regions)

        # OPTIMIZATION: Single bulk write to Manager.dict() at the end
        varmean.update(local_varmean)
        vartrend.update(local_vartrend)


def _gm_variable(var, util, ref, face, diag):
    """
    Compute the global mean (and trend) of a single variable for all seasons and regions.

    Args:
        var (str): Variable name.
        util (Supporter): Utility dictionary for areas and masks.
        ref (dict): Reference climatology dictionary.
        face (dict): Interface dictionary.
        diag (Diagnostic): Diagnostic instance.

    Returns:
        tuple: (result, trend) nested dictionaries by season and region, NaN where not available.
    """
    result = init_mydict(diag.seasons, diag.regions)
    trend = init_mydict(diag.seasons, diag.regions)

    loaded = load_variable(var, face, diag)
    if loaded is None:
        return result, trend
    field, varunit = loaded
    domain = get_domain(var, face)
    offset, factor = UnitsHandler(var, org_units=varunit, clim=ref, face=face).units_converter()

    # compute here since the operation are built on this
    cfield = field.compute()

    for season in diag.seasons:
        tfield = cfield if season == "ALL" else cfield.sel(time=cfield.time.dt.season.isin(season))
        tfield = tfield.groupby("time.year").mean("time") if diag.ftrend else tfield.mean(dim="time")

        for region in diag.regions:
            avg = _gm_region_mean(tfield, util, ref[var], domain, region)
            result[season][region] = float(apply_units_conversion(np.nanmean(avg), offset, factor))

            if diag.ftrend and len(avg) == len(diag.years_joined):
                trend[season][region] = np.polyfit(diag.years_joined, avg, 1)[0]
            if season == "ALL" and region == "Global":
                loggy.info("Average: %s %s %s %s", var, season, region, result[season][region])

    return result, trend


def _gm_region_mean(tfield, util, varref, domain, region):
    """
    Weighted (and masked) mean or sum of a field over a region, as defined in the reference.

    Args:
        tfield (xarray.DataArray): Time-averaged field (or yearly averages if trends are requested).
        util (Supporter): Utility dictionary for areas and masks.
        varref (dict): Reference entry of the variable, with operation and mask type.
        domain (str): Domain of the variable (atm or oce).
        region (str): Region name.

    Returns:
        numpy.ndarray: Regional average, one value per year if trends are requested.
    """
    domain_mask = getattr(util, domain + "mask")
    slicemask = select_region(domain_mask, region) if isinstance(domain_mask, xr.DataArray) else 0.0

    avg = masked_meansum(
        xfield=select_region(tfield, region),
        weights=select_region(getattr(util, domain + "area"), region),
        mask=slicemask,
        operation=varref.get("operation", "mean"),
        mask_type=varref.get("mask", "global"),
        domain=domain,
    )
    if isinstance(avg, dask.array.core.Array):
        avg = avg.compute()
    return avg


def gm_entry_point():
    """ "
    Entry point for the global mean computation.
    """
    args = parse_arguments(sys.argv[1:], script="gm")
    global_mean(
        exp=args.exp,
        year1=args.year1,
        year2=args.year2,
        numproc=args.numproc,
        trend=args.trend,
        line=args.line,
        loglevel=args.loglevel,
        interface=args.interface,
        config=args.config,
        model=args.model,
        ensemble=args.ensemble,
        addnan=args.addnan,
        outputdir=args.outputdir,
        reference=args.reference,
    )
    print("ECmean Global Mean successfully computed!")


def global_mean(
    exp,
    year1,
    year2,
    config="config.yml",
    loglevel="WARNING",
    numproc=1,
    interface=None,
    model=None,
    ensemble="r1i1p1f1",
    addnan=False,
    trend=None,
    line=None,
    outputdir=None,
    xdataset=None,
    reference=None,
    title=None,
):
    """
    Compute the global mean, store the YAML table and produce the figure.

    Wrapper around :class:`GlobalMean`: it runs ``prepare()``, ``run()``, ``store()`` and ``plot()``.

    Args:
        exp (str): Experiment name.
        year1 (int): Start year of the experiment.
        year2 (int): End year of the experiment.
        config (str or dict): Path to the configuration file or configuration dictionary. Default is 'config.yml'.
        loglevel (str): Logging level. Default is 'WARNING'.
        numproc (int): Number of processes to use. Default is 1.
        interface (str): Interface name or path to the interface file. Default is None (from configuration).
        model (str): Model name. Default is None (from configuration).
        ensemble (str): Ensemble identifier. Default is 'r1i1p1f1'.
        addnan (bool): Whether to plot also variables without reference. Default is False.
        trend (bool): Whether to compute trends. Default is None.
        line (bool): Whether to write the single-line tuning table. Default is None.
        outputdir (str): Output directory. Default is None (from configuration).
        xdataset (xarray.Dataset): Dataset to use instead of reading the files. Default is None.
        reference (str): Reference for the comparison, one of the supported global mean references.
            Default is None (from configuration, then EC26-CMIP).
        title (str): Title of the plot, overrides default title. Default is None.
    """
    gm = GlobalMean(
        exp,
        year1,
        year2,
        config,
        loglevel=loglevel,
        numproc=numproc,
        interface=interface,
        model=model,
        ensemble=ensemble,
        addnan=addnan,
        trend=trend,
        line=line,
        outputdir=outputdir,
        xdataset=xdataset,
        reference=reference,
        title=title,
    )
    gm.prepare()
    gm.run()
    gm.store()
    gm.plot()
    gm.final_toc()
