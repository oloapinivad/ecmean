#!/usr/bin/env python3
"""
python3 version of ECmean performance indices tool
Using a reference file from yaml and cdo bindings

@author Paolo Davini (paolo.davini@cnr.it), 2022
@author Jost von Hardenberg (jost.hardenberg@polito.it), 2022
"""

__author__ = "Paolo Davini (paolo.davin@cnr.it)"

import logging
import os
import sys
from multiprocessing import Manager, Process

import dask
import numpy as np
import xarray as xr
import yaml

from ecmean import Diagnostic, Supporter, UnitsHandler
from ecmean.libs.areas import guess_bounds
from ecmean.libs.ecplotter import ECPlotter
from ecmean.libs.files import (
    get_clim_files,
    get_inifiles,
    load_output_yaml,
    load_variable,
    load_yaml,
)
from ecmean.libs.general import (
    Timer,
    check_var_climatology,
    check_workers,
    get_domain,
    init_mydict,
    set_multiprocessing_start_method,
    weight_split,
)
from ecmean.libs.loggy import setup_logger
from ecmean.libs.masks import mask_field, select_region
from ecmean.libs.ncfixers import adjust_clim_file, xr_preproc
from ecmean.libs.parser import parse_arguments
from ecmean.libs.units import apply_units_conversion, units_extra_definition

dask.config.set(scheduler="synchronous")

# same logger configured by setup_logger() in the workers
loggy = logging.getLogger("ecmean")


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
        climatology (str): Climatology to use. Default is 'EC26-CMIP'.
        interface (str): Path to the interface file.
        model (str): Model name.
        ensemble (str): Ensemble identifier. Default is 'r1i1p1f1'.
        xdataset (xarray.Dataset): Dataset to use.
        outputdir (str): Directory to store output files.
        extrafigure (bool): Special key for manual debugging, producing extra figures. Do not use.
        loggy (logging.Logger): Logger instance.
        diag (Diagnostic): Diagnostic instance.
        face (dict): Interface dictionary.
        piclim (dict): Climatology dictionary.
        util_dictionary (Supporter): Utility dictionary for remapping and masks.
        varstat (dict): Dictionary to store variable statistics.
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
        climatology=None,
        interface=None,
        model=None,
        ensemble="r1i1p1f1",
        xdataset=None,
        outputdir=None,
        extrafigure=False,
        title=None,
    ):
        """Initialize the PerformanceIndices class with the given parameters."""

        self.loglevel = loglevel
        self.loggy = setup_logger(level=self.loglevel)
        self.diag = Diagnostic(
            exp=exp,
            year1=year1,
            year2=year2,
            config=config,
            funcname=self.__class__.__name__,
            numproc=numproc,
            climatology=climatology,
            interface=interface,
            modelname=model,
            ensemble=ensemble,
            outputdir=outputdir,
            xdataset=xdataset,
        )
        self.face = None
        self.piclim = None
        self.util_dictionary = None
        self.varstat = None
        self.extrafigure = extrafigure  # special key to be set for manual debugging, producing extra figures: DO NOT USE
        self.outarray = None
        self.timer = Timer(self.loggy)
        self.title = title

    def toc(self, message):
        """Update the timer and log the elapsed time."""
        self.timer.toc(message)

    def final_toc(self):
        """Log the total elapsed time since the start."""
        self.timer.final_toc()

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
        comp = self.face["model"]["component"]  # Get component for each domain

        # all clim have the same grid, read from the first clim available and get target grid
        clim, _ = get_clim_files(self.piclim, "tas", self.diag, "ALL")
        target_remap_grid = xr.open_dataset(clim)

        # get file info files
        inifiles = get_inifiles(self.face, self.diag)

        # create remap dictionary with atm and oce interpolators
        self.util_dictionary = Supporter(
            comp,
            inifiles["atm"],
            inifiles["oce"],
            areas=False,
            remap=True,
            targetgrid=target_remap_grid,
        )

        # verify if we can run amip, omip or coupled run
        self.diag.configure_amip_omip_cpld(self.util_dictionary, self.piclim)

        self.toc("Preparation")

    def run(self):
        """Run the performance indices calculation.

        A variable that fails is logged and stored as NaN, without affecting the others.

        Raises:
            RuntimeError: If a worker process dies without completing its variables.
        """
        # main loop: manager is required for shared variables
        mgr = Manager()

        # dictionaries are shared, so they have to be passed as functions
        self.varstat = mgr.dict()
        processes = []

        # special dictionary for extra figures
        if self.extrafigure:
            self.loggy.debug("Extra figures requested, defining arrays")
            self.outarray = mgr.dict()
            for kind in ["bias", "map"]:
                self.outarray[kind] = mgr.dict()
        else:
            self.outarray = False

        # loop on the variables, create the parallel process
        for varlist in weight_split(self.diag.field_all, self.diag.numproc):
            core = Process(
                target=self.pi_worker,
                args=(
                    self.util_dictionary,
                    self.piclim,
                    self.face,
                    self.diag,
                    self.diag.field_atm3d,
                    self.varstat,
                    self.outarray,
                    varlist,
                    self.loglevel,
                ),
            )
            core.start()
            processes.append((core, varlist))

        # wait for the processes to finish
        for proc, _ in processes:
            proc.join()
        check_workers(processes)
        self.toc("Computation")

    def store(self, yamlfile=None):
        """Store the performance indices in a yaml file."""

        # order according to the original request the fields in the yaml file
        self.varstat = {var: self.varstat[var] for var in self.diag.field_all}

        # dump the yaml file for PI, including all the seasons (need to copy to avoid mess)
        if yamlfile is None:
            yamlfile = self.diag.filenames("yml")
        self.loggy.info("Storing the performance indices in %s", yamlfile)
        with open(yamlfile, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.varstat, file, default_flow_style=False, sort_keys=False)
        self.toc("Storing")

    def plot(
        self,
        diagname="performance_indices",
        mapfile=None,
        figformat="pdf",
        storefig=True,
        returnfig=False,
    ):
        """
        Generate the heatmap for performance indices.

        Args:
            diagname (str): Name of the diagnostic. Default is 'performance_indices'.
            mapfile (str): Path to the output file. If None, it will be defined automatically following ECmean syntax.
            figformat (str): Format of the output file. Default is 'pdf'.
            storefig (bool): If True, store the figure in the specified file. Default is True.
            returnfig (bool): If True, return the figure object. Default is False.
        """
        plotter = ECPlotter(
            diagnostic=diagname,
            modelname=self.diag.modelname,
            expname=self.diag.expname,
            year1=self.diag.year1,
            year2=self.diag.year2,
            regions=self.diag.regions,
            seasons=self.diag.seasons,
        )
        if self.varstat is None:
            self.varstat = load_output_yaml(self.diag.filenames("yml"))
        if mapfile is None:
            mapfile = self.diag.filenames(figformat)

        fig = plotter.heatmap_plot(
            data=self.varstat,
            base=self.piclim,
            variables=self.diag.field_all,
            climatology=self.diag.climatology,
            filename=mapfile,
            storefig=storefig,
            title=self.title,
        )

        self.toc("Plotting")

        if returnfig:
            self.loggy.info("Returning figure object")
            return fig

    @staticmethod
    def pi_worker(util, piclim, face, diag, field_3d, varstat, dictarray, varlist, loglevel):
        """
        Main parallel diagnostic worker for performance indices.
        Errors are caught per variable: a failing variable is logged and stored as NaN.

        Args:
            util (Supporter): Utility dictionary for remapping and masks.
            piclim (dict): Climatology dictionary.
            face (dict): Interface dictionary.
            diag (Diagnostic): Diagnostic instance.
            field_3d (list): List of 3D fields.
            varstat (dict): Dictionary to store variable statistics.
            dictarray (dict): Dictionary to store the output array.
            varlist (list): List of variables to process.
            loglevel (str): Logging level for the worker.
        """
        loggy = setup_logger(level=loglevel)

        # Local accumulation - avoid Manager().dict() overhead during computation
        local_varstat = {}

        # from python 3.14 this has to be into the worker
        units_extra_definition()

        for var in varlist:
            try:
                local_varstat[var], maps = _pi_variable(var, util, piclim, face, diag, field_3d)
            except Exception:
                # keep the other variables of this worker alive, store NaN for the failing one
                loggy.exception("Failed to compute performance indices for %s, results will be NaN", var)
                local_varstat[var], maps = init_mydict(diag.seasons, diag.regions), (None, None)

            # debug array for extrafigures
            if not isinstance(dictarray, bool):
                dictarray["map"][var], dictarray["bias"][var] = maps

        # store the local results into the shared dictionary
        varstat.update(local_varstat)


def _pi_variable(var, util, piclim, face, diag, field_3d):
    """
    Compute the performance indices of a single variable for all seasons and regions.

    Args:
        var (str): Variable name.
        util (Supporter): Utility dictionary for remapping and masks.
        piclim (dict): Climatology dictionary.
        face (dict): Interface dictionary.
        diag (Diagnostic): Diagnostic instance.
        field_3d (list): List of 3D fields.

    Returns:
        tuple: (result, (complete, bias)), with result the nested dictionary of PIs by season and region
            (NaN where not available), and the PI and bias maps of the last season for the extra figures.
    """
    result = init_mydict(diag.seasons, diag.regions)
    complete = final = cfield = None

    loaded = load_variable(var, face, diag)
    if loaded is None:
        return result, (None, None)
    field, varunit = loaded
    domain = get_domain(var, face)
    offset, factor = UnitsHandler(var, org_units=varunit, clim=piclim, face=face).units_converter()

    # Load once to avoid recomputing for each season
    outfield = field.persist()

    fix = getattr(util, f"{domain}fix")
    remap = getattr(util, f"{domain}remap")

    for season in diag.seasons:
        loggy.debug(season)
        cfield, vfield = _load_clim(piclim, var, diag, season)
        tmean = outfield if season == "ALL" else outfield.sel(time=outfield.time.dt.season.isin(season))

        # Single compute call for all three arrays
        tmean, cfield, vfield = dask.compute(
            apply_units_conversion(tmean.mean(dim="time"), offset, factor),
            cfield.mean(dim="time"),
            vfield.mean(dim="time"),
        )

        # apply interpolation, if fixer is available and with different grids
        if fix:
            tmean = fix(tmean, keep_attrs=True)
        try:
            final = remap(tmean, keep_attrs=True)
        except ValueError:
            loggy.error("Cannot interpolate %s with the current weights...", var)
            continue

        # 3D fields are compared as zonal means on the climatology pressure levels
        if var in field_3d:
            final = _match_vertical_levels(final, cfield, var).mean(dim="lon")

        # compute PI
        complete = (final - cfield) ** 2 / vfield

        # vertical mean for 3D fields, horizontal averaging with land-sea mask for 2D fields
        if var in field_3d:
            outarray = _vertical_mean(complete)
        else:
            outarray = mask_field(
                xfield=complete,
                mask_type=piclim[var]["mask"],
                dom=domain,
                mask=getattr(util, domain + "mask"),
            )

        # latitude-based averaging on the different regions
        for region in diag.regions:
            slicearray = select_region(outarray, region)
            weights = np.cos(np.deg2rad(slicearray.lat))
            result[season][region] = round(float(slicearray.weighted(weights).mean().data), 3)
            if region == "Global":
                loggy.info("PI for %s %s %s %s", region, season, var, result[season][region])

    bias = final - cfield if final is not None else None
    return result, (complete, bias)


def _load_clim(piclim, var, diag, season):
    """
    Open the climatology average and variance of a variable for a season.

    Args:
        piclim (dict): Climatology dictionary.
        var (str): Variable name.
        diag (Diagnostic): Diagnostic instance.
        season (str): Season name, ALL for the yearly climatology.

    Returns:
        tuple: (cfield, vfield) lazy climatology average and variance, restricted to the season.
    """
    # get filenames for climatology (season-specific files)
    clim, vvvv = get_clim_files(piclim, var, diag, season)
    loggy.debug("Climatology files for %s %s: %s, %s", var, season, clim, vvvv)

    # open climatology files, fix their metadata
    cfield = adjust_clim_file(xr.open_mfdataset(clim, preprocess=xr_preproc))
    vfield = adjust_clim_file(xr.open_mfdataset(vvvv, preprocess=xr_preproc), remove_zero=True)

    if season != "ALL":
        cfield = cfield.sel(time=cfield.time.dt.season.isin(season))
        vfield = vfield.sel(time=vfield.time.dt.season.isin(season))
    return cfield, vfield


def _match_vertical_levels(final, cfield, var):
    """
    Bring a 3D field on the pressure levels (Pa) of the climatology, interpolating if needed.

    Args:
        final (xarray.DataArray): Model field on the climatology horizontal grid.
        cfield (xarray.DataArray): Climatology field, providing the target pressure levels.
        var (str): Variable name, for logging.

    Returns:
        xarray.DataArray: Model field on the climatology pressure levels.
    """
    # xarray interpolation on plev, forcing to be in Pascal
    final = final.metpy.convert_coordinate_units("plev", "Pa")
    if set(final["plev"].data) == set(cfield["plev"].data):
        return final

    loggy.warning("%s: Need to interpolate vertical levels...", var)
    final = final.interp(plev=cfield["plev"].data, method="linear")

    # safety check for missing values
    sample = final.isel(lon=0, lat=0)
    nanlev = cfield["plev"].data[np.isnan(sample).data]
    if nanlev.size > 0:
        # number of missing points on the horizontal grid, for each affected level
        nanpts = np.isnan(final.sel(plev=nanlev)).sum(dim=["lon", "lat"]).data
        npts = final.sizes["lon"] * final.sizes["lat"]
        loggy.warning(
            "%s: NaN after vertical interpolation on %d of %d levels (Pa, missing points): %s. This will affect your PIs...",
            var,
            nanlev.size,
            cfield["plev"].size,
            ", ".join(f"{lev:g} ({n}/{npts})" for lev, n in zip(nanlev, nanpts)),
        )
    return final


def _vertical_mean(complete):
    """
    Pressure-weighted vertical mean, with the layer thickness of each level as weight (see #150).

    Args:
        complete (xarray.DataArray): Zonal-mean PI field on pressure levels.

    Returns:
        xarray.DataArray: Vertically averaged PI field.
    """
    bounds_lev = guess_bounds(complete["plev"], name="plev")
    bounds = abs(bounds_lev[:, 0] - bounds_lev[:, 1])
    www = xr.DataArray(bounds, coords=[complete["plev"]], dims=["plev"])
    return complete.weighted(www).mean(dim="plev")


def pi_entry_point():
    """
    Command line interface to run the performance indices calculation.
    """
    # read arguments from command line
    args = parse_arguments(sys.argv[1:], script="pi")

    performance_indices(
        exp=args.exp,
        year1=args.year1,
        year2=args.year2,
        numproc=args.numproc,
        loglevel=args.loglevel,
        climatology=args.climatology,
        interface=args.interface,
        config=args.config,
        model=args.model,
        ensemble=args.ensemble,
        outputdir=args.outputdir,
    )


def performance_indices(
    exp,
    year1,
    year2,
    config="config.yml",
    loglevel="WARNING",
    numproc=1,
    climatology=None,
    interface=None,
    model=None,
    ensemble="r1i1p1f1",
    xdataset=None,
    outputdir=None,
    title=None,
    plot=True,
):
    """
    Compute the performance indices, store the YAML file and produce the figure.

    Wrapper around :class:`PerformanceIndices`: it runs ``prepare()``, ``run()``, ``store()`` and, if requested, ``plot()``.

    Args:
        exp (str): Experiment name.
        year1 (int): Start year of the experiment.
        year2 (int): End year of the experiment.
        config (str or dict): Path to the configuration file or configuration dictionary. Default is 'config.yml'.
        loglevel (str): Logging level. Default is 'WARNING'.
        numproc (int): Number of processes to use. Default is 1.
        climatology (str): Climatology to use. Default is None (from configuration, then EC26-CMIP).
        interface (str): Interface name or path to the interface file. Default is None (from configuration).
        model (str): Model name. Default is None (from configuration).
        ensemble (str): Ensemble identifier. Default is 'r1i1p1f1'.
        xdataset (xarray.Dataset): Dataset to use instead of reading the files. Default is None.
        outputdir (str): Output directory. Default is None (from configuration).
        title (str): Title of the plot, overrides default title. Default is None.
        plot (bool): Whether to produce the figure after storing the YAML file. Default is True.
    """
    pi = PerformanceIndices(
        exp=exp,
        year1=year1,
        year2=year2,
        config=config,
        loglevel=loglevel,
        numproc=numproc,
        climatology=climatology,
        interface=interface,
        model=model,
        ensemble=ensemble,
        xdataset=xdataset,
        outputdir=outputdir,
        title=title,
    )
    pi.prepare()
    pi.run()
    pi.store()
    if plot:
        pi.plot()
    pi.final_toc()
