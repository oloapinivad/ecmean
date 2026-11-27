#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" "
Global mean evaluation of CMIP6 climatology
It runs and then reads the output of the cmip6 models to provide an assessment of the
climatology for CMIP6 models to be stored
"""

__author__ = "Paolo Davini, Sep 2022."

import os
import subprocess
import warnings
import glob
from collections import defaultdict
import copy
import logging

import yaml
import numpy as np
import xarray as xr

from ecmean.performance_indices import PerformanceIndices, performance_indices
from ecmean.libs.files import load_yaml, make_input_filename
from ecmean.libs.formula import formula_variables
from ecmean.utils.utils import timeframe_years, parse_create_args, variance_iqr

warnings.simplefilter("ignore")

config_file = "config_create_clim.yml"
climdir = "../climatology/"

# CMIP6 coupled models (historical), one line per modelling centre. Selected from the esgpull archive
# as the models with at least 5 PIs over 1985-2014 and a land-sea mask (sftlf) on the atmospheric grid:
# models without sftlf (e.g. CAS-ESM2-0, HadGEM3-GC31-LL/MM, NESM3, UKESM1-1-LL) are left out
models = [
    "TaiESM1",  # AS-RCEC
    "AWI-CM-1-1-MR", "AWI-ESM-1-1-LR", "AWI-ESM-1-REcoM",  # AWI
    "BCC-CSM2-MR", "BCC-ESM1",  # BCC
    "FGOALS-f3-L", "FGOALS-g3",  # CAS
    "CanESM5", "CanESM5-CanOE",  # CCCma
    "CMCC-CM2-HR4", "CMCC-CM2-SR5", "CMCC-ESM2",  # CMCC
    "CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1",  # CNRM-CERFACS
    "ACCESS-ESM1-5", "ACCESS-CM2",  # CSIRO, CSIRO-ARCCSS
    "E3SM-1-0", "E3SM-1-1", "E3SM-1-1-ECA",  # E3SM-Project
    "EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-Veg", "EC-Earth3-Veg-LR",  # EC-Earth-Consortium
    "MPI-ESM-1-2-HAM",  # HAMMOZ-Consortium
    "INM-CM4-8", "INM-CM5-0",  # INM
    "IPSL-CM6A-LR",  # IPSL
    "MIROC6", "MIROC-ES2L",  # MIROC
    "UKESM1-0-LL",  # MOHC
    "ICON-ESM-LR", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",  # MPI-M
    "MRI-ESM2-0",  # MRI
    "GISS-E2-1-G",  # NASA-GISS
    "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2",  # NCAR
    "NorCPM1", "NorESM2-LM", "NorESM2-MM",  # NCC
    "GFDL-CM4", "GFDL-ESM4",  # NOAA-GFDL
    "SAM0-UNICON",  # SNU
]
# models = ['EC-Earth3']

# HighResMIP high-resolution coupled models (hist-1950), used for the -HR climatologies (e.g. EC26-CMIP-HR),
# one line per modelling centre. Selected from the esgpull archive as the high-resolution models with at least
# 5 PIs over 1985-2014. MPI-ESM1-2-XR has no land-sea mask (sftlf): it is kept for the PIs that do not
# need it, while tas, net_sfc, tauu and tauv are NaN. CESM1-CAM5-SE-HR has no sftlf either and, among
# the atmospheric fields, publishes psl only: it provides psl, tos and siconc. AWI-CM-1-1-HR publishes
# ocean data only (no atmosphere, no siconc) and provides tos and sos.
# The standard-resolution counterparts (e.g. CMCC-CM2-HR4, HadGEM3-GC31-MM, ECMWF-IFS-LR) are left out
highresmip_models = [
    "AWI-CM-1-1-HR",  # AWI
    "BCC-CSM2-HR",  # BCC
    "CESM1-CAM5-SE-HR",  # NCAR
    "CMCC-CM2-VHR4",  # CMCC
    "CNRM-CM6-1-HR",  # CNRM-CERFACS
    "EC-Earth3P-HR",  # EC-Earth-Consortium
    "ECMWF-IFS-HR",  # ECMWF
    "INM-CM5-H",  # INM
    "HadGEM3-GC31-HH", "HadGEM3-GC31-HM",  # MOHC, NERC
    "MPI-ESM1-2-XR",  # MPI-M
]

# models whose ensemble member or consortium differ from the default r1i1p1f1 and '*'
ENSEMBLES = {
    "CNRM-CM6-1": "r1i1p1f2",
    "CNRM-ESM2-1": "r1i1p1f2",
    "CanESM5-CanOE": "r1i1p2f1",
    "MIROC-ES2L": "r1i1p1f2",
    "UKESM1-0-LL": "r1i1p1f2",
    "AWI-CM-1-1-HR": "r1i1p1f2",
    "CNRM-CM6-1-HR": "r1i1p1f2",
    "EC-Earth3P-HR": "r2i1p2f1",
}
CONSORTIA = {"HadGEM3-GC31-HH": "NERC"}
# Three rules decide which performance indices enter the climatology. The PI files are kept as they are, and
# the reason of each exclusion is stored in the `excluded` entry of the climatology file.
# 1. errors in the published data: the tables below, as {model: {variable: reason}}, one per family
#    (a model may be in both, e.g. CNRM-CM6-1-HR, with different simulations);
# 2. incomplete period: less than MIN_YEARS_FRACTION of the years found in the archive (see years_available);
# 3. outliers: global annual PI above the upper whisker of the boxplot of log10(PI) of the models left
#    (see upper_whisker). Only the upper whisker is used: a very low PI is a good model, not an error.
MIN_YEARS_FRACTION = 0.8

CMIP6_EXCLUDE = {
    "CMCC-CM2-HR4": {"tauu": "wind stress with flipped sign", "tauv": "wind stress with flipped sign"},
    "CMCC-CM2-SR5": {"tauu": "wind stress with flipped sign", "tauv": "wind stress with flipped sign"},
    "CMCC-ESM2": {"tauu": "wind stress with flipped sign", "tauv": "wind stress with flipped sign"},
    "FGOALS-f3-L": {
        "tauu": "wind stress with flipped sign",
        "tauv": "wind stress with flipped sign",
    },
    "E3SM-1-1": {
        "ta": "undeclared fill value 9.97e36 below the ground",
        "ua": "undeclared fill value 9.97e36 below the ground",
        "va": "undeclared fill value 9.97e36 below the ground",
    },
    "E3SM-1-1-ECA": {"ta": "values close to zero below the ground"},
    "GISS-E2-1-G": {"hus": "undeclared fill value -1e27"},
    "CESM2": {"net_sfc": "prsn published over land only: the ocean-masked net_sfc is left with coastal cells"},
    "CESM2-FV2": {"net_sfc": "prsn published over land only: the ocean-masked net_sfc is left with coastal cells"},
    "CESM2-WACCM-FV2": {"net_sfc": "prsn published over land only: the ocean-masked net_sfc is left with coastal cells"},
}
HIGHRESMIP_EXCLUDE = {
    "CESM1-CAM5-SE-HR": {"sos": "unphysical values over about 7000 ocean cells"},
    "CMCC-CM2-VHR4": {"siconc": "undeclared fill values 1e20 in the coordinates and 1e30 in the data"},
}

# experiment, MIP, interface, models and exclusions for each climatology family (CMIP6 by default)
CMIP6_SETUP = {"expname": "historical", "mip": "CMIP", "interface": None, "models": models, "exclude": CMIP6_EXCLUDE}
HIGHRESMIP_SETUP = {
    "expname": "hist-1950",
    "mip": "HighResMIP",
    "interface": "CMIP6_highresmip",
    "models": highresmip_models,
    "exclude": HIGHRESMIP_EXCLUDE,
}


def years_available(var, face, diag):
    """Count the years of the period for which the input files of a variable are found.

    The files are those ecmean selects for the diagnostic. Their names are assumed to follow CMOR
    (``<var>_..._<first year>??-<last year>??.nc``): for a derived variable a year counts only if
    all its components are found.

    Args:
        var (str): the variable name.
        face (dict): the interface file.
        diag (Diagnostic): the diagnostic instance of the model.

    Returns:
        int: the number of years found, or None if no file is found (e.g. archive not reachable).
    """
    info = face["variables"][var]
    found = {name: set() for name in (formula_variables(info["derived"]) if "derived" in info else [var])}
    for filename in make_input_filename(var, face, diag):
        name, *_, dates = os.path.basename(filename).removesuffix(".nc").split("_")
        first, last = (int(date[:4]) for date in dates.split("-"))
        if name in found:
            found[name].update(range(first, last + 1))
    if not any(found.values()):
        return None
    return len(set(diag.years_joined).intersection(*found.values()))


def upper_whisker(values):
    """Upper whisker of the boxplot of log10(values), i.e. the third quartile plus 1.5 times the
    interquartile range, back in linear units. PIs are positive and close to log-normal, so the
    boxplot is drawn on their logarithm, as for the variance filter of the climatology (variance_iqr).

    Args:
        values (list): the PIs of the models.

    Returns:
        float: the largest PI not considered an outlier.
    """
    return float(variance_iqr(xr.DataArray(values))[2])


def code_date():
    """Date of the last commit touching the PI code, to spot PI files computed with an older version.

    Returns:
        int: seconds since the epoch, or None if the script does not run in a git checkout.
    """
    command = ["git", "log", "-1", "--format=%ct", "--", "../libs", "../performance_indices.py"]
    try:
        return int(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def main(
    clim="EC26",
    timeframe="CMIP",
    highresmip=False,
    nprocs=4,
    do_definitive=False,
    do_compute=True,
    do_create_clim=True,
    models=None,
):
    """
    Main function to compute the performance indices for the CMIP6 climatology
    and then create the new climatology file.
    Args:
        clim (str): Climatology name, e.g., 'EC26'.
        timeframe (str): Time period for the climatology, e.g., 'HIST',
                            'PDAY', or 'CMIP'.
        highresmip (bool): Whether to compute the reference PIs on the HighResMIP models
                            instead of the CMIP6 ones, creating the '-HR' climatology (e.g. EC26-CMIP-HR).
        nprocs (int): Number of processors to use for computation.
        do_definitive (bool): Whether to overwrite the definitive climatology file.
        do_compute (bool): Whether to compute the performance indices.
        do_create_clim (bool): Whether to create the climatology file after computation.
        models (list): Models whose performance indices are computed (default: all the models of the setup).
                            The climatology is always created averaging all the models of the setup.
    """

    climname = f"{clim}-{timeframe}-HR" if highresmip else f"{clim}-{timeframe}"
    year1, year2 = timeframe_years(timeframe)
    setup = HIGHRESMIP_SETUP if highresmip else CMIP6_SETUP
    expname = setup["expname"]

    # call the loop of global mean on all the models
    if do_compute:
        defaultconfig = load_yaml(config_file)

        for model in sorted(models or setup["models"]):
            logging.warning("Computing model %s", model)

            model_config = copy.deepcopy(defaultconfig)

            performance_indices(
                expname,
                year1,
                year2,
                config=model_config,
                model=model,
                ensemble=ENSEMBLES.get(model, "r1i1p1f1"),
                consortium=CONSORTIA.get(model),
                mip=setup["mip"],
                interface=setup["interface"],
                numproc=nprocs,
                climatology=climname,
                loglevel=logging.getLogger().getEffectiveLevel(),
                plot=False,
            )

    if do_create_clim:
        cfg = load_yaml(config_file)

        # dictionary with all elements
        full = {}
        excluded = {}
        stale = []
        newest_code = code_date()
        for model in setup["models"]:
            logging.warning("Climatology for model %s", model)
            ensemble = ENSEMBLES.get(model, "r1i1p1f1")
            filein = glob.glob(
                os.path.join(cfg["dirs"]["tab"], f"PI_{climname}_{expname}_{model}_{ensemble}_{year1}_{year2}.yml")
            )
            if not filein:
                logging.warning("No PI file found for model %s, skipping it", model)
                continue
            full[model] = load_yaml(filein[0])
            if newest_code and os.path.getmtime(filein[0]) < newest_code:
                stale.append(model)

            # rule 1: errors in the published data
            for var, reason in setup["exclude"].get(model, {}).items():
                if full[model].pop(var, None) is not None:
                    excluded.setdefault(var, {})[model] = reason

            # rule 2: incomplete period, from the files found in the archive
            diag = PerformanceIndices(
                expname,
                year1,
                year2,
                config=copy.deepcopy(cfg),
                model=model,
                ensemble=ensemble,
                consortium=CONSORTIA.get(model),
                mip=setup["mip"],
                interface=setup["interface"],
                climatology=climname,
                loglevel="ERROR",
            ).diag
            face = load_yaml(diag.interface)
            nyears = len(diag.years_joined)
            for var in [v for v in full[model] if not np.isnan(full[model][v]["ALL"]["Global"])]:
                found = years_available(var, face, diag)
                if found is None:
                    logging.warning("No input file found for %s of %s: the period cannot be checked", var, model)
                elif found < MIN_YEARS_FRACTION * nyears:
                    del full[model][var]
                    excluded.setdefault(var, {})[model] = f"only {found} of the {nyears} years available"

        if stale:
            logging.warning("PI files older than the last change to the PI code, recompute them if it matters: %s", stale)

        # rule 3: outliers, on the global annual PI of the models left for each variable
        for var in dict.fromkeys(v for model_data in full.values() for v in model_data):
            values = {model: model_data[var]["ALL"]["Global"] for model, model_data in full.items() if var in model_data}
            values = {model: value for model, value in values.items() if not np.isnan(value)}
            whisker = upper_whisker(list(values.values()))
            for model, value in values.items():
                if value > whisker:
                    del full[model][var]
                    excluded.setdefault(var, {})[model] = (
                        f"outlier: global PI {value:.1f} above the upper whisker ({whisker:.1f}) "
                        f"of the log10 boxplot of {len(values)} models"
                    )

        # union of the variables of all models, in order of appearance: a variable can be missing
        # in some models (e.g. no land-sea mask or no atmosphere), and it is then averaged on the others.
        # seasons and regions are taken from the first model providing the variable
        first = {}
        for model_data in full.values():
            for var, var_data in model_data.items():
                first.setdefault(var, var_data)

        # idiot averaging
        out = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for var, var_data in first.items():
            for season in var_data.keys():
                for region in var_data[season].keys():
                    element = []
                    for model, model_data in full.items():
                        if var in model_data:
                            if not np.isnan(model_data[var][season][region]):
                                element.append(model_data[var][season][region])
                    out[var][season][region] = float(round(np.mean(element), 3))

        # clumsy way to get the models for each var
        mout = {}
        for var in first:
            melement = []
            for model, model_data in full.items():
                if var in model_data:
                    if not np.isnan(model_data[var]["ALL"]["Global"]):
                        melement.append(model)
            mout[var] = melement

        # clim files
        pifile = os.path.join(climdir, climname, f"pi_climatology_{climname}.yml")
        if not do_definitive:
            update_pifile = os.path.join(climdir, climname, f"pi_climatology_{climname}_test.yml")
        else:
            update_pifile = os.path.join(climdir, climname, f"pi_climatology_{climname}.yml")
        piclim = load_yaml(pifile)

        # Update the climatology
        for var, season_data in out.items():
            piclim[var]["cmip6"] = {}
            for season, region_data in season_data.items():
                piclim[var]["cmip6"][season] = {}
                for region, value in region_data.items():
                    piclim[var]["cmip6"][season][region] = np.nan if value == 0 else float(value)
            piclim[var]["cmip6"]["models"] = mout[var]
            piclim[var]["cmip6"]["nmodels"] = len(mout[var])
            piclim[var]["cmip6"]["year1"] = year1
            piclim[var]["cmip6"]["year2"] = year2
            if var in excluded:
                piclim[var]["cmip6"]["excluded"] = excluded[var]
            # models of the family without a valid PI (no data, or a failed computation): listed so that
            # a model silently dropping out of the average is visible in the file
            missing = [m for m in setup["models"] if m not in mout[var] and m not in excluded.get(var, {})]
            if missing:
                piclim[var]["cmip6"]["missing"] = missing

        # dump the new file
        with open(update_pifile, "w", encoding="utf8") as file:
            yaml.safe_dump(piclim, file, sort_keys=False)


if __name__ == "__main__":
    parser = parse_create_args()
    parser.add_argument(
        "--highresmip",
        action="store_true",
        help="compute the reference PIs on the HighResMIP models, creating the -HR climatology (e.g. EC26-CMIP-HR)",
    )
    parser.add_argument(
        "--definitive", action="store_true", help="overwrite the definitive climatology file instead of creating a test one"
    )
    parser.add_argument(
        "--no-compute",
        dest="compute",
        action="store_false",
        help="do not compute the performance indices, just create the climatology file",
    )
    parser.add_argument(
        "--no-create-clim",
        dest="create_clim",
        action="store_false",
        help="create the climatology file after computing the performance indices",
    )

    parser.add_argument(
        "--models", nargs="+", help="compute the performance indices of these models only (the climatology uses all of them)"
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.loglevel.upper())
    main(
        clim=args.climdata,
        timeframe=args.timeframe,
        highresmip=args.highresmip,
        nprocs=args.cores,
        do_definitive=args.definitive,
        do_compute=args.compute,
        do_create_clim=args.create_clim,
        models=args.models,
    )
