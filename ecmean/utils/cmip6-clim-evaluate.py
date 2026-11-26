#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" "
Global mean evaluation of CMIP6 climatology
It runs and then reads the output of the cmip6 models to provide an assessment of the
climatology for CMIP6 models to be stored
"""

__author__ = "Paolo Davini, Sep 2022."

import os
import warnings
import glob
from collections import defaultdict
import copy
import logging

import yaml
import numpy as np

from ecmean.performance_indices import performance_indices
from ecmean.libs.files import load_yaml
from ecmean.utils.utils import timeframe_years, parse_create_args

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
# 5 PIs over 1985-2014. CESM1-CAM5-SE-HR and MPI-ESM1-2-XR have no land-sea mask (sftlf): they are kept
# for the PIs that do not need it (global variables and those on the ocean grid), while tas, net_sfc,
# tauu and tauv are NaN. AWI-CM-1-1-HR has no atmospheric data and provides the ocean PIs only.
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
# performance indices left out of the climatology for a specific model, as {model: {variable: reason}}, one table
# per family: a model may be in both (e.g. CNRM-CM6-1-HR), with different simulations and different ensembles
# to be compared with. The PI files are kept as they are, and the reasons are stored in the `excluded` entry
# of the climatology file
CMIP6_EXCLUDE = {
    # errors in the published CMIP6 data
    "CMCC-CM2-HR4": {"tauu": "wind stress with flipped sign", "tauv": "wind stress with flipped sign"},
    "CMCC-CM2-SR5": {"tauu": "wind stress with flipped sign", "tauv": "wind stress with flipped sign"},
    "CMCC-ESM2": {"tauu": "wind stress with flipped sign", "tauv": "wind stress with flipped sign"},
    "FGOALS-f3-L": {
        "tauu": "wind stress with flipped sign",
        "tauv": "wind stress with flipped sign",
        "siconc": "grid stored in reverse order with respect to areacello and tos",
    },
    "FGOALS-g3": {"siconc": "grid stored in reverse order with respect to areacello and tos"},
    "E3SM-1-1": {
        "ta": "undeclared fill value 9.97e36 below the ground",
        "ua": "undeclared fill value 9.97e36 below the ground",
        "va": "undeclared fill value 9.97e36 below the ground",
    },
    "E3SM-1-1-ECA": {"ta": "values close to zero below the ground"},
    "GISS-E2-1-G": {"hus": "undeclared fill value -1e27"},
    "EC-Earth3-Veg": {
        "hus": "only 11-16 of the 30 years available",
        "tauu": "only 11-16 of the 30 years available",
        "tauv": "only 11-16 of the 30 years available",
        "va": "only 11-16 of the 30 years available",
    },
}
HIGHRESMIP_EXCLUDE = {
    "CESM1-CAM5-SE-HR": {"sos": "unphysical values over about 7000 ocean cells"},
    "AWI-CM-1-1-HR": {"sos": "Baltic Sea 13-24 psu against 5-7 psu in ORAS5: 64 cells give 53% of the global PI"},
    "BCC-CSM2-HR": {"sos": "Red Sea up to 59 psu against 38 psu in ORAS5: 29 cells give 66% of the global PI"},
    "CNRM-CM6-1-HR": {
        "tas": "cold bias over land: PI 61 against 14-22 for the other models",
        "ta": "troposphere about 4 K colder than ERA5 (500 hPa global mean 254.4 K against 258.5 K): PI 197",
        "hus": "35-65% of the ERA5 humidity between 300 and 100 hPa: PI 80",
    },
    "INM-CM5-H": {
        "ta": "troposphere about 2 K warmer than ERA5 and tropopause biases (PI 137 at 150 hPa): PI 51",
        "hus": "30-50% more humid than ERA5 in the upper troposphere (PI 342 at 150 hPa): PI 72",
    },
    # errors in the published HighResMIP data
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
            for var, reason in setup["exclude"].get(model, {}).items():
                if full[model].pop(var, None) is not None:
                    excluded.setdefault(var, {})[model] = reason

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
