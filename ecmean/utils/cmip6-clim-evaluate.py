#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" "
Global mean evaluation of CMIP6 climatology
It runs and then reads the output of the cmip6 models to provide an assessment of the
climatology for CMIP6 models to be stored
"""

__author__ = "Paolo Davini (paolo.davini@cnr.it), Sep 2022."

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

# models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'TaiESM1', 'CanESM5', 'CESM2',
#          'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR', 'CMCC-CM2-SR5', 'NorESM2-MM', 'GFDL-CM4']
models = [
    "EC-Earth3",
    "IPSL-CM6A-LR",
    "FGOALS-g3",
    "CanESM5",
    "CESM2",
    "CNRM-CM6-1",
    "GISS-E2-1-G",
    "ACCESS-CM2",
    "CNRM-CM6-1",
    "SAM0-UNICON",
    "UKESM1-0-LL",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "AWI-CM-1-1-MR",
    "NorESM2-MM",
    "GFDL-CM4",
]
# models = ['EC-Earth3']

# models currently missing on the ESGF
# models= ['CMCC-CM2-SR5', 'TaiESM1']

# HighResMIP high-resolution coupled models (hist-1950), used for the -HR climatologies (e.g. EC26-CMIP-HR).
# Several of them do not provide fx/Ofx files: the CMIP6_highresmip interface uses
# a data file as gridfile, and the variables requiring the land-sea mask are skipped
highresmip_models = [
    "AWI-CM-1-1-HR",
    "BCC-CSM2-HR",
    "CESM1-CAM5-SE-HR",
    "CMCC-CM2-VHR4",
    "CNRM-CM6-1-HR",
    "EC-Earth3P-HR",
    "ECMWF-IFS-HR",
    "HadGEM3-GC31-HH",
    "INM-CM5-H",
    "MPI-ESM1-2-XR",
]

# experiment, MIP, interface and models for each climatology family (CMIP6 by default)
CMIP6_SETUP = {"expname": "historical", "mip": "CMIP", "interface": None, "models": models}
HIGHRESMIP_SETUP = {"expname": "hist-1950", "mip": "HighResMIP", "interface": "CMIP6_highresmip", "models": highresmip_models}

# models whose ensemble member or consortium differ from the default r1i1p1f1 and '*'
ENSEMBLES = {
    "CNRM-CM6-1": "r1i1p1f2",
    "UKESM1-0-LL": "r1i1p1f2",
    "AWI-CM-1-1-HR": "r1i1p1f2",
    "CNRM-CM6-1-HR": "r1i1p1f2",
    "EC-Earth3P-HR": "r1i1p2f1",
}
CONSORTIA = {"HadGEM3-GC31-HH": "NERC"}
# performance indices left out of the climatology for a specific model, because its output is biased
# (the PI files are kept as they are): CESM1-CAM5-SE-HR sos has about 7000 ocean cells with unphysical values,
# CNRM-CM6-1-HR and INM-CM5-H have 3D PIs several times larger than the other models (ta 197 and 51 against 8-19)
EXCLUDE = {
    "CESM1-CAM5-SE-HR": ["sos"],
    "CNRM-CM6-1-HR": ["ta", "hus"],
    "INM-CM5-H": ["ta", "hus"],
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
            for var in EXCLUDE.get(model, []):
                full[model].pop(var, None)

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
