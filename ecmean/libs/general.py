"""
Shared functions for XArray ecmean
"""

import logging
import math
import multiprocessing
import os
import platform
from time import time

import numpy as np
import pandas as pd
import xarray as xr

##################
# HELP FUNCTIONS #
##################


loggy = logging.getLogger(__name__)


class Timer:
    """Timer logging the time elapsed since the previous call and since the start."""

    def __init__(self, logger=None):
        self.logger = logger or loggy
        self.start_time = time()
        self.current_time = self.start_time

    def toc(self, message):
        """Update the timer and log the elapsed time."""
        elapsed_time = time() - self.current_time
        self.current_time = time()
        self.logger.info("%s time: %.2f seconds", message, elapsed_time)

    def final_toc(self):
        """Log the total elapsed time since the start."""
        self.logger.info("Total execution time: %.2f seconds", time() - self.start_time)


def set_multiprocessing_start_method():
    """Function to set the multiprocessing spawn method to fork"""
    plat = platform.system()
    if plat == "Windows":
        raise OSError("Windows does not support 'fork' start method.")
    if plat in ["Darwin", "Linux"]:
        multiprocessing.set_start_method("fork", force=True)
    else:
        raise OSError(f"Unsupported operative system {plat}")
    loggy.debug("Multiprocessing start method is %s", multiprocessing.get_start_method())
    return plat, multiprocessing.get_start_method()


def check_time_axis(xtime, years):
    """Check if we have 12 months per year and if the required years
    have been found in the NetCDF files."""

    # unique, counts = np.unique(xtime.dt.month, return_counts=True)
    # unique, counts = np.unique(xtime.time.resample(time='1M').mean(), return_counts=True)
    unique, counts = np.unique(xtime.time.dt.month, return_counts=True)
    if len(unique) != 12 or not all(counts == counts[0]):
        loggy.warning("Check your data: some months might be missing...")
        loggy.warning("Month counts: %s", counts)

    # apparently this is also satisfied by the file browsing
    set1 = set(years)
    set2 = set(xtime.dt.year.values)
    missing = list(set1.difference(set2))
    if missing:
        loggy.warning("Check your data: some years are missing")
        loggy.warning("Year missing: %s", missing)


def select_season(xfield, season):
    """Select the time steps of a season from an xarray object (lazily if dask-backed).

    Args:
        xfield (xarray.DataArray or xarray.Dataset): Field with a time axis.
        season (str): Season name (DJF, MAM, JJA, SON), or ALL for the whole record.

    Returns:
        xarray object: The field restricted to the season, unchanged for ALL.
    """
    if season == "ALL":
        return xfield
    return xfield.sel(time=xfield.time.dt.season == season)


SEASONS = ["DJF", "MAM", "JJA", "SON"]


def season_means(xfield, seasons):
    """Time mean of a field for each season (ALL for the whole record), in a single pass over the data.

    The record is loaded one block of time chunks at a time (see _time_blocks) and the seasonal sums and
    counts of valid points are accumulated, so memory holds one block plus the seasonal maps, whatever the
    length of the record: a single dask graph with one reduction per season would instead keep each chunk
    in memory until all the reductions have used it. ALL is the sum of the four seasons. Each mean is the
    plain mean of the available time steps: NaNs are skipped as in mean(), sums are accumulated in float64.

    Args:
        xfield (xarray.DataArray): Field with a time dimension, chunked on time.
        seasons (list): Season names among ALL, DJF, MAM, JJA and SON.

    Returns:
        xarray.DataArray: Time means with a season dimension, NaN where a season has no data.
    """
    # chunks of the data only: chunksizes would also look at the coordinates, which xarray may chunk
    # differently (e.g. the 2D lon/lat of large curvilinear grids) and raise an inconsistency error
    chunks = xfield.chunks[xfield.get_axis_num("time")] if xfield.chunks else (xfield.sizes["time"],)
    total = count = 0
    for tslice in _time_blocks(chunks):
        grouped = xfield.isel(time=tslice).load().groupby("time.season")
        total = total + grouped.sum("time").reindex(season=SEASONS, fill_value=0).astype("float64")
        count = count + grouped.count("time").reindex(season=SEASONS, fill_value=0)
        del grouped  # release the block before loading the next one, otherwise two blocks are in memory
    total, count = (xr.concat([stat.sum("season").expand_dims(season=["ALL"]), stat], "season") for stat in (total, count))
    return (total / count.where(count > 0)).sel(season=seasons)


def _time_blocks(chunks, minsize=12):
    """Slices of the time axis made of whole consecutive chunks, merged up to at least minsize steps
    (e.g. one file per month into one year); the last block can be shorter.

    Args:
        chunks (tuple): Sizes of the chunks along the time axis.
        minsize (int): Minimum number of time steps of a block.

    Yields:
        slice: The time steps of each block.
    """
    start = end = 0
    for size in chunks:
        end += size
        if end - start >= minsize:
            yield slice(start, end)
            start = end
    if start < end:
        yield slice(start, end)


def runtime_weights(varlist):
    """
    Define the weights to estimate the best repartition of the cores
    This is done a-priori, considering that
    1) compound variables are more difficult to compute
    2) 3d variables requires more evaluation
    """

    w = {}
    for k in varlist:
        if k in ["ua", "ta", "va", "hus"]:
            t = 8
        elif k in [
            "pme",
            "net_sfc_nosn",
            "net_sfc",
            "toamsfc_nosn",
            "toamsfc",
            "pr_oce",
            "pme_oce",
            "pr_land",
            "pme_land",
            "net_toa",
        ]:
            t = 3
        else:
            t = 1
        w[k] = t

    return w


def weight_split(a, n):
    """
    Use the weights by runtime_weights to provide a number of n chunks based on the
    minimum possible value of each chunk computing time. Then, provide the list of variables
    to be used in the multiprocessing routine
    """

    ws = sorted(runtime_weights(a).items(), key=lambda x: x[1], reverse=True)
    ordered = dict(ws)

    loads = [0] * n
    olists = [[] for _ in range(n)]

    # greedy longest-processing-time: assign each item to the least loaded bin
    for fff, value in ordered.items():
        count = loads.index(min(loads))
        loads[count] += value
        olists[count].append(fff)

    return olists


def check_workers(processes):
    """
    Check that all the worker processes ended successfully, after they have been joined.
    Workers catch the errors of each variable, so a non-zero exit code means that the
    process died anyway (e.g. killed by the OS) and its results are missing.

    Args:
        processes (list): List of (Process, list of variables) tuples.

    Raises:
        RuntimeError: If any worker exited with a non-zero exit code.
    """

    failed = [varlist for proc, varlist in processes if proc.exitcode != 0]
    if failed:
        raise RuntimeError(f"Worker processes failed while computing variables {failed}")


def check_var_interface(var, face):
    """Check if a var is defined in the interface file"""

    if var in face["variables"]:
        return True

    loggy.warning("%s is not defined in the interface file, skipping it!", var)
    return False


def check_var_climatology(varlist, reference):
    """Check if a var is defined in the climatology/reference file"""

    missing = [element for element in varlist if element not in reference]
    if missing:
        raise KeyError(f"Variable/Variables {missing} is/are not defined in the climatology, aborting!")


def get_domain(var, face):
    """Given a variable var extract its domain (oce or atm) from the interface.
    To do so it creates a dictionary providing the domain associated with a component.
    (the interface file specifies the component for each domain instead)"""

    comp = face["filetype"][face["variables"][var]["filetype"]]["component"]
    d = face["model"]["component"]
    domain = dict(zip([list(d.values())[x] for x in range(len(d.values()))], d.keys()))
    return domain[comp]


####################
# OUTPUT FUNCTIONS #
####################


def dict_to_dataframe(varstat, allowed=None):
    """
    Converts a nested 3-level dictionary to a pandas DataFrame.

    Parameters:
    varstat (dict): Nested dictionary with 3 levels.
    allowed (list): List of keys to be included in the DataFrame at the lower level.

    Returns:
    pd.DataFrame: Transformed DataFrame with hierarchical keys.
    """
    # Use set for O(1) lookup instead of O(n) list lookup
    if allowed is None:
        allowed = ["ALL", "DJF", "JJA", "MAM", "SON"]
    allowed_seasons = set(allowed)

    data_table = {}
    for var, seasons_dict in varstat.items():
        var_data = {}
        for season, regions_dict in seasons_dict.items():
            # Only process seasons that are in the allowed list
            if season in allowed_seasons:
                for region, value in regions_dict.items():
                    var_data[(season, region)] = value

        data_table[var] = var_data
    return pd.DataFrame(data_table).T


def write_tuning_table(linefile, varmean, var_table, diag, ref):
    """Write results appending one line to a text file.
    Write a tuning table: need to fix reference to face/ref.
    Variables missing from varmean (e.g. removed in OMIP/AMIP runs) are written as NaN,
    so that the columns stay aligned with the header across runs."""

    if not os.path.isfile(linefile):
        with open(linefile, "w", encoding="utf-8") as f:
            print("%model  ens  exp from   to ", end="", file=f)
            for var in var_table:
                print(f"{var:>12s}", end=" ", file=f)
            print("\n%                         ", end=" ", file=f)
            for var in var_table:
                print("{:>12s}".format(ref.get(var, {}).get("units", "")), end=" ", file=f)
            print(file=f)

    with open(linefile, "a", encoding="utf-8") as f:
        print(f"{diag.modelname} {diag.ensemble} {diag.expname}", f"{diag.year1:4d} {diag.year2:4d} ", end="", file=f)
        for var in var_table:
            value = varmean.get(var, {}).get("ALL", {}).get("Global", float("nan"))
            print("{:12.5f}".format(value * ref.get(var, {}).get("factor", 1)), end=" ", file=f)
        print(file=f)


def init_mydict(one, two):
    """Initialize a two level dictionary"""
    dd = {}
    for o in one:
        dd[o] = {}
        for t in two:
            dd[o][t] = float("NaN")

    return dd


def are_dicts_equal(dict1, dict2, tolerance=1e-6):
    """
    Recursively compare two dictionaries with a given tolerance for float comparisons.
    To be used for testing purposes
    """
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        if keys1 != keys2:
            return False

        for key in keys1:
            if not are_dicts_equal(dict1[key], dict2[key], tolerance):
                return False
        return True

    try:
        dict1 = float(dict1)
        dict2 = float(dict2)
        if math.isnan(dict1) and math.isnan(dict2):
            return True
        return math.isclose(dict1, dict2, rel_tol=tolerance)
    except ValueError:
        return dict1 == dict2
