"""Climatology/reference routines."""

import os
import logging
import argparse
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import medcouple
from ecmean.libs.climatology import CLIMATOLOGY_PREFIXES

loggy = logging.getLogger(__name__)


def get_climatology_files(tgtdir, var, dataset, grid, year1, year2):
    """Generate climatology file paths."""

    suffix = f"{var}_{dataset}_{grid}_{year1}-{year2}.nc"

    climatology_files = {prefix: os.path.join(tgtdir, grid, prefix + suffix) for prefix in CLIMATOLOGY_PREFIXES}

    return climatology_files


def timeframe_years(timeframe):
    """Timeframe to years mapping."""
    if timeframe == "HIST":
        year1 = 1981
        year2 = 2010
    elif timeframe == "PDAY":
        year1 = 2000
        year2 = 2024
    elif timeframe == "CMIP":
        year1 = 1985
        year2 = 2014
    else:
        raise ValueError(f"TIMEFRAME {timeframe} not recognized.Please choose from HIST, PDAY, CMIP.")
    return year1, year2


def select_time_data(yearly, seasonly, season):
    """Select time data based on season.

    Args:
        yearly (xarray.DataArray): yearly means (resample "YS").
        seasonly (xarray.DataArray): seasonal means (resample "QE-NOV").
        season (str): "ALL", "DJF", "MAM", "JJA" or "SON".

    Returns:
        xarray.DataArray: the yearly means for "ALL", otherwise the means of the season.
    """

    if season == "ALL":
        return yearly

    gfield = seasonly.sel(time=seasonly.time.dt.season.isin(season))
    # for winter, we drop first and last to have only complete season.
    # this reduces the sample by one but it is safer for variance.
    # It assumes data from January of the first year to December of the last year,
    # as guaranteed by select_time_period(): first DJF is Jan-Feb only, last DJF is Dec only.
    if season == "DJF":
        gfield = gfield.drop_isel(time=[0, gfield.sizes["time"] - 1])

    return gfield


def expand_filedata(directory, var, info):
    """Expand filedata directory with environment variables and info dictionary."""

    return os.path.expandvars(directory).format(
        datadir=info["dirs"]["datadir"],
        mswepdir=info["dirs"]["mswepdir"],
        eradir=info["dirs"]["eradir"],
        esadir=info["dirs"]["esadir"],
        dataset=info[var]["dataset"],
    )


def select_time_period(xfield, var, year1, year2):
    """
    Select time period based on data availability, using complete years only.

    A year is complete when it has valid data in all 12 months. Incomplete years at the
    start or end of the record (e.g. CERES-EBAF starting in March 2000) are excluded, so
    that the yearly means are not computed on a partial year and the first/last DJF
    seasons dropped by select_time_data() are exactly the incomplete ones.
    Incomplete years inside the period are kept, with a warning.

    Args:
        xfield (xarray.Dataset): dataset containing the variable.
        var (str): variable name.
        year1 (int): first requested year.
        year2 (int): last requested year.

    Returns:
        tuple: selected DataArray, effective first year and effective last year.
    """
    # Check available complete years (12 months with valid data)
    valid_time = xfield[var].dropna("time", how="all").time
    years = valid_time.dt.year.values
    months = valid_time.dt.month.values
    avail_years = np.unique(years)
    complete_years = np.array([y for y in avail_years if np.unique(months[years == y]).size == 12])
    if complete_years.size == 0:
        raise ValueError(f"{var}: no complete year available in the data ({avail_years.min()}-{avail_years.max()})")

    # Calculate intersection between requested and available
    real_year1 = max(year1, int(complete_years.min()))
    real_year2 = min(year2, int(complete_years.max()))

    # Warn about incomplete years inside the selected period
    incomplete = [int(y) for y in avail_years if real_year1 <= y <= real_year2 and y not in complete_years]
    if incomplete:
        logging.warning("%s: incomplete years kept inside the period: %s", var, incomplete)

    # Raise error if incompatible
    if real_year1 > real_year2:
        raise ValueError(
            f"{var}: requested period {year1}-{year2} "
            f"not compatible with complete years available "
            f"{complete_years.min()}-{complete_years.max()}"
        )

    # Warn if years differ from requested
    if real_year1 != year1 or real_year2 != year2:
        logging.warning("%s: using %s-%s instead of requested %s-%s", var, real_year1, real_year2, year1, year2)

    # Select time
    years_eff = list(range(real_year1, real_year2 + 1))
    cfield = xfield[var].sel(time=xfield[var].time.dt.year.isin(years_eff))

    return cfield, real_year1, real_year2


def variance_threshold(xvariance, sigma=3):
    """
    This defines the two thresholds (high and low) for filtering the dataset
    So far it is done on the 3-std of the log10 distribution
    """
    f = np.log10(xvariance.where(xvariance > 0))
    m = float(f.mean())
    loggy.debug("Median variance: %s", m)
    s = float(f.std())
    low = 10 ** (m - sigma * s)
    high = 10 ** (m + sigma * s)
    m = 10**m
    return low, m, high


def variance_iqr_adjusted(xvariance):
    """This defines the two thresholds (high and low) for filtering the dataset
    based on the interquartile range of the log10 distribution applying
    the medcouple adjustment for skewness from Hubert, M. & Vandervieren, E. (2008)"""

    f = np.log10(xvariance.where(xvariance > 0))
    median = float(f.median())
    qqq = f.quantile([0.25, 0.75])
    mc = medcouple(f.values[~np.isnan(f.values)])
    loggy.info("Medcouple: %s", mc)
    iqr = qqq[1] - qqq[0]
    f1, f2 = (-4, 3) if mc > 0 else (-3, 4)
    iqleft = 10 ** (qqq[0] - 1.5 * iqr * np.exp(f1 * mc))
    iqright = 10 ** (qqq[1] + 1.5 * iqr * np.exp(f2 * mc))

    return iqleft, 10**median, iqright


def variance_combined(xvariance, fraction=3, sigma=3):
    """
    This defines the two thresholds (high and low) for filtering the dataset
    based on the combination of the 3-sigma method and the fraction of the median in
    log10 space. The final thresholds are defined as the maximum of the low thresholds
    and the minimum of the high thresholds, to avoid too wide thresholds in case of low variance fields.
    """

    f = np.log10(xvariance.where(xvariance > 0))
    median = f.median().values
    s = f.std().values
    sigmaleft = 10 ** (median - sigma * s)
    sigmaright = 10 ** (median + sigma * s)
    loggy.info("Sigma thresholds: low = %s, high = %s", sigmaleft, sigmaright)
    varleft = 10 ** (median - fraction)
    varright = 10 ** (median + fraction)
    loggy.info("Variance fraction: low = %s, high = %s", varleft, varright)
    left = max(sigmaleft, varleft)
    right = min(sigmaright, varright)

    return left, 10**median, right


def variance_iqr(xvariance):
    """This defines the two thresholds (high and low) for filtering the dataset
    based on the interquartile range of the log10 distribution."""

    f = np.log10(xvariance.where(xvariance > 0))
    median = float(f.median())
    qqq = f.quantile([0.25, 0.75])
    iqr = qqq[1] - qqq[0]
    iqleft = 10 ** (qqq[0] - 1.5 * iqr)
    iqright = 10 ** (qqq[1] + 1.5 * iqr)
    return iqleft.values, 10**median, iqright.values


def variance_fraction(xvariance, fraction=3):
    """
    Alternative method for variance clipping, based on the median of the distribution.
    The threshold is defined as a fraction of the median.
    """
    f = np.log10(xvariance.where(xvariance > 0))
    m = float(f.median())
    loggy.debug("Median variance: %s", m)
    low = 10 ** (m - fraction)
    high = 10 ** (m + fraction)
    return low, 10**m, high


def full_histogram(field, figname, n_bins=100):
    """Compute the histogram of the full field before it is processed.
    this is done to check the presence of irrealistic values within the dataset
    dask.array.histogram is used to speed up the computation."""

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(15, 5))

    # test using underlying dask array
    mmm = da.nanmin(field.data).compute()
    xxx = da.nanmax(field.data).compute()
    extra = (xxx - mmm) / 20
    hist, bins = da.histogram(field.data, bins=n_bins, range=[mmm - extra, xxx + extra])
    x = 0.5 * (bins[1:] + bins[:-1])
    width = np.diff(bins)
    axs.bar(x, hist.compute(), width, log=True)
    axs.title.set_text("Complete original values " + field.name)
    fig.savefig(figname)


def check_histogram(yvar, yvar_filtered, figname, n_bins=100, method="sigma", center=None, low=None, high=None):
    """Four histograms made for inspection of mean and variance of the field
    Mean field, variance and variance after filtering are passed and then plotted
    using histograms. log10 scale is used to highlight outliers."""

    fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True, figsize=(15, 10))

    # log10 fields
    f = np.log10(yvar.where(yvar > 0))
    g = np.log10(yvar_filtered.where(yvar_filtered > 0))

    # Convert thresholds to log10 scale for plotting
    center_log = np.log10(center) if center is not None and center > 0 else None
    low_log = np.log10(low) if low is not None and low > 0 else None
    high_log = np.log10(high) if high is not None and high > 0 else None

    # xlim in log10 scale
    left = np.min([low_log if low_log is not None else f.min(skipna=True), f.min(skipna=True)])
    right = np.max([high_log if high_log is not None else f.max(skipna=True), f.max(skipna=True)])
    extra = abs(left - right) / 20

    # mean and variance field
    yvar.plot.hist(ax=axs[0], bins=n_bins, yscale="log", color="goldenrod")
    axs[0].title.set_text("Original variance " + yvar.name)

    color = (
        "magenta" if method == "sigma" else "green" if method == "fraction" else "cyan" if method == "iqr_adjusted" else "blue"
    )

    # log10 plots with linear x-axis
    f.plot.hist(ax=axs[1], bins=n_bins, yscale="log", xlim=[left - extra, right + extra])
    axs[1].title.set_text("Original variance log10 " + yvar.name)
    g.plot.hist(ax=axs[2], bins=n_bins, yscale="log", color="red", xlim=[left - extra, right + extra])
    axs[2].title.set_text("Filtered variance log10 " + yvar.name + " (" + method + " method)")
    for k in [1, 2]:
        if center_log is not None:
            axs[k].axvline(center_log, color="k", linewidth=1)
        if low_log is not None:
            axs[k].axvline(low_log, color=color, linestyle="--", linewidth=2)
            # Add text annotation for low threshold
            axs[k].text(
                low_log,
                axs[k].get_ylim()[1] * 0.5,
                f" low={low:.2e}",
                verticalalignment="bottom",
                fontsize=10,
                fontweight="bold",
            )
        if high_log is not None:
            axs[k].axvline(high_log, color=color, linestyle="--", linewidth=2)
            # # Add text annotation for high threshold
            # axs[k].text(high_log, axs[k].get_ylim()[1] * 0.5,
            #            f' high={high:.2e}',
            #            verticalalignment='bottom',
            #            fontsize=10, fontweight='bold')

    fig.savefig(figname)


# get domain of the variable from the fraction of NaN: UNDER TESTING
def mask_from_field(xfield):
    """get the domain to be passed to the climatology .yml file from the number of
    missing point. Special treatment for sea ice. Use with caution."""
    ratio = float(xfield.count() / np.prod(np.array(xfield.shape)))
    loggy.info(ratio)
    if ratio < 0.2:  # this is a special case for ice, need to be double checked
        mask = "ocean"
    elif 0.2 < ratio < 0.3:
        mask = "land"
        # Check if Antarctica is missing (lat < -60)
        antarctica_missing = False
        antarctic_data = xfield.where(xfield["lat"] < -60, drop=True)
        if antarctic_data.size > 0:
            antarctic_ratio = float(antarctic_data.count() / antarctic_data.size)
            antarctica_missing = antarctic_ratio < 0.1  # Most of Antarctica is missing
        if antarctica_missing:
            mask = "land-no-antarctica"

    elif 0.55 < ratio < 0.7:
        mask = "ocean"
    elif ratio > 0.95:
        mask = "global"
    else:
        mask = "undefined"
        raise ValueError("ERROR: cant recognize mask")

    loggy.debug(mask)
    return mask


def parse_create_args():
    """Parse command line arguments for reference/climatology creation."""
    parser = argparse.ArgumentParser(description="Create ecmean reference/climatology for global mean/performance indices.")
    parser.add_argument(
        "-c",
        "--climdata",
        type=str,
        default="EC26",
        choices=["EC26", "EC24"],
        help="Climatology/reference dataset name (default: EC26)",
    )
    parser.add_argument(
        "-t",
        "--timeframe",
        type=str,
        default="HIST",
        choices=["HIST", "PDAY", "CMIP"],
        help="Time period for the climatology (default: HIST)",
    )
    parser.add_argument("--machine", type=str, default="wilma", help="Machine name for input data path (default: wilma)")
    parser.add_argument(
        "-j",
        "--cores",
        type=int,
        default=1,
        help="Number of cores for parallel processing (default: 1). "
        "Two or more cores will activate Dask for parallel computation.",
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        type=str.upper,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: WARNING)",
    )
    return parser
