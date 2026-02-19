"""Common functions for reference and climatology creation."""

import argparse
import logging
import os

import numpy as np


# Climatology file prefixes
CLIMATOLOGY_PREFIXES = [
    'climate_variance_',
    'climate_average_',
    'seasons_variance_',
    'seasons_average_'
]



def timeframe_years(timeframe):
    """Timeframe to years mapping."""
    if timeframe == 'HIST':
        year1 = 1981
        year2 = 2010
    elif timeframe == 'PDAY':
        year1 = 2000
        year2 = 2024
    elif timeframe == 'CMIP':
        year1 = 1985
        year2 = 2014
    else:
        raise ValueError(f"TIMEFRAME {timeframe} not recognized."
                         "Please choose from HIST, PDAY, CMIP.")
    return year1, year2

def expand_filedata(directory, var, info):
    """Expand filedata directory with environment variables and info dictionary."""

    return os.path.expandvars(directory).format(
        datadir=info['dirs']['datadir'], mswepdir=info['dirs']['mswepdir'],
        eradir=info['dirs']['eradir'], esadir=info['dirs']['esadir'],
        dataset=info[var]['dataset']
    )

def select_time_data(yearly, seasonly, season):
    """Select time data based on season."""

    if season == 'ALL':
        return yearly

    gfield = seasonly.sel(time=seasonly.time.dt.season.isin(season))
    # for winter, we drop first and last to have only complete season.
    # this reduces the sample by one but it is safer for variance
    if season == 'DJF':
        gfield = gfield.drop_isel(time=[0, gfield.sizes['time'] - 1])

    return gfield

def select_time_period(xfield, var, year1, year2):
    """
    Select time period based on data availability.
    """
    # Check available years
    valid_time = xfield[var].dropna("time", how="all").time
    avail_years = np.unique(valid_time.dt.year.values)
    
    # Calculate intersection between requested and available
    real_year1 = max(year1, int(avail_years.min()))
    real_year2 = min(year2, int(avail_years.max()))

    # Raise error if incompatible
    if real_year1 > real_year2:
        raise ValueError(
            f"{var}: requested period {year1}-{year2} "
            f"not compatible with data availability "
            f"{avail_years.min()}-{avail_years.max()}"
        )

    # Warn if years differ from requested
    if real_year1 != year1 or real_year2 != year2:
        logging.warning(
            "%s: using %s-%s instead of requested %s-%s",
            var, real_year1, real_year2, year1, year2
        )

    # Select time
    years_eff = list(range(real_year1, real_year2 + 1))
    cfield = xfield[var].sel(time=xfield[var].time.dt.year.isin(years_eff))

    return cfield, real_year1, real_year2


def get_climatology_files(tgtdir, var, dataset, grid, year1, year2):
    """Generate climatology file paths.
    """
    
    suffix = f'{var}_{dataset}_{grid}_{year1}_{year2}.nc'
    
    climatology_files = {
        prefix: os.path.join(tgtdir, grid, prefix + suffix)
        for prefix in CLIMATOLOGY_PREFIXES
    }
    
    return climatology_files


def parse_create_args():
    """Parse command line arguments for reference/climatology creation."""
    parser = argparse.ArgumentParser(
        description='Create ecmean reference/climatology for global mean/performance indices.'
    )
    parser.add_argument(
        '-c', '--climdata',
        type=str,
        default='EC26',
        choices=['EC26', 'EC24'],
        help='Climatology/reference dataset name (default: EC26)'
    )
    parser.add_argument(
        '-t', '--timeframe',
        type=str,
        default='HIST',
        choices=['HIST', 'PDAY', 'CMIP'],
        help='Time period for the climatology (default: HIST)'
    )
    parser.add_argument(
        '--machine',
        type=str,
        default='wilma',
        help='Machine name for input data path (default: wilma)'
    )
    parser.add_argument(
        '-j', '--cores',
        type=int,
        default=1,
        help='Number of cores for parallel processing (default: 1). ' \
        'Two or more cores will activate Dask for parallel computation.'
    )
    parser.add_argument(
        '-l', '--loglevel',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: WARNING)'
    )
    return parser
