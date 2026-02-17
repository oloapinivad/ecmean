#!/usr/bin/env python3
'''
Shared functions for XArray ECmean4
'''

import os
import re
import logging
from pathlib import Path
from glob import glob
import yaml
import xarray as xr

loggy = logging.getLogger(__name__)

##################
# FILE FUNCTIONS #
##################


def inifiles_priority(inidict):
    """
    For areas dictionary and remap dictionary, provides a priority of which
    files to be used for interpolation and area computation
    Areas files comes first, then gridfile and finally land-sea mask.
    Provides flexibility for multiple models with different data access
    """
    priority_keys = ['areafile', 'gridfile', 'maskfile']
    for key in priority_keys:
        filepath = inidict.get(key)
        if filepath:
            return filepath

    loggy.error("No valid inifile found in the provided dictionary: %s", inidict)
    return None


def var_is_there(flist, var, face):
    """
    Check if a variable is available in the input file and provide its units.
    Args:
        flist (list or xarray.DataArray or xarray.Dataset): list of files to
            check for the variable, or an xarray object already loaded.
        var (str): the variable name to check for.
        face (dict): the interface file containing variable definitions.
    
    Returns:
        isavail (bool): if the variable is found or not
        varunit (string):  if the variable is there, its unit (None otherwise)
    """

    # Step 1: Ensure input is valid
    if isinstance(flist, (xr.DataArray, xr.Dataset)):
        xfield = flist
    else:
        if not flist or not all(Path(f).is_file() for f in flist):
            loggy.error("No valid files found for variable %s. Ignoring it.", var)
            return False, None
        xfield = xr.open_mfdataset(
            flist, combine='by_coords', data_vars='all', 
            join='outer', compat='no_conflicts')

    # if variable is derived, extract required vars
    var_req = _get_variables_to_load(var, face)

    missing_vars = [v for v in var_req if v not in xfield.data_vars]
    if missing_vars:
        loggy.warning("Variable %s requires missing variables: %s", var, ', '.join(missing_vars))
        return False, None

    units_avail = {}
    for vname in xfield.data_vars:
        # Try to get the 'units' attribute, or assume it's a fraction
        unit = getattr(xfield[vname], 'units', 'frac')
        if unit == '(0 - 1)':
            unit = 'frac'
        units_avail[vname] = unit

    # get units
    varunit = face['variables'][var].get('units', None)
    if not varunit:
        # If not defined, fall back to the unit of the first required variable
        varunit = units_avail.get(var_req[0])

        if len(var_req) > 1:
            loggy.debug(
                "Variable '%s' is derived. Using unit from its first component '%s': %s",
                var, var_req[0], varunit
            )

    return True, varunit


def get_clim_files(piclim, var, diag, season):
    """Function to extra names for the climatology files"""

    # extract info from pi_climatology.yml
    # reference dataset and reference varname
    # as well as years when available
    dataref = piclim[var]['dataset']
    datayear1 = piclim[var].get('year1', None)
    datayear2 = piclim[var].get('year2', None)

    if diag.climatology not in ['EC23', 'EC24', 'HIST26', 'PDAY26']:
        raise ValueError(f'Climatology {diag.climatology} not supported/existing!')
    
    stringname='climate' if season == 'ALL' else 'seasons'

    clim = str(
        diag.resclmdir /
        f'{stringname}_average_{var}_{dataref}_{diag.resolution}_{datayear1}-{datayear2}.nc')
    vvvv = str(
        diag.resclmdir /
        f'{stringname}_variance_{var}_{dataref}_{diag.resolution}_{datayear1}-{datayear2}.nc')

    return clim, vvvv


def get_inifiles(face, diag):
    """
    Resolves initialization files for each model component.
    
    For each component defined in face['model']['component'], this function attempts to:
    - Look up related input file names in face['component']
    - Build the absolute path (if relative)
    - Expand wildcard patterns
    - Filter the result by the target year

    Returns:
        dict: A nested dictionary of the form ifiles[component][name] = resolved_file_path or ''
    """

    component_map = face['model']['component']
    ifiles = {}

    for comp_name, component_key in component_map.items():
        ifiles[comp_name] = {}

        file_definitions = face['component'][component_key]

        for file_label, file_path in file_definitions.items():
            resolved_path = ''

            if file_path:
                # Make path absolute if it's relative
                path_obj = Path(file_path)
                if not path_obj.is_absolute():
                    path_obj = (
                        Path(diag.ecedir) /
                        Path(face['model']['basedir']) /
                        path_obj
                    )

                # Expand wildcards and resolve files
                expanded = _expand_filename(path_obj, '', diag)
                matching_files = glob(str(expanded))

                if matching_files:
                    # Pick file matching the target year
                    filtered_files = _filter_filename_by_year(str(path_obj), matching_files, diag.year1)
                    if filtered_files:
                        resolved_path = str(filtered_files[0])
                else:
                    loggy.warning('Inifile %s cannot be found!', expanded)

            ifiles[comp_name][file_label] = resolved_path
            loggy.info('%s for component %s is: %s', file_label, comp_name, resolved_path or 'MISSING')

    return ifiles

def _expand_filename(filenames, var, diag):
    """Expands a path (filename or dir) for var, expname, frequency, ensemble etc.
    and environment variables. Years are set as a wildcard and filtered by _filter_by_year"""

    return Path(str(os.path.expandvars(filenames)).format(
        expname=diag.expname,
        year1='*',
        year2='*',
        var=var,
        frequency=diag.frequency,
        ensemble=diag.ensemble,
        grid=diag.grid,
        model=diag.modelname,
        version=diag.version
    ))


def _filter_filename_by_year(template, filenames, year):
    """Find filename containing a given year in a list of filenames"""

    # if year1 is used in the file template
    if 'year1' in template:
        # Assumes that the file name ends with 199001-199012.nc or 1990-1991.nc
        year1 = [int(x.split('_')[-1].split('-')[0][0:4]) for x in filenames]
        # if year2 is used in the file template
        if 'year2' in template:
            year2 = [int(x.split('_')[-1].split('-')[1][0:4]) for x in filenames]
        else:
            year2 = year1
        # filter names
        filternames = [fname for y1, y2, fname in zip(year1, year2, filenames)
                       if year >= y1 and year <= y2]
    else:
        # this is introduced for file that does not have year in their filename
        filternames = filenames

    # safety warning if something is missing
    if not filternames and len(filenames) > 0:
        loggy.warning('Data for year %s has not been found!', str(year))

    loggy.debug('Filtered filenames: %s', filternames)
    return filternames


def load_yaml(infile):
    """
    Load and parse a YAML configuration file.

    Args:
        infile (str | Path): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
        ValueError: If the YAML content is empty or invalid.
    """
    path = Path(infile)

    if not path.is_file():
        raise FileNotFoundError(f"[YAML Loader] File not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"[YAML Loader] Failed to parse YAML file {path}: {e}")

    if data is None:
        raise ValueError(f"[YAML Loader] Parsed YAML file is empty: {path}")

    return data

def _create_filepath(cmorname, face, diag):
    """Create filepath with wildcards"""

    filetype = face['variables'][cmorname]['filetype']
    filepath = Path(diag.ecedir) / \
        Path(face['model']['basedir']) / \
        Path(face['filetype'][filetype]['dir']) / \
        Path(face['filetype'][filetype]['filename'])
    loggy.debug('Filepath: %s', filepath)

    return filepath


def make_input_filename(cmorname, face, diag):
    """Create full input filepaths for the required variable and a given year

    Returns:
        a list of files to be loaded by xarray
    """

    # if a dataarray is provided
    if diag.xdataset is not None:
        return diag.xdataset

    # detect if it is a derived vars and get the list of var to be loaded
    varname = _get_variables_to_load(cmorname, face)

    # create filepath
    filepath = _create_filepath(cmorname, face, diag)

    # create the file structure according to the interface file
    filename = []
    for year in diag.years_joined:
        filename1 = []
        for var in varname:
            expandfile = _expand_filename(filepath, var, diag)
            filenames = glob(str(expandfile))
            fname = _filter_filename_by_year(str(filepath), filenames, year)
            filename1 = filename1 + fname
        filename = filename + filename1
    filename = list(dict.fromkeys(filename))  # Filter unique ones
    loggy.debug("Filenames: %s", filename)
    return filename


def _get_variables_to_load(var, face):
    """Function to extract from the interface file the list of derived variable,
    i.e. the real variables to be loaded, for each of the cmorname introduced in the
    interface file

    Args:
        var: the cmorname variable of the data to be loaded
        face: the interface file
    """

    if 'derived' in face['variables'][var].keys():
        cmd = face['variables'][var]['derived']
        return [x for x in re.split('[*+-/]', cmd) if not x.isdigit()]

    return [var]

def load_output_yaml(yamlfile):

    """Load the variable statistics from the yaml file."""
    loggy.info('Loading the stored data from the yaml file %s', yamlfile)
    if os.path.isfile(yamlfile):
        with open(yamlfile, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    raise FileNotFoundError(f'File {yamlfile} not found.')
    
