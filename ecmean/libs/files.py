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

    if diag.climatology not in ['EC23', 'EC24', 'HM25']:
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
        version=diag.version,
        consortium=diag.consortium,
        mip=diag.mip
    ))


def _filter_filename_by_year(template, filenames, year):
    """
    Filter filenames that contain data for a specific year.
    
    Args:
        template (str): File template pattern used to determine if year filtering applies
        filenames (list): List of actual filenames to filter
        year (int): Target year to filter for
        
    Returns:
        list: Filtered list of filenames containing the target year
        
    The function handles two filename patterns:
    - Files with year ranges: 'file_199001-199012.nc' or 'file_1990-1991.nc'
    - Files without year information: returns all files unchanged
    """

    # Early return if no filenames provided
    if not filenames:
        return []

    # If template doesn't use year placeholders, return all files
    if 'year1' not in template:
        loggy.debug('Template has no year placeholders, returning all %d files', len(filenames))
        return filenames

    # Extract year ranges from filenames and filter
    filtered_files = []

    for filename in filenames:
        try:
            # Extract the date part (assumes format: *_YYYYMM-YYYYMM.nc or *_YYYY-YYYY.nc)
            date_part = filename.split('_')[-1].split('.')[0]  # Remove extension

            if '-' not in date_part:
                loggy.warning('Filename %s does not contain expected date range format', filename)
                continue
   
            start_date, end_date = date_part.split('-', 1)

            # Extract 4-digit years from start and end dates
            start_year = int(start_date[:4])
            end_year = int(end_date[:4]) if 'year2' in template else start_year

            # Check if target year falls within the file's time range
            if start_year <= year <= end_year:
                filtered_files.append(filename)

        except (ValueError, IndexError) as e:
            loggy.warning('Failed to parse year from filename %s: %s', filename, e)
            continue

    # Log results
    if not filtered_files and filenames:
        loggy.warning('No files found containing data for year %d', year)
    else:
        loggy.debug('Filtered %d files for year %d: %s',
                   len(filtered_files), year, filtered_files)
    
    return filtered_files


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
    """
    Create full input filepaths for the required variable and given years.
    
    Args:
        cmorname (str): CMOR variable name to process
        face (dict): Interface configuration containing variable and file definitions
        diag (object): Diagnostic object containing years, paths, and other metadata
        
    Returns:
        list or xarray.Dataset: Either a list of file paths to be loaded by xarray,
                               or the provided xarray dataset if already available
    """
    
    # Early return if dataset is already provided
    if diag.xdataset is not None:
        loggy.debug('Using provided xarray dataset for variable %s', cmorname)
        return diag.xdataset

    # Get variables to load (handles derived variables)
    variables_to_load = _get_variables_to_load(cmorname, face)
    loggy.debug('Variables to load for %s: %s', cmorname, variables_to_load)

    # Create base filepath template
    filepath_template = _create_filepath(cmorname, face, diag)

    # Collect all matching files across years and variables
    all_files = set()  # Use set to automatically handle duplicates
    
    for year in diag.years_joined:
        for variable in variables_to_load:
            # Expand filepath with variable-specific information
            expanded_filepath = _expand_filename(filepath_template, variable, diag)
            
            # Find matching files using glob
            matching_files = glob(str(expanded_filepath))
            
            if not matching_files:
                loggy.warning('No files found for variable %s, year %d with pattern: %s',
                             variable, year, expanded_filepath)
                continue
            
            # Filter files by year and add to collection
            year_filtered_files = _filter_filename_by_year(
                str(filepath_template), matching_files, year
            )
            
            if year_filtered_files:
                all_files.update(year_filtered_files)
            else:
                loggy.warning('No files found for variable %s, year %d after year filtering',
                             variable, year)
    
    # Convert to sorted list for consistent ordering
    final_filelist = sorted(list(all_files))
    
    if not final_filelist:
        loggy.error('No input files found for variable %s across all requested years', cmorname)
    else:
        loggy.info('Collected %d unique files for variable %s', len(final_filelist), cmorname)
        loggy.debug('Final file list: %s', final_filelist)
    
    return final_filelist


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
    
