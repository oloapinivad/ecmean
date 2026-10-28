"""
Shared functions for XArray ecmean
"""

import logging
import os
from glob import glob
from pathlib import Path

import xarray as xr
import yaml

from ecmean.libs.climatology import SUPPORTED_CLIMATOLOGY
from ecmean.libs.formula import formula_variables, formula_wrapper
from ecmean.libs.general import check_time_axis, check_var_interface
from ecmean.libs.ncfixers import xr_preproc

loggy = logging.getLogger(__name__)

# maximum number of input files kept open by xarray (see open_input_files)
FILE_CACHE_MAXSIZE = 16

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
    priority_keys = ["areafile", "gridfile", "maskfile"]
    for key in priority_keys:
        filepath = inidict.get(key)
        if filepath:
            return filepath

    loggy.error("No valid inifile found in the provided dictionary: %s", inidict)
    return None


def open_input_files(flist, **kwargs):
    """Open a list of files as a single dataset, with the same combine/join
    options for all the diagnostics. Extra keyword arguments (e.g. preprocess,
    chunks) are passed to xarray.open_mfdataset."""

    # xarray keeps up to 128 files open, each holding the 64 MiB chunk cache of netCDF-C: streaming monthly
    # files, a worker would accumulate gigabytes of useless cache (12 GB instead of 5 GB on 10 years of 3D
    # CMCC-CM2-VHR4 fields), so the open files are limited to about one block of time chunks (season_means)
    xr.set_options(file_cache_maxsize=FILE_CACHE_MAXSIZE)
    return xr.open_mfdataset(flist, combine="by_coords", data_vars="all", join="outer", compat="no_conflicts", **kwargs)


def files_exist(flist, var):
    """Check that a list of input files is non-empty and all files exist"""

    if not flist or not all(Path(f).is_file() for f in flist):
        loggy.error("No valid files found for variable %s. Ignoring it.", var)
        return False
    return True


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
        if not files_exist(flist, var):
            return False, None
        xfield = open_input_files(flist)

    # if variable is derived, extract required vars
    var_req = _get_variables_to_load(var, face)

    missing_vars = [v for v in var_req if v not in xfield.data_vars]
    if missing_vars:
        loggy.warning("Variable %s requires missing variables: %s", var, ", ".join(missing_vars))
        return False, None

    units_avail = {}
    for vname in xfield.data_vars:
        # Try to get the 'units' attribute, or assume it's a fraction
        unit = getattr(xfield[vname], "units", "frac")
        if unit == "(0 - 1)":
            unit = "frac"
        units_avail[vname] = unit

    # get units
    varunit = face["variables"][var].get("units", None)
    if not varunit:
        # If not defined, fall back to the unit of the first required variable
        varunit = units_avail.get(var_req[0])

        if len(var_req) > 1:
            loggy.debug("Variable '%s' is derived. Using unit from its first component '%s': %s", var, var_req[0], varunit)

    return True, varunit


def load_variable(var, face, diag):
    """
    Load the field of a variable for the years of the diagnostic, shared by global mean and PI workers.

    Args:
        var (str): the variable name.
        face (dict): the interface file containing variable definitions.
        diag (Diagnostic): the diagnostic instance.

    Returns:
        tuple: (field, units), with field as a lazy xarray.DataArray and units the original model units,
            or None if the variable is not defined in the interface or not available.
    """

    if not check_var_interface(var, face):
        return None

    infile = make_input_filename(var, face, diag)

    # open the files only once (chunking on time only, might be improved) and check the variable on the dataset
    if isinstance(infile, (xr.DataArray, xr.Dataset)):
        xfield = infile
    elif files_exist(infile, var):
        xfield = open_input_files(infile, preprocess=xr_preproc, chunks={"time": 12})
    else:
        return None

    isavail, varunit = var_is_there(xfield, var, face)
    if not isavail:
        return None

    # in case of big files with multi year, be sure of having opened the right records
    xfield = xfield.sel(time=xfield.time.dt.year.isin(diag.years_joined))
    check_time_axis(xfield.time, diag.years_joined)

    return formula_wrapper(var, face, xfield), varunit


def get_clim_files(piclim, var, diag, season):
    """Function to extra names for the climatology files"""

    # extract info from pi_climatology.yml
    # reference dataset and reference varname
    # as well as years when available
    dataref = piclim[var]["dataset"]
    datayear1 = piclim[var].get("year1", None)
    datayear2 = piclim[var].get("year2", None)

    if diag.climatology not in SUPPORTED_CLIMATOLOGY:
        raise ValueError(f"Climatology {diag.climatology} not supported/existing!")

    stringname = "climate" if season == "ALL" else "seasons"

    clim = str(diag.resclmdir / f"{stringname}_average_{var}_{dataref}_{diag.resolution}_{datayear1}-{datayear2}.nc")
    vvvv = str(diag.resclmdir / f"{stringname}_variance_{var}_{dataref}_{diag.resolution}_{datayear1}-{datayear2}.nc")

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

    component_map = face["model"]["component"]
    ifiles = {}

    for comp_name, component_key in component_map.items():
        ifiles[comp_name] = {}

        file_definitions = face["component"][component_key]

        for file_label, file_path in file_definitions.items():
            resolved_path = ""

            if file_path:
                # Make path absolute if it's relative
                path_obj = Path(file_path)
                if not path_obj.is_absolute():
                    path_obj = Path(diag.ecedir) / Path(face["model"]["basedir"]) / path_obj

                # Expand wildcards and resolve files
                expanded = _expand_filename(path_obj, "", diag)
                matching_files = glob(str(expanded))

                if matching_files:
                    # Pick file matching the target year
                    filtered_files = _filter_filename_by_year(str(path_obj), matching_files, diag.year1)
                    if filtered_files:
                        resolved_path = str(filtered_files[0])
                else:
                    loggy.warning("Inifile %s cannot be found!", expanded)

            ifiles[comp_name][file_label] = resolved_path
            loggy.info("%s for component %s is: %s", file_label, comp_name, resolved_path or "MISSING")

    return ifiles


def _expand_filename(filenames, var, diag):
    """Expands a path (filename or dir) for var, expname, frequency, ensemble etc.
    and environment variables. Years are set as a wildcard and filtered by _filter_by_year"""

    return Path(
        str(os.path.expandvars(filenames)).format(
            expname=diag.expname,
            year1="*",
            year2="*",
            var=var,
            frequency=diag.frequency,
            ensemble=diag.ensemble,
            grid=diag.grid,
            model=diag.modelname,
            version=diag.version,
            consortium=diag.consortium,
            mip=diag.mip,
        )
    )


def _filter_filename_by_year(template, filenames, year):
    """Find filename containing a given year in a list of filenames"""

    # if year1 is used in the file template
    if "year1" in template:
        filternames = []
        for fname in filenames:
            # Assumes that the file name ends with 199001-199012.nc or 1990-1991.nc:
            # files matched by the glob wildcards that do not follow this pattern are skipped
            try:
                dates = fname.split("_")[-1].split("-")
                year1 = int(dates[0][0:4])
                # if year2 is used in the file template
                year2 = int(dates[1][0:4]) if "year2" in template else year1
            except (ValueError, IndexError) as err:
                loggy.warning("Cannot extract the years from %s, skipping it: %s", fname, err)
                continue
            if year1 <= year <= year2:
                filternames.append(fname)
    else:
        # this is introduced for file that does not have year in their filename
        filternames = filenames

    # safety warning if something is missing
    if not filternames and len(filenames) > 0:
        loggy.warning("Data for year %s has not been found!", str(year))

    loggy.debug("Filtered filenames: %s", filternames)
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

    filetype = face["variables"][cmorname]["filetype"]
    filepath = (
        Path(diag.ecedir)
        / Path(face["model"]["basedir"])
        / Path(face["filetype"][filetype]["dir"])
        / Path(face["filetype"][filetype]["filename"])
    )
    loggy.debug("Filepath: %s", filepath)

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
    # the pattern does not depend on the year (years are wildcards): glob once per variable
    globbed = {var: glob(str(_expand_filename(filepath, var, diag))) for var in varname}
    for var, files in globbed.items():
        if not files:
            loggy.warning("No files found for variable %s with pattern %s", var, _expand_filename(filepath, var, diag))

    filename = []
    for year in diag.years_joined:
        filename1 = []
        for var in varname:
            fname = _filter_filename_by_year(str(filepath), globbed[var], year)
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

    if "derived" in face["variables"][var]:
        cmd = face["variables"][var]["derived"]
        return formula_variables(cmd)

    return [var]


def load_output_yaml(yamlfile):
    """Load the variable statistics from the yaml file."""
    loggy.info("Loading the stored data from the yaml file %s", yamlfile)
    if os.path.isfile(yamlfile):
        with open(yamlfile, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    raise FileNotFoundError(f"File {yamlfile} not found.")
