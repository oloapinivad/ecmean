"""
Shared functions for Xarray
"""

import numpy as np

#########################
# FILE FORMAT FUNCTIONS #
#########################


def xr_preproc(ds):
    """Preprocessing function to adjust coordinate and dimension
    names to a common format. To be called by xr.open_mf_dataset()

    Parameters:
    ds (xarray.Dataset): The dataset to be preprocessed.

    Returns:
    xarray.Dataset: The preprocessed dataset with adjusted coordinate and dimension names.
    """

    # exact renames
    exact_dict = {
        "time_counter": "time",
        "longitude": "lon",
        "latitude": "lat",
        "pressure_levels": "plev",
        "values": "cell",
        "x_grid_T": "x",
        "y_grid_T": "y",
    }

    # prefix-based renames: any dim/coord whose name starts with
    # the key gets renamed to the corresponding value
    prefix_dict = {
        "plev": "plev",
        "nav_lon": "lon",
        "nav_lat": "lat",
    }

    # safe check for NEMO output in domain_cfg.nc
    if "nav_lon" in ds.data_vars and "nav_lat" in ds.data_vars:
        ds = ds.set_coords(["nav_lon", "nav_lat"])

    all_dims_and_coords = set(ds.dims) | set(ds.coords)

    # exact matches first
    to_rename = {k: v for k, v in exact_dict.items() if k in all_dims_and_coords}

    # prefix matches (skip names already handled or already correct)
    for name in all_dims_and_coords:
        if name not in to_rename:
            for prefix, target in prefix_dict.items():
                if name.startswith(prefix) and name != target:
                    to_rename[name] = target
                    break

    # a vertical axis named 'lev' is renamed only when it is a pressure axis (e.g. MPI-ESM1-2-XR 3D fields on
    # pressure levels): hybrid sigma levels and ocean depths have other standard names and are left alone
    if "lev" in ds.variables and ds["lev"].attrs.get("standard_name") == "air_pressure":
        to_rename["lev"] = "plev"

    # skip renames whose target already exists, keeping the existing one: e.g. CMOR files
    # with both a 1D 'lat' dimension and a 2D 'latitude' auxiliary coordinate (BCC-CSM2-HR ocean)
    taken = set(ds.dims) | set(ds.variables)
    for name, target in list(to_rename.items()):
        if target in taken:
            del to_rename[name]
        else:
            taken.add(target)

    if to_rename:
        ds = ds.rename(to_rename)

    return ds


def is_bounds_name(name):
    """Whether a variable name is a usual cell-bounds name (``*_bnds``, ``*_bounds``, ``bounds_*``)."""

    return name.endswith(("_bnds", "_bounds")) or name.startswith("bounds_")


def bounds_names(ds):
    """Names of the cell-bounds variables of a dataset.

    Args:
        ds (xarray.Dataset): the dataset to inspect.

    Returns:
        set: the variables referenced by a ``bounds`` attribute (or encoding, where xarray
            moves it when decoding times), plus those named ``*_bnds``, ``*_bounds`` or ``bounds_*``.
    """

    names = set()
    for var in ds.variables.values():
        for meta in (var.attrs, var.encoding):
            if isinstance(meta.get("bounds"), str):
                names.add(meta["bounds"])
    names.update(n for n in ds.variables if is_bounds_name(n))
    return names & set(ds.variables)


def drop_bounds(ds):
    """Preprocessing fixer that drops the cell-bounds variables (see bounds_names).

    Meant for input data only: the area files need their ``lat_bnds``/``lon_bnds``.

    Args:
        ds (xarray.Dataset): the dataset to fix.

    Returns:
        xarray.Dataset: the dataset without its bounds variables.
    """

    return ds.drop_vars(bounds_names(ds))


def rounded_duplicates(ds, rtol=1e-6, atol=1e-6):
    """Find the dimension coordinates (all but time) with neighbouring values equal up to
    rounding: the symptom of an outer join of files whose coordinates differ by rounding,
    which duplicates the grid points (to be fixed with align_coords).

    Args:
        ds (xarray.Dataset): the dataset to check.
        rtol (float): relative tolerance.
        atol (float): absolute tolerance.

    Returns:
        dict: dimension name -> number of values close to their neighbour, for the affected dimensions.
    """

    found = {}
    for dim in ds.dims:
        # rounding only affects floating-point coordinates (integer ones are indices)
        if dim == "time" or dim not in ds.coords or not np.issubdtype(ds[dim].dtype, np.floating):
            continue
        values = np.sort(ds[dim].values)
        close = int(np.isclose(values[1:], values[:-1], rtol=rtol, atol=atol).sum())
        if close:
            found[dim] = close
    return found


def align_coords(reference):
    """Build a preprocessing fixer that replaces the dimension coordinates of a dataset with
    the reference ones when they have the same size and differ only by rounding
    (e.g. float32-rounded latitudes), so that an outer join does not duplicate grid points.

    Args:
        reference (dict): dimension name -> xarray.DataArray with the reference coordinate.

    Returns:
        function: the fixer, taking and returning an xarray.Dataset.
    """

    def fixer(ds):
        for dim, coord in reference.items():
            if (
                dim in ds.coords
                and ds.sizes[dim] == coord.size
                and np.issubdtype(ds[dim].dtype, np.number)
                and np.allclose(ds[dim].values, coord.values, rtol=1e-6, atol=1e-6)
            ):
                ds = ds.assign_coords({dim: coord})
        return ds

    return fixer


def adjust_clim_file(cfield, remove_zero=False):
    """Routine to fix file format of climatology"""

    # fix coordinates
    # org = ['LONGITUDE', 'LATITUDE', 'lev']
    # new = ['lon', 'lat', 'plev']
    # for o, n in zip(org, new):
    #    if o in cfield.coords:
    #        cfield = cfield.rename({o: n})

    # extract data_array
    cname = list(cfield.data_vars)[-1]
    field = cfield[cname]

    if remove_zero:
        field = field.where(field != 0)

    # convert vertical levels
    if "plev" in cfield.coords:
        field = field.metpy.convert_coordinate_units("plev", "Pa")

    return field
