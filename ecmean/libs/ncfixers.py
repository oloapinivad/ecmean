"""
Shared functions for Xarray
"""

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
