"""Unit tests for ecmean.libs.ncfixers."""

import numpy as np
import xarray as xr

from ecmean.libs.ncfixers import xr_preproc


def test_xr_preproc_renames_coordinates():
    """Non-standard coordinate names are renamed to lon/lat/time."""
    ds = xr.Dataset(
        {"tos": (("time_counter", "latitude", "longitude"), np.zeros((1, 3, 4)))},
        coords={"time_counter": [0], "latitude": np.arange(3.0), "longitude": np.arange(4.0)},
    )
    out = xr_preproc(ds)
    assert out["tos"].dims == ("time", "lat", "lon")


def test_xr_preproc_keeps_existing_target():
    """A 2D 'latitude' auxiliary coordinate is not renamed onto an existing 1D 'lat' (BCC-CSM2-HR ocean)."""
    lat, lon = np.arange(3.0), np.arange(4.0)
    ds = xr.Dataset(
        {"tos": (("lat", "lon"), np.zeros((3, 4)))},
        coords={
            "lat": lat,
            "lon": lon,
            "latitude": (("lat", "lon"), np.zeros((3, 4))),
            "longitude": (("lat", "lon"), np.zeros((3, 4))),
        },
    )
    out = xr_preproc(ds)
    assert out["lat"].dims == ("lat",)
    assert out["lon"].dims == ("lon",)
