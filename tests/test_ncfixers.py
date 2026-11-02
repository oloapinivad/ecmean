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


def _lev_dataset(standard_name):
    lev = xr.DataArray(np.array([100000.0, 85000.0]), dims="lev", attrs={"standard_name": standard_name, "units": "Pa"})
    return xr.Dataset({"ta": (("lev", "lat"), np.zeros((2, 3)))}, coords={"lev": lev, "lat": np.arange(3.0)})


def test_xr_preproc_pressure_lev_to_plev():
    """A 'lev' axis declared as air pressure is renamed to plev (MPI-ESM1-2-XR)."""
    assert xr_preproc(_lev_dataset("air_pressure"))["ta"].dims == ("plev", "lat")


def test_xr_preproc_other_lev_untouched():
    """Hybrid sigma levels and ocean depths named 'lev' are not renamed."""
    for standard_name in ["atmosphere_hybrid_sigma_pressure_coordinate", "depth", ""]:
        assert xr_preproc(_lev_dataset(standard_name))["ta"].dims == ("lev", "lat")
