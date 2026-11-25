"""Unit tests for the filename handling of ecmean.libs.files."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecmean.libs.files import _expand_filename, _filter_filename_by_year, open_input_files

TEMPLATE = "{var}_Amon_{model}_{expname}_{ensemble}_gn_{year1}??-{year2}??.nc"


def test_expand_filename_cmor_identifiers():
    diag = SimpleNamespace(
        expname="hist-1950",
        frequency="*mon",
        ensemble="r1i1p2f1",
        grid="*",
        modelname="EC-Earth3P-HR",
        version="*",
        consortium="EC-Earth-Consortium",
        mip="HighResMIP",
    )
    path = _expand_filename("CMIP6/{mip}/{consortium}/{model}/{expname}/" + TEMPLATE, "tas", diag)
    assert str(path) == (
        "CMIP6/HighResMIP/EC-Earth-Consortium/EC-Earth3P-HR/hist-1950/tas_Amon_EC-Earth3P-HR_hist-1950_r1i1p2f1_gn_*??-*??.nc"
    )


def test_filter_filename_by_year():
    files = ["a/tas_gn_199001-199012.nc", "a/tas_gn_199101-199112.nc", "a/tas_gn_1985-1994.nc"]
    assert _filter_filename_by_year(TEMPLATE, files, 1990) == [files[0], files[2]]


def test_filter_filename_by_year_skips_unparsable():
    files = ["a/tas_gn_199001-199012.nc", "a/tas_gn_clim.nc", "a/tas_gn_1990xx.nc"]
    assert _filter_filename_by_year(TEMPLATE, files, 1990) == [files[0]]


def test_filter_filename_without_years():
    files = ["a/sftlf_fx_gn.nc"]
    assert _filter_filename_by_year("sftlf_fx_{grid}.nc", files, 1990) == files


LATS = np.array([-89.05759162, 45.0])


def _monthly_file(path, var, first_bound, lat=LATS):
    """Write a small monthly file with time bounds, the first one set to first_bound"""
    time = pd.date_range("1850-01-15", periods=3, freq="MS") + pd.Timedelta(days=14)
    bounds = np.array([pd.date_range("1850-01-01", periods=3, freq="MS"), pd.date_range("1850-02-01", periods=3, freq="MS")]).T
    bounds[0, 0] = first_bound
    xfield = xr.Dataset(
        {var: (("time", "lat"), np.arange(6.0).reshape(3, 2)), "time_bnds": (("time", "bnds"), bounds)},
        coords={"time": time, "lat": lat},
    )
    xfield["time"].attrs["bounds"] = "time_bnds"
    for name in ("time", "time_bnds"):
        xfield[name].encoding.update(units="days since 1850-01-01", dtype="float64")
    xfield.to_netcdf(path)
    return str(path)


def test_open_input_files_conflicting_bounds(tmp_path):
    """Files whose time_bnds and latitudes differ by rounding (CESM2 prsn) are opened
    without the bounds and on the latitudes of the first file"""
    flist = [
        _monthly_file(tmp_path / "rsds.nc", "rsds", np.datetime64("1850-01-01")),
        _monthly_file(
            tmp_path / "prsn.nc",
            "prsn",
            np.datetime64("1849-12-31T23:29:59.999997"),
            lat=LATS.astype("float32").astype("float64"),
        ),
    ]
    xfield = open_input_files(flist)
    assert set(xfield.data_vars) == {"rsds", "prsn"}
    assert xfield.sizes["time"] == 3
    np.testing.assert_array_equal(xfield["lat"].values, LATS)
    np.testing.assert_array_equal(xfield["prsn"].values, np.arange(6.0).reshape(3, 2))


def test_open_input_files_matching_bounds_kept(tmp_path):
    """Files with identical bounds are merged as before, keeping time_bnds"""
    flist = [_monthly_file(tmp_path / f"{v}.nc", v, np.datetime64("1850-01-01")) for v in ("rsds", "prsn")]
    assert "time_bnds" in open_input_files(flist)


def test_open_input_files_conflicting_data_raises(tmp_path):
    """Conflicts in variables other than bounds still raise"""
    rsds = _monthly_file(tmp_path / "rsds.nc", "rsds", np.datetime64("1850-01-01"))
    prsn = _monthly_file(tmp_path / "prsn.nc", "prsn", np.datetime64("1849-12-31T23:29:59.999997"))
    # a second file with prsn and a different rsds
    xfield = xr.open_dataset(prsn).load()
    xfield["rsds"] = xfield["prsn"] + 1
    xfield.to_netcdf(tmp_path / "both.nc")
    with pytest.raises(xr.MergeError):
        open_input_files([rsds, str(tmp_path / "both.nc")])


def test_open_input_files_rounded_coords_without_bounds(tmp_path):
    """Latitudes differing by rounding are aligned even if no merge error is raised"""
    rsds = _monthly_file(tmp_path / "rsds.nc", "rsds", np.datetime64("1850-01-01"))
    prsn = _monthly_file(
        tmp_path / "prsn.nc", "prsn", np.datetime64("1850-01-01"), lat=LATS.astype("float32").astype("float64")
    )
    for path in (rsds, prsn):
        with xr.open_dataset(path) as xfield:
            xfield = xfield.load()
        xfield.drop_vars("time_bnds").to_netcdf(path)
    xfield = open_input_files([rsds, prsn])
    assert xfield.sizes["lat"] == 2
    np.testing.assert_array_equal(xfield["lat"].values, LATS)
