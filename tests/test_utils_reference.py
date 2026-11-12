"""Unit tests for the time selection helpers of the offline reference/climatology tools (ecmean.utils.utils)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ecmean.utils.utils imports statsmodels, which is only in the tests extra
utils = pytest.importorskip("ecmean.utils.utils")
select_time_data = utils.select_time_data
select_time_period = utils.select_time_period


def _dataset(start, end, nan_months=None):
    """Synthetic monthly dataset (like CERES-EBAF, which starts in March 2000)."""
    time = pd.date_range(start, end, freq="MS")
    data = np.ones((time.size, 2, 3))
    for month in nan_months or []:
        data[time == pd.Timestamp(month)] = np.nan
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time}).to_dataset(name="rsnt")


def test_select_time_period_drops_incomplete_years():
    """A record starting in March and ending in June uses complete years only."""
    cfield, year1, year2 = select_time_period(_dataset("2000-03-01", "2015-06-01"), "rsnt", 1985, 2014)
    assert (year1, year2) == (2001, 2014)
    assert cfield.time.values[0] == np.datetime64("2001-01-01")
    assert cfield.time.values[-1] == np.datetime64("2014-12-01")

    # every yearly mean is made of 12 months
    yearly = cfield.resample(time="YS").count("time")
    assert yearly.sizes["time"] == 14
    assert (yearly == 12).all()


def test_select_time_period_all_nan_months():
    """Months filled with NaN do not count as valid."""
    ds = _dataset("2000-01-01", "2014-12-01", nan_months=["2000-01-01", "2014-11-01"])
    _, year1, year2 = select_time_period(ds, "rsnt", 1985, 2014)
    assert (year1, year2) == (2001, 2013)


def test_select_time_period_requested_years_kept():
    """Complete data over the requested period are not modified."""
    cfield, year1, year2 = select_time_period(_dataset("1980-01-01", "2020-12-01"), "rsnt", 1985, 2014)
    assert (year1, year2) == (1985, 2014)
    assert cfield.sizes["time"] == 30 * 12


def test_select_time_period_no_complete_year():
    """Less than one complete year raises an error."""
    with pytest.raises(ValueError):
        select_time_period(_dataset("2000-03-01", "2001-02-01"), "rsnt", 1985, 2014)


@pytest.mark.parametrize("season, expected", [("DJF", 13), ("MAM", 14), ("JJA", 14), ("SON", 14)])
def test_select_time_data_complete_seasons(season, expected):
    """With data starting in March, only complete seasons are kept, and no complete DJF is lost."""
    cfield, _, _ = select_time_period(_dataset("2000-03-01", "2014-12-01"), "rsnt", 1985, 2014)
    yearly = cfield.resample(time="YS").count("time")
    seasonly = cfield.resample(time="QE-NOV").count("time")
    gfield = select_time_data(yearly, seasonly, season)
    assert gfield.sizes["time"] == expected
    assert (gfield == 3).all()
