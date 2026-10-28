"""Unit tests for the time averaging helpers of ecmean.libs.general."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecmean.libs.general import _time_blocks, season_means, select_season

SEASONS = ["ALL", "DJF", "MAM", "JJA", "SON"]


def _field(chunks=None):
    time = pd.date_range("1990-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(0)
    data = rng.normal(280, 10, size=(36, 3, 4)).astype(np.float32)
    data[5, 0, 0] = np.nan  # missing in a single month: skipped, as in mean()
    data[:, 2, 3] = np.nan  # always missing: NaN in the output
    field = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time}, name="tas")
    return field.chunk({"time": chunks}) if chunks else field


def _assert_matches_mean(field, means):
    """The season means equal the xarray mean of the season time steps."""
    for season in means.season.values:
        expected = select_season(field, season).astype(np.float64).mean("time").compute()
        np.testing.assert_allclose(means.sel(season=season).values, expected.values, rtol=1e-6)


@pytest.mark.parametrize("chunks", [None, 5, 12, (5, 12, 7, 12), (1,) * 36, (30, 6)])
def test_season_means_matches_xarray_mean(chunks):
    """Regular, irregular or no chunks give the same means as xarray mean()."""
    field = _field().chunk({"time": chunks}) if chunks else _field()
    means = season_means(field, SEASONS)
    assert means.dims == ("season", "lat", "lon")
    assert list(means.season.values) == SEASONS
    _assert_matches_mean(field, means)
    assert np.isnan(means.values[:, 2, 3]).all()


def test_season_means_missing_months():
    """Missing months, including a whole season in one year: plain mean of the available steps."""
    field = _field(chunks=12)
    missing = pd.DatetimeIndex(["1990-03-01", "1991-06-01", "1991-07-01", "1991-08-01", "1992-12-01"])
    field = field.sel(time=~field.time.isin(missing))
    _assert_matches_mean(field, season_means(field, SEASONS))

    # a single year without JJA, as in the global mean trend loop
    year = field.sel(time=field.time.dt.year == 1991)
    means = season_means(year, SEASONS)
    assert np.isnan(means.sel(season="JJA").values).all()
    _assert_matches_mean(year, means.sel(season=["ALL", "DJF", "MAM", "SON"]))


def test_season_means_subset_and_order():
    """Only the requested seasons are returned, in the requested order."""
    means = season_means(_field(chunks=12), ["JJA", "ALL"])
    assert list(means.season.values) == ["JJA", "ALL"]


def test_season_means_coordinates_chunked_differently():
    """2D dask coordinates chunked differently from the data (large curvilinear grids) do not break the blocks."""
    field = _field(chunks=12)
    lon = xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon")).chunk({"lat": 1, "lon": 2})
    field = field.assign_coords(lon2d=lon)
    with pytest.raises(ValueError, match="inconsistent chunks"):
        _ = field.chunksizes

    means = season_means(field, ["ALL", "DJF", "JJA"])
    _assert_matches_mean(field, means)


@pytest.mark.parametrize(
    "chunks, blocks",
    [
        ((12, 12, 12), [slice(0, 12), slice(12, 24), slice(24, 36)]),
        ((1,) * 30, [slice(0, 12), slice(12, 24), slice(24, 30)]),
        ((5, 12, 7), [slice(0, 17), slice(17, 24)]),
        ((40,), [slice(0, 40)]),
        ((), []),
    ],
)
def test_time_blocks(chunks, blocks):
    """Blocks are made of whole chunks, with at least 12 time steps (except the last)."""
    assert list(_time_blocks(chunks)) == blocks
