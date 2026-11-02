"""Unit tests for the helper functions of ecmean.libs.support."""

import numpy as np
import pytest
import xarray as xr

from ecmean.libs.support import Supporter, check_file_exist, identify_grid, mask_coincident_cells


def _regular_field(lat):
    return xr.DataArray(np.zeros((len(lat), 4)), coords={"lat": lat, "lon": np.arange(4.0)}, dims=["lat", "lon"], name="area")


def test_identify_grid_lonlat_float_spacing():
    """A regular grid whose float32 spacings differ by rounding is still lonlat."""
    lat = np.linspace(-89.5, 89.5, 180, dtype=np.float32)
    assert identify_grid(_regular_field(lat)) == "lonlat"


def test_identify_grid_gaussian():
    lat = np.polynomial.legendre.leggauss(8)[0] * 90.0
    assert identify_grid(_regular_field(lat)) == "gaussian"


def test_identify_grid_curvilinear_checks_lat():
    """2D lon with 1D lat must not be flagged as curvilinear."""
    lon2d = np.zeros((3, 4))
    xfield = xr.DataArray(
        np.zeros((3, 4)),
        coords={"lon": (("y", "x"), lon2d), "lat": ("x", np.arange(4.0))},
        dims=["y", "x"],
    )
    assert identify_grid(xfield) != "curvilinear"

    xfield = xfield.assign_coords(lat=(("y", "x"), lon2d))
    assert identify_grid(xfield) == "curvilinear"


@pytest.mark.parametrize("file", [None, ""])
def test_check_file_exist_undefined(file):
    with pytest.raises(KeyError, match="not defined"):
        check_file_exist(file)


def test_check_file_exist_no_match(tmp_path):
    with pytest.raises(KeyError, match="cannot be found"):
        check_file_exist(str(tmp_path / "missing*.nc"))


def test_check_file_exist_found(tmp_path):
    target = tmp_path / "mask.nc"
    target.write_text("")
    assert check_file_exist(str(tmp_path / "ma*.nc")) == str(target)


def test_supporter_omip_without_masks():
    """OMIP: ocean grid only, no atmospheric files and no ocean mask, with remapping."""
    lon, lat = np.arange(0.5, 360, 10.0), np.arange(-85, 90, 10.0)
    targetgrid = xr.Dataset(
        {"tas": (("lat", "lon"), np.zeros((len(lat), len(lon))))},
        coords={"lon": lon, "lat": lat},
    )
    atmdict = {"maskfile": "", "gridfile": "", "areafile": ""}
    ocedict = {"maskfile": "", "gridfile": "", "areafile": "tests/data/omip/domain_cfg.nc"}

    support = Supporter({"atm": "oifs", "oce": "nemo"}, atmdict, ocedict, areas=False, remap=True, targetgrid=targetgrid)

    assert support.atmmask is None
    assert support.atmarea is None
    assert support.ocemask is None
    assert support.oceremap is not None
    assert support.oceremap.method == "bilinear"


def test_mask_coincident_cells():
    """Cells sharing their coordinates with many others (placeholders outside the domain) are masked,
    the few duplicates of tripolar folds are not, regular grids are untouched."""
    lon2d, lat2d = np.meshgrid(np.arange(0.5, 360, 10.0), np.arange(-85.0, 90, 10.0))
    lat2d[-1] = lat2d[-2]  # fold: the last row duplicates the previous one
    lon2d[:5, :10], lat2d[:5, :10] = -1.0, -1.0  # placeholder block outside the domain
    grid = xr.Dataset(
        {"tos": (("y", "x"), np.ones(lon2d.shape))}, coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d)}
    )

    masked = mask_coincident_cells(grid)
    assert int((~masked["mask"]).sum()) == 50
    assert bool(masked["mask"][-1].all())

    regular = xr.Dataset({"tos": (("lat", "lon"), np.ones((18, 36)))}, coords={"lon": lon2d[0], "lat": lat2d[:, 0]})
    assert "mask" not in mask_coincident_cells(regular)


def test_supporter_without_atm_mask(tmp_path):
    """An atmospheric grid without land-sea mask provides areas and no mask, without failing."""
    gridfile = tmp_path / "grid.nc"
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.arange(0.5, 360.0, 1.0)
    xr.Dataset({"tas": (("lat", "lon"), np.zeros((len(lat), len(lon))))}, coords={"lon": lon, "lat": lat}).to_netcdf(gridfile)
    atmdict = {"maskfile": "", "gridfile": str(gridfile), "areafile": ""}
    ocedict = {"maskfile": "", "gridfile": "", "areafile": ""}

    support = Supporter({"atm": "cmoratm", "oce": "cmoroce"}, atmdict, ocedict, areas=True, remap=False)

    assert support.atmmask is None
    assert support.atmarea is not None


def test_supporter_gridfile_from_data_file(tmp_path):
    """A data file with a time axis can be used as gridfile: only its first time step is loaded."""
    gridfile = tmp_path / "tas.nc"
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.arange(0.5, 360.0, 1.0)
    time = np.array(["1990-01-15", "1990-02-15", "1990-03-15"], dtype="datetime64[ns]")
    xr.Dataset(
        {"tas": (("time", "lat", "lon"), np.zeros((len(time), len(lat), len(lon))))},
        coords={"time": time, "lon": lon, "lat": lat},
    ).to_netcdf(gridfile)
    atmdict = {"maskfile": "", "gridfile": str(gridfile), "areafile": ""}
    ocedict = {"maskfile": "", "gridfile": "", "areafile": ""}

    support = Supporter({"atm": "cmoratm", "oce": "cmoroce"}, atmdict, ocedict, areas=True, remap=False)

    assert "time" not in support.atmfield.dims
    assert support.atmarea.dims == ("lat", "lon")
