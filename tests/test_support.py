"""Unit tests for the helper functions of ecmean.libs.support."""

import numpy as np
import pytest
import xarray as xr

from ecmean.libs.interpolators import cdo_available, cdo_method
from ecmean.libs.support import Supporter, check_file_exist, identify_grid

# interpolation tools to test, CDO only when a recent enough executable is available
TOOLS = ["ESMF", pytest.param("CDO", marks=pytest.mark.skipif(not cdo_available(), reason="CDO >= 2.6.2 not available"))]


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


@pytest.mark.parametrize(
    "gridtype,method", [("lonlat", "bil"), ("gaussian", "bil"), ("curvilinear", "bil"), ("unstructured", "nn")]
)
def test_cdo_method(gridtype, method):
    """Bilinear everywhere but on unstructured grids, as ESMF."""
    assert cdo_method(gridtype) == method


@pytest.mark.parametrize("tool", TOOLS)
def test_supporter_omip_without_masks(tool):
    """OMIP: ocean grid only, no atmospheric files and no ocean mask, with remapping."""
    lon, lat = np.arange(0.5, 360, 10.0), np.arange(-85, 90, 10.0)
    targetgrid = xr.Dataset(
        {"tas": (("lat", "lon"), np.zeros((len(lat), len(lon))))},
        coords={"lon": lon, "lat": lat},
    )
    atmdict = {"maskfile": "", "gridfile": "", "areafile": ""}
    ocedict = {"maskfile": "", "gridfile": "", "areafile": "tests/data/omip/domain_cfg.nc"}

    support = Supporter(
        {"atm": "oifs", "oce": "nemo"}, atmdict, ocedict, areas=False, remap=True, targetgrid=targetgrid, tool=tool
    )

    assert support.atmmask is None
    assert support.atmarea is None
    assert support.ocemask is None
    assert support.oceinterpolator is not None
