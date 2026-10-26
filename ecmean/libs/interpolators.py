"""
Interpolation classes for ECmean4: ESMF (xESMF) and CDO (smmregrid) interpolators
sharing the same interface, so that the diagnostics do not depend on the tool.
"""

import logging

import numpy as np
import xarray as xr
import xesmf as xe
from smmregrid import CdoGenerate, Regridder, check_cdo_version

loggy = logging.getLogger(__name__)


def cdo_available():
    """Whether a CDO executable recent enough for smmregrid is found (see `smmregrid.check_cdo_version`).

    Returns:
        bool: True when the `CDO` tool can be used.
    """
    # two clauses: ruff (target py314) would strip the parentheses of a tuple, which breaks python < 3.14
    try:
        check_cdo_version()
    except FileNotFoundError:
        return False
    except RuntimeError:
        return False
    return True


def select_grid_variable(xfield):
    """Select the variable describing the grid from the field used for the weights.

    Args:
        xfield (xarray.Dataset or xarray.DataArray): preprocessed field of the area/mask/grid file.

    Returns:
        xarray.DataArray: the cell area when available (CMOR `areacello`/`areacella`, ECE4 NEMO `cell_area`),
            otherwise the last variable of the dataset with lon and lat coordinates (bounds are skipped),
            or the DataArray itself.

    Raises:
        KeyError: if no variable with lon and lat coordinates is found.
    """
    if isinstance(xfield, xr.DataArray):
        return xfield
    for name in ("areacello", "areacella", "cell_area"):
        if name in xfield.data_vars:
            return xfield[name]
    for name in reversed(list(xfield.data_vars)):
        if all(coord in xfield[name].coords for coord in ("lon", "lat")):
            return xfield[name]
    raise KeyError(f"No variable with lon/lat coordinates to describe the grid among {list(xfield.data_vars)}")


def gaussian_regular_grid(xfield):
    """Regular gaussian grid with the latitudes of a gaussian reduced field and the longitudes of its
    equatorial latitude circle, used as intermediate grid to interpolate from the reduced grid.

    Args:
        xfield (xarray.DataArray): field on the gaussian reduced grid, with 1D lon/lat coordinates on the cells.

    Returns:
        xarray.Dataset: dataset with the 1D lon and lat coordinates of the regular gaussian grid
            and a dummy variable on them (CDO needs a variable to describe a grid).
    """
    m = xfield.isel(time=0).load() if "time" in xfield.dims else xfield.load()
    g = np.unique(m.lat.data)
    f = np.unique(m.sel(cell=m.lat == g[int(len(g) / 2)]).lon.data)
    grid = xr.Dataset({"grid": (["lat", "lon"], np.zeros((len(g), len(f))))}, coords={"lon": f, "lat": g})
    return cf_lonlat(grid)


def cf_lonlat(xfield):
    """Add the CF units and standard_name to the lon/lat coordinates, needed by CDO to
    recognise the grid (missing e.g. in the NEMO `domain_cfg.nc`).

    Args:
        xfield (xarray.Dataset or xarray.DataArray): field with lon and lat coordinates.

    Returns:
        xarray.Dataset or xarray.DataArray: the same field with CF attributes on lon and lat.
    """
    return xfield.assign_coords(
        lon=xfield.lon.assign_attrs(units="degrees_east", standard_name="longitude"),
        lat=xfield.lat.assign_attrs(units="degrees_north", standard_name="latitude"),
    )


def cdo_method(gridtype):
    """CDO remapping method consistent with the ESMF one for a given grid type.

    ESMF uses bilinear interpolation, falling back to nearest neighbour on unstructured grids;
    CDO cannot do bilinear interpolation on these grids either, so nearest neighbour is used.
    The gaussian reduced grid is handled in two steps by both tools (see `gaussian_regular_grid`).

    Args:
        gridtype (str): grid type as returned by `support.identify_grid`.

    Returns:
        str: `nn` for unstructured grids, `bil` otherwise.
    """
    return "nn" if gridtype == "unstructured" else "bil"


class BaseInterpolator:
    """Base class for all interpolators"""

    def __init__(self, component, gridtype, targetgrid):
        self.component = component
        self.gridtype = gridtype
        self.targetgrid = targetgrid
        self.fix = None
        self.remap = None
        self.is_cdo = False  # Flag to identify CDO interpolators

    def interpolate(self, data, keep_attrs=True):
        """Apply the (optional) fix and the remap steps to data; keep_attrs is for ESMF only"""
        kwargs = {} if self.is_cdo else {"keep_attrs": keep_attrs}
        if callable(self.fix):
            data = self.fix(data, **kwargs)
        if callable(self.remap):
            data = self.remap(data, **kwargs)
        return data


class ESMFInterpolator(BaseInterpolator):
    """ESMF interpolator for both atmospheric and oceanic data"""

    def __init__(self, domain, component, gridtype, targetgrid):
        super().__init__(component, gridtype, targetgrid)
        self.domain = domain.lower()

    def create_weights(self, xfield):
        """Create ESMF interpolation weights based on domain"""

        if self.domain in ["atm", "atmospheric", "atmosphere"]:
            self._create_atmospheric_weights(xfield)
        elif self.domain in ["oce", "oceanic", "ocean"]:
            self._create_oceanic_weights(xfield)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

        loggy.info("Created ESMF %s interpolator for %s grid", self.domain, self.gridtype)

    def _create_atmospheric_weights(self, xfield):
        """Create ESMF atmospheric interpolation weights"""

        if self.component == "oifs":
            # Handle OpenIFS gaussian reduced grid: two-step interpolation,
            # nearest neighbour to the regular gaussian grid, then bilinear to target
            xfield = select_grid_variable(xfield)
            gaussian_regular = gaussian_regular_grid(xfield)
            self.fix = xe.Regridder(xfield, gaussian_regular, method="nearest_s2d", locstream_in=True, periodic=True)
            self.remap = xe.Regridder(self.fix(xfield), self.targetgrid, periodic=True, method="bilinear")

        elif self.component in ["cmoratm", "globo"]:
            # Handle regular atmospheric grids
            self.fix = None
            self.remap = xe.Regridder(xfield, self.targetgrid, periodic=True, method="bilinear")

        else:
            raise KeyError(f"ESMF atmospheric weights not defined for component '{self.component}'")

    def _create_oceanic_weights(self, xfield):
        """Create ESMF oceanic interpolation weights"""

        if self.component not in ["nemo", "cmoroce"]:
            raise KeyError(f"ESMF oceanic weights not defined for component '{self.component}'")

        xfield = select_grid_variable(xfield)

        # Choose interpolation method based on grid type
        if self.gridtype == "unstructured":
            self.remap = xe.Regridder(xfield, self.targetgrid, method="nearest_s2d", locstream_in=True, periodic=True)
        else:
            # Regular or curvilinear grids
            self.remap = xe.Regridder(xfield, self.targetgrid, method="bilinear", periodic=True, ignore_degenerate=True)

        self.fix = None


class CDOInterpolator(BaseInterpolator):
    """CDO (smmregrid) interpolator for both atmospheric and oceanic data"""

    def __init__(self, domain, component, gridtype, targetgrid):
        super().__init__(component, gridtype, targetgrid)
        self.domain = domain.lower()
        self.is_cdo = True  # Mark as CDO interpolator

    def create_weights(self, xfield):
        """Create CDO interpolation weights from the same preprocessed field used by ESMF.

        The variable describing the grid is selected as for ESMF, reduced to a single time step,
        and its lon/lat coordinates (as those of the target grid) are given the CF attributes CDO
        needs to recognise the grid (missing e.g. in the NEMO `domain_cfg.nc`). The remapping method follows the grid type
        (see `cdo_method`), and the gaussian reduced grid goes through the regular gaussian grid
        as in ESMF; `skipna=True` lets smmregrid skip missing source points and renormalise the
        weights, as ESMF does with its masked grids.

        Args:
            xfield (xarray.Dataset or xarray.DataArray): preprocessed field of the area/mask/grid file.
        """

        if self.domain not in ["atm", "atmospheric", "atmosphere", "oce", "oceanic", "ocean"]:
            raise ValueError(f"Unknown domain: {self.domain}")

        grid = select_grid_variable(xfield)
        if "time" in grid.dims:
            grid = grid.isel(time=0, drop=True)
        grid = cf_lonlat(grid)

        self.fix = None
        if self.gridtype == "gaussian_reduced":
            # two-step interpolation: nearest neighbour to the regular gaussian grid, then bilinear
            gaussian_regular = gaussian_regular_grid(grid)
            weights = CdoGenerate(grid, gaussian_regular, skipna=True).weights(method="nn")
            self.fix = Regridder(weights=weights, skipna=True).regrid
            grid = self.fix(grid)
        method = cdo_method(self.gridtype)
        weights = CdoGenerate(grid, cf_lonlat(self.targetgrid), skipna=True).weights(method=method)
        self.remap = Regridder(weights=weights, skipna=True).regrid

        loggy.info("Created CDO %s interpolator for %s grid with method %s", self.domain, self.gridtype, method)


class InterpolatorFactory:
    """Factory class for creating appropriate interpolators"""

    @staticmethod
    def create_interpolator(domain, component, gridtype, targetgrid, tool="ESMF"):
        """Create the appropriate interpolator based on tool

        Args:
            domain (str): 'atmospheric' or 'oceanic'
            component (str): Component name (e.g., 'oifs', 'nemo', etc.)
            gridtype (str): Grid type (e.g., 'gaussian', 'unstructured', etc.)
            targetgrid: Target grid for interpolation
            tool (str): Interpolation tool ('ESMF' or 'CDO')

        Returns:
            Appropriate interpolator instance
        """

        tool = tool.upper()

        if tool == "ESMF":
            return ESMFInterpolator(domain, component, gridtype, targetgrid)
        if tool == "CDO":
            return CDOInterpolator(domain, component, gridtype, targetgrid)
        raise ValueError(f"Unknown tool: {tool}. Use 'ESMF' or 'CDO'")
