#!/usr/bin/env python3
'''
Shared functions for XArray ecmean
'''

import logging
import numpy as np
import xarray as xr

##################################
# AREA-WEIGHT FUNCTIONS #
##################################

# def cdo_identify_grid(file):
#     cdogrid = cdo.sinfo(input=os.path.join(dir, file))
#     for gg in ['unstructured', 'lonlat', 'curvilinear', 'gaussian_reduced', 'gaussian']:
#         grid_type = [gg for i in cdogrid if gg in i]
#         if grid_type :
#             break
#     return grid_type[0]

loggy = logging.getLogger(__name__)


def guess_bounds(axis, name='lon'):
    """Basic function that estimates the boundaries for lon and lat if they are not
    available. Works only with regular grids.
    It also avoid having values larger than 90N/90S as well as 0 and 10^5 Pa"""
    # inspired by
    # https://gist.github.com/dennissergeev/60bf7b03443f1b2c8eb96ce0b1880150

    # this define the proportion of the bounds, assumed to half of the levels
    bound_position = 0.5
    diffs = np.diff(axis)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    # pair of dounds
    min_bounds = axis - diffs[:-1] * bound_position
    max_bounds = axis + diffs[1:] * (1 - bound_position)

    # safety check, to be generalized
    if name in 'lat':
        max_bounds[-1] = 90
        min_bounds[0] = -90
    if name in 'plev':
        min_bounds[0] = 0
        max_bounds[-1] = 100000

    # should we use a xarray object instead of a numpy?
    bounds = np.array([min_bounds, max_bounds]).transpose()
    return bounds

class AreaCalculator:
    """
    Class to calculate the area of grid cells on a spherical Earth.
    It can handle different grid types and formulas for area calculation.
    The default Earth radius is set to 6371000 meters (mean radius).
    """
    def __init__(self, earth_radius=6371000.0):
        self.earth_radius = earth_radius
        self.dlon = None
        self.dlat = None

    @staticmethod
    def _lonlat_to_sphere(lon, lat):
        """Convert longitude and latitude to 3D Cartesian coordinates on a unit sphere."""
        return np.array([
            np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lat))
        ])

    @staticmethod
    def _huilier(a, b, c):
        """L'Huilier's formula to compute the area of a spherical triangle given its sides a, b, c."""
        s = (a + b + c) * 0.5
        t = np.tan(s * 0.5) * np.tan((s - a) * 0.5) * np.tan((s - b) * 0.5) * np.tan((s - c) * 0.5)
        return abs(4. * np.arctan(np.sqrt(abs(t))))

    def _vector_spherical_triangle(self, p1, p2, p3):
        """Compute the area of a spherical triangle defined by three points on the sphere using vector cross products."""
        a = np.arcsin(np.linalg.norm(np.cross(p1, p2), axis=1))
        b = np.arcsin(np.linalg.norm(np.cross(p1, p3), axis=1))
        c = np.arcsin(np.linalg.norm(np.cross(p3, p2), axis=1))
        return self._huilier(a, b, c)

    def _area_by_triangles(self, bounds_lon, bounds_lat):
        """Calculate the area of grid cells by dividing them into two spherical triangles."""
        p1 = self._lonlat_to_sphere(bounds_lon[:, 0], bounds_lat[:, 0]).T
        p2 = self._lonlat_to_sphere(bounds_lon[:, 0], bounds_lat[:, 1]).T
        p3 = self._lonlat_to_sphere(bounds_lon[:, 1], bounds_lat[:, 1]).T
        p4 = self._lonlat_to_sphere(bounds_lon[:, 1], bounds_lat[:, 0]).T

        area = self._vector_spherical_triangle(p1, p2, p3) + \
               self._vector_spherical_triangle(p1, p4, p3)
        return area * self.earth_radius**2

    # def _area_by_trapezoids(self, bounds_lat):
    #     arclon1 = self.earth_radius * 
    # np.abs(np.cos(np.deg2rad(bounds_lat[:, 0]))) * np.deg2rad(self.dlon)
    #     arclon2 = self.earth_radius * 
    # np.abs(np.cos(np.deg2rad(bounds_lat[:, 1]))) * np.deg2rad(self.dlon)
    #     arclat = self.earth_radius * np.deg2rad(self.dlat)

    #     return (arclon1 + arclon2) * arclat / 2

    # def _area_by_squares(self, full_lat):

    #     arclon = self.earth_radius * np.abs(np.cos(np.deg2rad(full_lat))) * np.deg2rad(self.dlon)
    #     arclat = self.earth_radius * np.deg2rad(self.dlat)

    #     return arclon * arclat

    def _area_computation(self, bounds_lon, bounds_lat, formula='triangles', full_lat=None):
        """
        Compute the area of grid cells based on the specified formula. 
        Only triangles is activated, other older method are kept for reference but not tested.
        """

        if formula == 'triangles':
            return self._area_by_triangles(bounds_lon, bounds_lat)

        # self.dlon=abs(bounds_lon[:, 0] - bounds_lon[:, 1])
        # self.dlat=abs(bounds_lat[:, 0] - bounds_lat[:, 1])

        # if formula == 'trapezoids':
        #     return self._area_by_trapezoids(bounds_lat)
        # if formula == 'squares':
        #     if full_lat is None:
        #         raise ValueError("full_lat is required for square-based area computation.")
        #     return self._area_by_squares(full_lat)

        raise ValueError(f"Unknown area formula: {formula}")

    def calculate_area(self, xfield, gridtype=None, formula='triangles'):
        """
        Calculate the area of grid cells in an xarray DataArray or Dataset.
        Args:
            xfield (xr.DataArray or xr.Dataset): Input data with grid coordinates.
            gridtype (str): Type of grid ('unstructured', 'gaussian_reduced', 'curvilinear',
                            'gaussian', 'lonlat').
            formula (str): Formula for area calculation ('triangles', 'trapezoids', 'squares').
        Returns:
            xr.DataArray: Area of grid cells.
        """
        if 'time' in xfield.dims:
            xfield = xfield.isel(time=0)

        cmor_bounds = all(x in xfield.data_vars for x in ['lon_bnds', 'lat_bnds'])

        if gridtype in ['unstructured', 'gaussian_reduced', 'curvilinear']:
            blondim = next((t for t in xfield.data_vars if t in ['lon_bnds', 'bounds_lon']), None)
            blatdim = next((t for t in xfield.data_vars if t in ['lat_bnds', 'bounds_lat']), None)

            if blondim is None or blatdim is None:
                raise ValueError("Cannot find lon/lat bounds for unstructured grid.")

            bounds_lon = np.column_stack((xfield[blondim].isel(nvertex=1),
                                          xfield[blondim].isel(nvertex=2)))
            bounds_lat = np.column_stack((xfield[blatdim].isel(nvertex=2),
                                          xfield[blatdim].isel(nvertex=3)))
            area_dims = 'cell'
            full_lat = xfield['lat'].data

        elif gridtype in ['gaussian', 'lonlat']:
            if cmor_bounds and 'bnds' not in xfield.dims:
                bdim = [g for g in xfield.dims if g not in ['lon', 'lat', 'time']][0]
                xfield = xfield.rename_dims({bdim: "bnds"})

            if not cmor_bounds:
                xfield['lat_bnds'] = (('lat', 'bnds'), guess_bounds(xfield['lat'], name='lat'))
                xfield['lon_bnds'] = (('lon', 'bnds'), guess_bounds(xfield['lon'], name='lon'))

            blon = np.column_stack((xfield['lon_bnds'].isel(bnds=0), xfield['lon_bnds'].isel(bnds=1)))
            blat = np.column_stack((xfield['lat_bnds'].isel(bnds=0), xfield['lat_bnds'].isel(bnds=1)))
            full_lat = np.repeat(xfield['lat'].data, len(xfield['lon']), axis=0)

            bounds_lon = np.stack([blon] * len(blat), axis=0).reshape(-1, 2)
            bounds_lat = np.stack([blat] * len(blon), axis=1).reshape(-1, 2)
            area_dims = ('lat', 'lon')

        else:
            raise ValueError("Gridtype undefined or unsupported.")

        area = self._area_computation(bounds_lon, bounds_lat, formula=formula, full_lat=full_lat)

        if gridtype in ['gaussian', 'lonlat']:
            area = area.reshape([len(xfield['lat']), len(xfield['lon'])])

        outfield = xr.DataArray(area, dims=area_dims, coords=xfield.coords, name='area')

        loggy.debug("Total Earth Surface: %s Km2", str(outfield.sum().values / 10**6))

        return outfield
