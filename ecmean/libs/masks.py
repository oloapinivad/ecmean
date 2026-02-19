#!/usr/bin/env python3
'''
Shared functions for masking in ecmean
'''

import logging
import xarray as xr

##########################
# MASK-RELATED FUNCTIONS #
##########################

loggy = logging.getLogger(__name__)


def masked_meansum(xfield, weights, mask, operation, domain, mask_type):
    """For global variables evaluate the weighted averaged
    or weighted integral when required by the variable properties"""

    # call the mask_field to mask where necessary
    # the mask field is area
    masked = mask_field(xfield, mask_type, domain, mask)

    # no time dimensions
    notimedim = [dim for dim in xfield.dims if dim not in ['time', 'year', 'month']]

    # global mean
    if operation in ['average', 'mean']:
        return masked.weighted(weights.fillna(0)).mean(dim=notimedim).data
    # global integrals
    if operation in ['integral', 'sum']:
        return masked.weighted(weights.fillna(0)).sum(dim=notimedim).data

    raise ValueError("ERROR: masked_meansum-> mask undefined, this cannot be handled!")

def mask_field(xfield, mask_type, dom, mask):
    """
    Apply a land/sea or hemisphere mask to an xarray DataArray or Dataset.

    Args:
        xfield (xr.DataArray or xr.Dataset): Input data to mask.
        mask_type (str): Type of mask to apply (e.g., 'land', 'sea', 'north').
        dom (str): Domain type ('oce' or 'atm').
        mask (xr.DataArray): Land-sea mask with values (1 = land, 0 = sea).

    Returns:
        xr.DataArray or xr.Dataset: Masked data.

    Raises:
        ValueError: If the mask_type or domain is invalid.
    """

    # Global — no mask applied
    if mask_type == 'global':
        return xfield

    # Hemisphere masks
    if mask_type == 'north':
        return xfield.where(xfield['lat'] > 0)
    if mask_type == 'south':
        return xfield.where(xfield['lat'] < 0)

    # Oceanic domain: optional ocean mask
    if dom == 'oce':
        if isinstance(mask, xr.DataArray):
            return xfield.where(mask < 0.5)
        return xfield  # no mask provided

    # Atmospheric domain: land/sea/land-no-antarctica
    if dom == 'atm':
        if not isinstance(mask, xr.DataArray):
            raise ValueError("Land-sea mask must be provided for atmospheric masking.")

        if mask_type == 'land':
            return xfield.where(mask >= 0.5)
        if mask_type == 'land-no-antarctica':
            return xfield.where(mask['lat'] > -60).where(mask >= 0.5)
        if mask_type in ['sea', 'ocean']:
            return xfield.where(mask < 0.5)

        raise ValueError(f"Invalid mask_type '{mask_type}' for domain 'atm'.")

    raise ValueError(f"Unknown domain '{dom}'. Expected 'oce' or 'atm'.")

def select_region(xfield, region):
    """
    Selects a latitude-defined region from an xarray object.

    Args:
        xfield (xarray.DataArray or xarray.Dataset): Input data with a latitude dimension.
        region (str): Name of the region to select.

    Returns:
        xarray object: Subset of xfield for the selected region.

    Raises:
        KeyError: If the region is unknown.
        AttributeError: If the input data does not contain 'lat'.
    """
    region_bounds = {
        'Global': (-90.0, 90.0),
        'NH': (20.0, 90.0),
        'SH': (-90.0, -20.0),
        'Equatorial': (-20.0, 20.0),
        'Tropical': (-30.0, 30.0),
        'North Midlat': (30.0, 90.0),
        'South Midlat': (-90.0, -30.0),
        'North Pole': (60.0, 90.0),
        'South Pole': (-90.0, -60.0)
    }

    if 'lat' not in xfield.coords:
        raise AttributeError("Input xarray object does not contain 'lat' coordinate.")

    if region not in region_bounds:
        raise KeyError(f"Region '{region}' is not supported. Choose from: {list(region_bounds.keys())}")

    lat_min, lat_max = region_bounds[region]

    # Apply latitude mask
    return xfield.where((xfield.lat >= lat_min) & (xfield.lat <= lat_max))

