#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dask-aware utility functions for handling xarray operations safely.

This module provides helper functions to handle common dask-related issues
when working with xarray data arrays, particularly for vertical interpolation
and scalar value extraction.

@author Paolo Davini (p.davini@isac.cnr.it), Aug 2025
"""

import logging
import numpy as np


loggy = logging.getLogger(__name__)


def vertical_interpolation(final, cfield, var):
    """
    Handle vertical interpolation for 3D fields with improved error handling.
    
    Args:
        final (xarray.DataArray): The data array to interpolate
        cfield (xarray.DataArray): The climatology field with target levels
        var (str): Variable name for logging
        
    Returns:
        xarray.DataArray: Interpolated data array
    """
    final = final.metpy.convert_coordinate_units('plev', 'Pa')
    
    # Get coordinate values - compute if dask arrays
    final_plevs = final['plev'].compute() if hasattr(final['plev'], 'compute') else final['plev']
    cfield_plevs = cfield['plev'].compute() if hasattr(cfield['plev'], 'compute') else cfield['plev']
    
    # Check if interpolation is needed using efficient numpy comparison
    if not np.array_equal(np.sort(final_plevs), np.sort(cfield_plevs)):
        loggy.warning('%s: Need to interpolate vertical levels...', var)
        final = final.interp(plev=cfield_plevs, method='linear')
        
        # Check for NaN values after interpolation
        check_interpolation_quality(final, var)
    
    return final



def check_interpolation_quality(data_array, var):
    """
    Check the quality of interpolated data by looking for NaN values.
    
    Args:
        data_array (xarray.DataArray): The interpolated data array
        var (str): Variable name for logging
    """
    # Select a small representative sample and compute it if needed
    if 'lon' in data_array.dims and 'lat' in data_array.dims:
        sample = data_array.isel(lon=0, lat=0)
    else:
        # If no spatial dimensions, take the first value along non-plev dims
        sample = data_array.isel({dim: 0 for dim in data_array.dims if dim != 'plev'})
    
    # Compute if it's a dask array
    if hasattr(sample, 'compute'):
        sample = sample.compute()
    
    # Count NaN values using numpy
    nan_count = np.sum(np.isnan(sample))
    
    if nan_count > 0:
        loggy.warning(
            '%s: Found %d NaN values after vertical interpolation, this may affect your PIs...', 
            var, nan_count)
        
        # Find which levels have NaN
        nan_mask = np.isnan(sample)
        if np.any(nan_mask) and hasattr(sample, 'plev'):
            plev_values = sample.plev.values if hasattr(sample.plev, 'values') else sample.plev
            nan_levels = plev_values[nan_mask]
            if len(nan_levels) > 0:
                loggy.warning('NaN found at pressure levels: %s Pa', nan_levels)


def extract_scalar(data_array):
    """
    Safely extract a scalar value from a data array, handling dask arrays properly.
    
    Args:
        data_array (xarray.DataArray): The data array to extract from
        
    Returns:
        float: The extracted scalar value
    """
    # If it's a dask array, compute it first
    if hasattr(data_array, 'compute'):
        computed = data_array.compute()
        return float(computed.item())
    else:
        return float(data_array.item())
