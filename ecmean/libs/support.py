#!/usr/bin/env python3
'''
Shared functions for Support class for ECmean4
'''

import os
from glob import glob
import logging
import xarray as xr
import xesmf as xe
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.files import inifiles_priority
from ecmean.libs.areas import AreaCalculator
from ecmean.libs.interpolators import InterpolatorFactory

loggy = logging.getLogger(__name__)

# mask to be searched
atm_mask_names = ['lsm', 'sftlf']
oce_mask_names = ['lsm', 'sftof', 'mask_opensea']


class Supporter():
    """
    Support class for ECmean4, including areas and masks to be used
    in global mean and performance indices
    """

    def __init__(self, component, atmdict, ocedict, tool='ESMF', areas=True, remap=False, targetgrid=None):
        """Class for masks, areas and interpolation (xESMF-based)
        for both atmospheric and oceanic component"""

        loggy.debug('Running with xesmf version %s', xe.__version__)

        # define the basics
        self.atmareafile = inifiles_priority(atmdict)
        self.oceareafile = inifiles_priority(ocedict)
        self.atmmaskfile = atmdict['maskfile']
        self.ocemaskfile = ocedict['maskfile']
        self.atmcomponent = component['atm']
        self.ocecomponent = component['oce']
        self.targetgrid = targetgrid
        self.tool = tool.upper()

        # remapping default
        self.atminterpolator = None
        self.oceinterpolator = None

        # areas and mask for amip case
        self.ocemask = None
        self.ocearea = None
        
        if self.atmareafile:
            # loading and examining atmospheric file
            self.atmfield = self.load_area_field(self.atmareafile, comp='atm')
            self.atmgridtype = identify_grid(self.atmfield)
            loggy.info('Atmosphere grid is is a %s grid!', self.atmgridtype)

            # compute atmopheric area
            if areas:
                self.atmarea = self.make_areas(self.atmgridtype, self.atmfield)

            # initialize the interpolation for atmosphere
            if self.targetgrid and remap:
                self.atminterpolator = InterpolatorFactory.create_interpolator(
                    'atmospheric', self.atmcomponent, self.atmgridtype, self.targetgrid, self.tool
                )
                if self.tool == 'ESMF':
                    self.atminterpolator.create_weights(self.atmfield)
                elif self.tool == 'CDO':
                    self.atminterpolator.create_weights(self.atmareafile)

            # init the land-sea mask for atm (mandatory)
            self.atmmask = self.make_atm_masks()

        # do the same if oceanic file is found
        if self.oceareafile:
            self.ocefield = self.load_area_field(self.oceareafile, comp='oce')
            self.ocegridtype = identify_grid(self.ocefield)
            loggy.info('Oceanic grid is is a %s grid!', self.ocegridtype)

            # compute oceanic area
            if areas:
                self.ocearea = self.make_areas(self.ocegridtype, self.ocefield)

            # init the ocean interpolation
            if self.targetgrid and remap:
                self.oceinterpolator = InterpolatorFactory.create_interpolator(
                    'oceanic', self.ocecomponent, self.ocegridtype, self.targetgrid, self.tool
                )
                if self.tool == 'ESMF':
                    self.oceinterpolator.create_weights(self.ocefield)
                elif self.tool == 'CDO':
                    self.oceinterpolator.create_weights(self.oceareafile)

            # ocean mask
            if self.ocemaskfile:
                self.ocemask = self.make_oce_masks()
            else:
                # if it is missing, when remapping I can use the atmospheric one
                if self.targetgrid and remap:
                    self.ocemask = self.atmmask
                # otherwise, no solution!
                else:
                    loggy.warning('No mask available for oceanic vars, this might lead to inconsistent results...')

    def make_atm_masks(self):
        """Create land-sea masks for atmosphere model"""

        # prepare ATM LSM
        if self.atmmaskfile:
            loggy.info('maskatmfile is %s', self.atmmaskfile)
        self.atmmaskfile = check_file_exist(self.atmmaskfile)

        if self.atmcomponent == 'oifs':
            # create mask: opening a grib and loading only lsm to avoid
            # inconsistencies in the grib structure ->
            # see here https://github.com/ecmwf/cfgrib/issues/13
            mask = xr.open_mfdataset(
                self.atmmaskfile,
                engine="cfgrib",
                indexpath=None,
                filter_by_keys={
                    'shortName': 'lsm'},
                preprocess=xr_preproc)['lsm']

        elif self.atmcomponent in ['cmoratm', 'globo']:

            dmask = xr.open_mfdataset(self.atmmaskfile, preprocess=xr_preproc)
            mvar = [var for var in dmask.data_vars if var in atm_mask_names]
            # the case we cannot find the variable we are looking for in the required file
            if len(mvar) > 0:
                mask = fix_mask_values(dmask[mvar[0]])
            else:
                raise KeyError(f"ERROR: make_atm_masks -> Cannot find mask variable in {self.atmmaskfile}")

        else:
            raise KeyError("ERROR: make_atm_masks -> Atmospheric component not supported!")

        # loading the mask to avoid issues with xesmf
        mask = mask.load()

        # interp the mask if required
        if self.atminterpolator is not None:
            mask = self.atminterpolator.interpolate(mask, keep_attrs=True)

        return mask

    def make_oce_masks(self):
        """Create land-sea masks for oceanic model. This is used only for CMIP"""

        # prepare ocean LSM:
        if self.ocemaskfile:
            loggy.info('maskocefile is %s', self.ocemaskfile)
        self.ocemaskfile = check_file_exist(self.ocemaskfile)

        if self.ocecomponent in ['cmoroce', 'nemo']:
            dmask = xr.open_mfdataset(self.ocemaskfile, preprocess=xr_preproc)

            mvar = [var for var in dmask.data_vars if var in oce_mask_names]
            # the case we cannot find the variable we are looking for in the required file
            if len(mvar) > 0:
                mask = fix_mask_values(dmask[mvar[0]])
            else:
                loggy.warning(
                    'No mask array found in %s for oceanic vars, this might lead to inconsistent results...',
                    self.ocemaskfile)
                return None

        else:
            raise KeyError("ERROR: make_oce_masks -> Oceanic component not supported!")

        # load mask to avoid issue with xesmf
        mask = mask.load()

        # interp the mask if required
        if self.oceinterpolator is not None:
            mask = self.oceinterpolator.interpolate(mask, keep_attrs=True)

        return mask

    def load_area_field(self, areafile, comp):
        """Loading files for area and interpolation"""

        if areafile:
            loggy.info('%smareafile is %s', comp, areafile)
        areafile = check_file_exist(areafile)
        return xr.open_mfdataset(areafile, preprocess=xr_preproc).load()

    def make_areas(self, gridtype, xfield):
        """Create weights for area operations.
        Minimal structure."""

        # this might be universal, but keep this as for supported components only
        if 'areacello' in xfield.data_vars:  # as oceanic CMOR case
            return xfield['areacello']
        elif 'cell_area' in xfield.data_vars:  # as ECE4 NEMO case for nemo-initial-state.nc
            return xfield['cell_area']
        elif 'e1t' in xfield.data_vars:  # ECE4 NEMO case for domaing_cfg.nc
            return xfield['e1t'] * xfield['e2t']
        else:  # automatic solution, wish you luck!
            return AreaCalculator().calculate_area(xfield, gridtype)

    
def fix_mask_values(mask):
    """
    Function to normalize the mask whatever format. By convention in ECmean
    masks are 1 over land and 0 over the ocean. It might cause a bit of slowdown since we
    need to load the data
    """

    if 'time' in mask.dims:
        mask = mask.isel(time=0).squeeze()

    # safety filler
    mask = mask.fillna(0)

    # if it is a percentage
    if mask.max() > 99:
        loggy.info('%s is being normalized', mask.name)
        mask = mask / 100

    # if it is an ocean mask
    if mask.mean() > 0.5:
        loggy.info('%s is being flipped', mask.name)
        mask = abs(1 - mask)

    return mask


def check_file_exist(file):
    """Simple check to verify that a file to be loaded is defined and found on disk"""

    if file is None:
        raise KeyError("ERROR: file not defined!")
    file = glob(file)[0]
    if not os.path.isfile(file):
        raise KeyError(f"ERROR: {file} cannot be found")
    return file


def identify_grid(xfield):
    """Receiveng an xarray object (DataArray or Dataset) investigates its coordinates
    and dimensions and provide the grid type (regular, gaussian, curvilinear,
    gaussian reduced, unstructured). It assumes that data is defined by 'lon' and 'lat'
    dimensions

    Args :
        xfield:

    Returns
        string with the grid type

    """

    # if the coordinates are lon/lat proceed
    if all(x in xfield.coords for x in ['lon', 'lat']):

        # if dimensions are lon/lat as well, this is a regular grid
        if all(x in xfield.dims for x in ['lon', 'lat']):
            lat = xfield.coords['lat']

            # if lat grid spacing is equal, is regular lonlat, otherwise gaussian
            if (lat[3] - lat[2]) == (lat[1] - lat[0]):
                return 'lonlat'
            return 'gaussian'
        
        # if the coords are 2D, we are curvilinear
        if xfield.coords['lon'].ndim == 2 and xfield.coords['lon'].ndim == 2:
            return 'curvilinear'
           
        # check the first four elements of the grid (arbitrary)
        lat = xfield.coords['lat'].values[0:5]

        # if they are all the same, we have a gaussian reduced, else unstructured
        if (lat == lat[0]).all():
            return 'gaussian_reduced'

        return 'unstructured'
    
    raise ValueError("Cannot find any lon/lat dimension, aborting...")
