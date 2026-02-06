#!/usr/bin/env python3
'''
Interpolation classes for ECmean4 - simple inheritance design
'''

import logging
from glob import glob
import numpy as np
import xarray as xr
import xesmf as xe
from smmregrid import cdo_generate_weights, Regridder

loggy = logging.getLogger(__name__)


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
        """Apply interpolation to data"""
        if callable(self.fix):
            data = self.fix(data, keep_attrs=keep_attrs)
        if callable(self.remap):
            if self.is_cdo:
                # CDO doesn't support keep_attrs
                return self.remap(data)
            # ESMF supports keep_attrs
            return self.remap(data, keep_attrs=keep_attrs)
        return data


class ESMFInterpolator(BaseInterpolator):
    """ESMF interpolator for both atmospheric and oceanic data"""
    
    def __init__(self, domain, component, gridtype, targetgrid):
        super().__init__(component, gridtype, targetgrid)
        self.domain = domain.lower()
    
    def create_weights(self, xfield):
        """Create ESMF interpolation weights based on domain"""
        
        if self.domain in ['atm', 'atmospheric', 'atmosphere']:
            self._create_atmospheric_weights(xfield)
        elif self.domain in ['oce', 'oceanic', 'ocean']:
            self._create_oceanic_weights(xfield)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
        
        loggy.info('Created ESMF %s interpolator for %s grid', self.domain, self.gridtype)
    
    def _create_atmospheric_weights(self, xfield):
        """Create ESMF atmospheric interpolation weights"""
        
        if self.component == 'oifs':
            # Handle OpenIFS gaussian reduced grid
            xname = list(xfield.data_vars)[-1]
            m = xfield[xname].isel(time=0).load()
            
            # Get unique lon/lat from equator
            g = np.unique(m.lat.data)
            f = np.unique(m.sel(cell=m.lat == g[int(len(g) / 2)]).lon.data)
            
            # Create gaussian regular grid
            gaussian_regular = xr.Dataset({
                "lon": (["lon"], f), 
                "lat": (["lat"], g)
            })
            
            # Two-step interpolation: first to regular, then to target
            self.fix = xe.Regridder(
                xfield[xname], gaussian_regular,
                method="nearest_s2d", locstream_in=True, periodic=True
            )
            
            self.remap = xe.Regridder(
                self.fix(xfield[xname]), self.targetgrid,
                periodic=True, method="bilinear"
            )
            
        elif self.component in ['cmoratm', 'globo']:
            # Handle regular atmospheric grids
            self.fix = None
            self.remap = xe.Regridder(
                xfield, self.targetgrid, 
                periodic=True, method="bilinear"
            )
            
        else:
            raise KeyError(
                f"ESMF atmospheric weights not defined for component '{self.component}'"
            )
    
    def _create_oceanic_weights(self, xfield):
        """Create ESMF oceanic interpolation weights"""
        
        if self.component not in ["nemo", "cmoroce"]:
            raise KeyError(
                f"ESMF oceanic weights not defined for component '{self.component}'"
            )
        
        # Determine variable name
        if "areacello" in xfield.data_vars:  # CMOR case
            xname = "areacello"
        elif "cell_area" in xfield.data_vars:  # ECE4 NEMO case
            xname = "cell_area"
        else:
            xname = list(xfield.data_vars)[-1]  # fallback
        
        # Choose interpolation method based on grid type
        if self.gridtype == "unstructured":
            self.remap = xe.Regridder(
                xfield[xname], self.targetgrid,
                method="nearest_s2d", locstream_in=True, periodic=True
            )
        else:
            # Regular or curvilinear grids
            self.remap = xe.Regridder(
                xfield[xname], self.targetgrid,
                method="bilinear", periodic=True, ignore_degenerate=True
            )
        
        self.fix = None


class CDOInterpolator(BaseInterpolator):
    """CDO interpolator for both atmospheric and oceanic data"""
    
    def __init__(self, domain, component, gridtype, targetgrid):
        super().__init__(component, gridtype, targetgrid)
        self.domain = domain.lower()
        self.is_cdo = True  # Mark as CDO interpolator
    
    def create_weights(self, areafile):
        """Create CDO interpolation weights based on domain"""
        
        if self.domain in ['atm', 'atmospheric', 'atmosphere']:
            self._create_atmospheric_weights(areafile)
        elif self.domain in ['oce', 'oceanic', 'ocean']:
            self._create_oceanic_weights(areafile)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
        
        loggy.info('Created CDO %s interpolator for %s grid', self.domain, self.gridtype)
    
    def _create_atmospheric_weights(self, areafile, remap='bil'):
        """Create CDO atmospheric interpolation weights"""
        
        self.fix = None
        weights = cdo_generate_weights(
            glob(areafile)[0],
            self.targetgrid,
            method=remap
        )
        self.remap = Regridder(weights=weights).regrid
    
    def _create_oceanic_weights(self, areafile):
        """Create CDO oceanic interpolation weights"""
        
        # Choose CDO method based on grid type
        cdomethod = 'bil' if self.gridtype == 'bilinear' else 'nn'

        self.fix = None
        weights = cdo_generate_weights(
            glob(areafile)[0],
            self.targetgrid,
            method=cdomethod
        )
        self.remap = Regridder(weights=weights).regrid


class InterpolatorFactory:
    """Factory class for creating appropriate interpolators"""
    
    @staticmethod
    def create_interpolator(domain, component, gridtype, targetgrid, tool='ESMF'):
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
        
        if tool == 'ESMF':
            return ESMFInterpolator(domain, component, gridtype, targetgrid)
        if tool == 'CDO':
            return CDOInterpolator(domain, component, gridtype, targetgrid)
        raise ValueError(f"Unknown tool: {tool}. Use 'ESMF' or 'CDO'")
