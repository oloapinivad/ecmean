#!/usr/bin/env python3
'''
Shared diagnostic class for ECmean4
'''

import os
import logging
from pathlib import Path
import xarray as xr
from ecmean.libs.files import load_yaml
from ecmean import __version__ as version

####################
# DIAGNOSTIC CLASS #
####################

loggy = logging.getLogger(__name__)


class Diagnostic():
    """General container class for common variables"""

    def __init__(self, exp, year1, year2, config, funcname,
                 line=False, trend=False,
                 resolution="r360x180", ensemble='r1i1p1f1', addnan=False,
                 interface=None, modelname=None, outputdir=None,
                 xdataset=None, silent=None,
                 numproc=1, climatology=None, reference=None):
        """
        Initialize the Diagnostic instance.

        Args:
            args: Arguments from command line/function.
        """

        # arguments from command line/function
        self.expname = exp
        self.year1 = year1
        self.year2 = year2
        self.years_joined = list(range(self.year1, self.year2 + 1))
        self.ftable = line
        self.ftrend = trend
        self.numproc = numproc
        self.climatology = climatology
        self.silent = silent
        self.resolution = resolution
        self.reference = reference
        self.ensemble = ensemble
        self.interface = interface
        self.funcname = funcname
        self.version = version
        self.addnan = addnan
        if self.year1 == self.year2:
            self.ftrend = False
        
        print(f'Welcome to ECmean4 v{self.version}: Running {self.funcname} '
              f'with {self.numproc} cores on experiment {self.expname} '
              f'for {len(self.years_joined)} years ({self.year1}-{self.year2})')

        #  These are here in prevision of future expansion to CMOR
        self.grid = '*'
        self.version = '*'
        self.frequency = '*mon'

        # base init
        self.field_all = []
        self.var_all = []

        # current path
        self.indir = Path(os.path.dirname(os.path.abspath(__file__)))

        # get the config file, can be a string or a dictionary
        if config:
            if isinstance(config, dict):
                cfg = config
            elif isinstance(config, str):
                cfg = load_yaml(config)
            else:
                raise ValueError('Cannot load the config file')
        else:
            cfg = load_yaml(self.indir / '../../config.yml')

        # Various raise and input and output directories
        if not cfg['dirs']['exp']:
            raise ValueError('No experiment directory defined in config file')
        self.ecedir = Path(os.path.expandvars(cfg['dirs']['exp']))
        if outputdir is None:
            if not cfg['dirs']['tab']:
                raise ValueError('No table directory defined in config file')
            self.tabdir = Path(os.path.expandvars(cfg['dirs']['tab']))
            if not cfg['dirs']['fig']:
                raise ValueError('No figure directory defined in config file')
            self.figdir = Path(os.path.expandvars(cfg['dirs']['fig']))
        else:
            self.tabdir = Path(os.path.join(outputdir, 'yml'))
            self.figdir = Path(os.path.join(outputdir, 'pdf'))

        # init for global mean
        if self.funcname == 'GlobalMean':
            self.cfg_global_mean(cfg)

        # init for performance indices
        if self.funcname in 'PerformanceIndices':
            self.cfg_performance_indices(cfg)

        # setting up interface file
        self.interface = interface or cfg['interface']

        # setting up model name
        self.modelname = modelname or cfg['model']['name']

        # allow for both interface name or interface file
        if not os.path.exists(self.interface):
            self.interface = self.indir / Path(
                '../interfaces',
                f'interface_{self.interface}.yml')

        # load the possible xarray dataset
        if xdataset is not None:
            if isinstance(xdataset, (xr.DataArray, xr.Dataset)):
                loggy.warning('You asked to use your own xarray dataset/datarray...')
                self.xdataset = xdataset
            else:
                raise ValueError('Cannot used the xdataset, is not Xarray object')
        else:
            self.xdataset = None

    def filenames(self, kind):
        """
        Return the filename for the output.
        """

        if self.funcname == 'GlobalMean':
            head = 'global_mean'
        elif self.funcname == 'PerformanceIndices':
            head = f'PI4_{self.climatology}'
        else:
            raise ValueError('Unknown function name')

        figurename = f'{head}_{self.expname}_{self.modelname}_{self.ensemble}_{self.year1}_{self.year2}.{kind}'

        if kind in ['yml', 'txt']:
            return self.tabdir / figurename
        if kind in ['pdf', 'png']:
            return self.figdir / figurename
        raise ValueError('Unknown file type')

    def cfg_global_mean(self, cfg):
        """
        Set up configuration details for global mean.

        Args:
            cfg: Configuration details.

        Returns:
            None
        """

        self.regions = cfg['global_mean']['regions']
        self.seasons = cfg['global_mean']['seasons']

        self.var_atm = cfg['global_mean']['variables'].get('atm', [])
        self.var_oce = cfg['global_mean']['variables'].get('oce', [])
        self.var_ice = cfg['global_mean']['variables'].get('ice', [])
        self.var_table = cfg['global_mean']['variables'].get('tab', [])

        self.var_all = self.var_atm + self.var_oce + self.var_ice

        if not self.reference:
            self.reference = cfg['global_mean']['reference']

        self.reffile = self.indir / f'../reference/gm_reference_{self.reference}.yml'

        if self.ftable:
            self.linefile = self.tabdir / 'global_means.txt'

    def cfg_performance_indices(self, cfg):
        """
        Set up configuration for performance indices.

        Args:
            cfg: Configuration details.

        Returns:
            None
        """

        self.regions = cfg['performance_indices']['regions']
        self.seasons = cfg['performance_indices']['seasons']

        self.field_atm2d = cfg['performance_indices']['variables'].get('atm2d', [])
        self.field_atm3d = cfg['performance_indices']['variables'].get('atm3d', [])
        self.field_oce = cfg['performance_indices']['variables'].get('oce', [])
        self.field_ice = cfg['performance_indices']['variables'].get('ice', [])

        self.field_all = self.field_atm2d + self.field_atm3d + self.field_oce + self.field_ice

        if not self.resolution:
            self.resolution = cfg['performance_indices']['resolution']

        if not self.climatology:
            self.climatology = cfg['performance_indices']['climatology']

        self.clmdir = Path(self.indir, '../climatology', self.climatology)
        self.resclmdir = Path(self.clmdir, self.resolution)
        self.climfile = self.clmdir / f'pi_climatology_{self.climatology}.yml'

    def _remove_variables(self, var_list, vars_to_remove):
        """Helper function to remove variables from a list while preserving order."""
        return [var for var in var_list if var not in set(vars_to_remove)]

    def configure_amip_omip_cpld(self, support_dictionary):
        """
        Configure the experiment for AMIP, OMIP, or coupled simulations.
        Updates the variables/fields on which to run as a function of the 
        experiment type, which is deduced from the availabe grids

        Args:
            util_dictionary: Dictionary containing output from Supporter()            
        """
        
        # check if we can run the performance indices
        if self.funcname == 'PerformanceIndices':
            if not support_dictionary.oceareafile and not support_dictionary.ocemaskfile:
                loggy.warning('No oceanic file available, assuming this is an AMIP run without oceanic variables.')
                oceanic_vars = self.field_oce + self.field_ice
                self.field_all = self._remove_variables(self.field_all, oceanic_vars)
            if not support_dictionary.atmareafile and not support_dictionary.atmmaskfile:
                loggy.warning('No atmospheric file found, assuming this is an OMIP run without atmospheric variables.')
                atmospheric_vars = self.field_atm2d + self.field_atm3d
                self.field_all = self._remove_variables(self.field_all, atmospheric_vars)
            if self.field_all == []:
                raise ValueError('No variables to process due to missing area/mask files, check your configuration file!')
            
        if self.funcname == 'GlobalMean':
            if not support_dictionary.oceareafile and not support_dictionary.ocemaskfile:
                loggy.warning('No oceanic file available, assuming this is an AMIP run without oceanic variables.')
                oceanic_vars = self.var_oce + self.var_ice
                self.var_all = self._remove_variables(self.var_all, oceanic_vars)
            if not support_dictionary.atmareafile and not support_dictionary.atmmaskfile:
                loggy.warning('No atmospheric file found, assuming this is an OMIP run without atmospheric variables.')
                self.var_all = self._remove_variables(self.var_all, self.var_atm)
            if self.var_all == []:
                raise ValueError('No variables to process due to missing area/mask files, check your configuration file!')

    

