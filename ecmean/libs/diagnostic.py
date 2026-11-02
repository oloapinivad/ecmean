"""
Shared diagnostic class for ecmean
"""

import logging
import os
import warnings
from pathlib import Path

import xarray as xr

from ecmean import __version__ as version
from ecmean.libs.climatology import (
    DEFAULT_CLIMATOLOGY,
    DEFAULT_REFERENCE,
    DEFAULT_RESOLUTION,
    DEPRECATED_CLIMATOLOGY,
    DEPRECATED_REFERENCE,
    SUPPORTED_RESOLUTION,
)
from ecmean.libs.files import load_yaml
from ecmean.libs.masks import LANDSEA_MASK_TYPES

####################
# DIAGNOSTIC CLASS #
####################

loggy = logging.getLogger(__name__)


class Diagnostic:
    """General container class for common variables"""

    def __init__(
        self,
        exp,
        year1,
        year2,
        config,
        funcname,
        line=False,
        trend=False,
        resolution=None,
        ensemble=None,
        addnan=False,
        interface=None,
        modelname=None,
        outputdir=None,
        xdataset=None,
        numproc=1,
        climatology=None,
        reference=None,
        consortium=None,
        mip=None,
    ):
        """
        Initialize the Diagnostic instance.

        Args:
            exp (str): Experiment name.
            year1 (int): Start year.
            year2 (int): End year.
            config (str or dict): Path to the configuration file, or configuration dictionary.
            funcname (str): Diagnostic name, 'GlobalMean' or 'PerformanceIndices'.
            line (bool): Write the single-line tuning table (global mean only).
            trend (bool): Compute trends (global mean only).
            resolution (str): Target grid for the interpolation. Default is 'r360x180'.
            ensemble (str): Ensemble identifier, overriding the configuration. Default is 'r1i1p1f1'.
            addnan (bool): Plot also variables without reference (global mean only).
            interface (str): Interface name or file, overriding the configuration.
            modelname (str): Model name, overriding the configuration.
            outputdir (str): Output directory, overriding the folders of the configuration.
            xdataset (xarray.Dataset or xarray.DataArray): Data to use instead of reading the files.
            numproc (int): Number of processes.
            climatology (str): Performance indices climatology, overriding the configuration.
            reference (str): Global mean reference, overriding the configuration.
            consortium (str): Consortium name (CMOR institution_id), overriding the configuration. Default is '*'.
            mip (str): MIP name (CMOR activity_id), overriding the configuration. Default is 'CMIP'.
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
        self.resolution = resolution  # for performance indices only: set in cfg_performance_indices
        self.reference = reference
        self.interface = interface
        self.funcname = funcname
        self.ecmean_version = version
        self.addnan = addnan
        if self.year1 == self.year2:
            self.ftrend = False
        print(
            f"Welcome to ecmean v{self.ecmean_version}: Running {self.funcname} "
            f"with {self.numproc} cores on experiment {self.expname} "
            f"for {len(self.years_joined)} years ({self.year1}-{self.year2})"
        )

        #  These are here in prevision of future expansion to CMOR: wildcards for the filename template
        self.grid = "*"
        self.version = "*"
        self.frequency = "*mon"

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
                raise ValueError("Cannot load the config file")
        else:
            default_config = Path.cwd() / "config.yml"
            if not default_config.is_file():
                raise FileNotFoundError(
                    f"No config file provided and no config.yml found in the current directory ({Path.cwd()}): "
                    "use -c/--config or copy config.tmpl to config.yml"
                )
            cfg = load_yaml(default_config)

        # Various raise and input and output directories
        if not cfg["dirs"]["exp"]:
            raise ValueError("No experiment directory defined in config file")
        self.ecedir = Path(os.path.expandvars(cfg["dirs"]["exp"]))
        if outputdir is None:
            if not cfg["dirs"]["tab"]:
                raise ValueError("No table directory defined in config file")
            self.tabdir = Path(os.path.expandvars(cfg["dirs"]["tab"]))
            if not cfg["dirs"]["fig"]:
                raise ValueError("No figure directory defined in config file")
            self.figdir = Path(os.path.expandvars(cfg["dirs"]["fig"]))
        else:
            self.tabdir = Path(os.path.join(outputdir, "YAML"))
            self.figdir = Path(os.path.join(outputdir, "PDF"))

        # init for global mean
        if self.funcname == "GlobalMean":
            self.cfg_global_mean(cfg)

        # init for performance indices
        if self.funcname == "PerformanceIndices":
            self.cfg_performance_indices(cfg)

        # setting up interface file
        self.interface = interface or cfg["interface"]

        # setting up model name and CMOR identifiers: arguments override the model section of the config
        self.modelname = modelname or cfg["model"]["name"]
        self.ensemble = ensemble or cfg["model"].get("ensemble") or "r1i1p1f1"
        self.consortium = consortium or cfg["model"].get("consortium") or "*"
        self.mip = mip or cfg["model"].get("mip") or "CMIP"

        # allow for both interface name or interface file
        if not os.path.exists(self.interface):
            self.interface = self.indir / Path("../interfaces", f"interface_{self.interface}.yml")

        # load the possible xarray dataset
        if xdataset is not None:
            if isinstance(xdataset, (xr.DataArray, xr.Dataset)):
                loggy.warning("You asked to use your own xarray dataset/datarray...")
                self.xdataset = xdataset
            else:
                raise ValueError("Cannot used the xdataset, is not Xarray object")
        else:
            self.xdataset = None

    def filenames(self, kind):
        """
        Return the filename for the output.
        """

        if self.funcname == "GlobalMean":
            head = f"GM_{self.reference}"
        elif self.funcname == "PerformanceIndices":
            head = f"PI_{self.climatology}"
        else:
            raise ValueError("Unknown function name")

        figurename = f"{head}_{self.expname}_{self.modelname}_{self.ensemble}_{self.year1}_{self.year2}.{kind}"

        if kind in ["yml", "txt"]:
            return self.tabdir / figurename
        if kind in ["pdf", "png"]:
            return self.figdir / figurename
        raise ValueError("Unknown file type")

    def check_global_all(self):
        """
        Check that the regions include "Global" and the seasons include "ALL".
        Global mean tables and plots rely on the ["ALL"]["Global"] entry, performance indices do not.

        Raises:
            ValueError: if "Global" is not in the regions or "ALL" is not in the seasons.
        """

        if "Global" not in self.regions:
            raise ValueError(f"The 'Global' region is mandatory in the config file, found regions: {self.regions}")
        if "ALL" not in self.seasons:
            raise ValueError(f"The 'ALL' season is mandatory in the config file, found seasons: {self.seasons}")

    def cfg_global_mean(self, cfg):
        """
        Set up configuration details for global mean.

        Args:
            cfg: Configuration details.

        Returns:
            None
        """

        self.regions = cfg["global_mean"]["regions"]
        self.seasons = cfg["global_mean"]["seasons"]
        self.check_global_all()

        self.var_atm = cfg["global_mean"]["variables"].get("atm", [])
        self.var_oce = cfg["global_mean"]["variables"].get("oce", [])
        self.var_ice = cfg["global_mean"]["variables"].get("ice", [])
        self.var_table = cfg["global_mean"]["variables"].get("tab", [])

        self.var_all = self.var_atm + self.var_oce + self.var_ice

        if not self.reference:
            self.reference = cfg["global_mean"].get("reference", DEFAULT_REFERENCE)

        if self.reference in DEPRECATED_REFERENCE:
            msg = (
                f"Reference {self.reference} is deprecated and will be removed in a future release. "
                "Please use EC26-CMIP, EC26-HIST or EC26-PDAY."
            )
            loggy.warning(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        self.reffile = self.indir / f"../reference/gm_reference_{self.reference}.yml"

        if self.ftable:
            self.linefile = self.tabdir / "global_means.txt"

    def cfg_performance_indices(self, cfg):
        """
        Set up configuration for performance indices.

        Args:
            cfg: Configuration details.

        Returns:
            None
        """

        self.regions = cfg["performance_indices"]["regions"]
        self.seasons = cfg["performance_indices"]["seasons"]

        self.field_atm2d = cfg["performance_indices"]["variables"].get("atm2d", [])
        self.field_atm3d = cfg["performance_indices"]["variables"].get("atm3d", [])
        self.field_oce = cfg["performance_indices"]["variables"].get("oce", [])
        self.field_ice = cfg["performance_indices"]["variables"].get("ice", [])

        self.field_all = self.field_atm2d + self.field_atm3d + self.field_oce + self.field_ice

        if not self.climatology:
            self.climatology = cfg["performance_indices"].get("climatology", DEFAULT_CLIMATOLOGY)

        if self.climatology in DEPRECATED_CLIMATOLOGY:
            msg = (
                f"Climatology {self.climatology} is deprecated and will be removed in a future release: "
                "its CMIP6 reference PIs for 3D fields (ta, ua, va, hus) are biased (see issue #150), "
                "so relative 3D PIs are unreliable. Please use EC26-CMIP or EC26-HIST."
            )
            loggy.warning(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        # the argument has priority over the config, which has priority over the default
        if not self.resolution:
            self.resolution = cfg["performance_indices"].get("resolution", DEFAULT_RESOLUTION)
        if self.resolution not in SUPPORTED_RESOLUTION:
            raise ValueError(f"Resolution {self.resolution} is not supported, available resolutions: {SUPPORTED_RESOLUTION}")

        self.clmdir = Path(self.indir, "../climatology", self.climatology)
        self.resclmdir = Path(self.clmdir, self.resolution)
        self.climfile = self.clmdir / f"pi_climatology_{self.climatology}.yml"

    def _remove_variables(self, var_list, vars_to_remove):
        """Helper function to remove variables from a list while preserving order."""
        return [var for var in var_list if var not in set(vars_to_remove)]

    def _remove_masked_variables(self, var_list, atm_vars, reference):
        """
        Remove from a list the atmospheric variables that need the land-sea mask, when it is not available.

        Args:
            var_list (list): Variables to be filtered.
            atm_vars (list): Variables of the atmospheric domain.
            reference (dict): Reference or climatology dictionary, providing the mask type of each variable.

        Returns:
            list: The variables that can be processed without the land-sea mask.
        """
        masked_vars = [var for var in var_list if var in atm_vars and reference[var].get("mask") in LANDSEA_MASK_TYPES]
        for var in masked_vars:
            loggy.warning(
                "Variable %s requires a land-sea mask (mask: %s), but no maskfile is available for the atmospheric "
                "component: %s will be skipped. Set the atmospheric 'maskfile' in the interface file to compute it.",
                var,
                reference[var]["mask"],
                var,
            )
        return self._remove_variables(var_list, masked_vars)

    def configure_amip_omip_cpld(self, support_dictionary, reference=None):
        """
        Configure the experiment for AMIP, OMIP, or coupled simulations.
        Updates the variables/fields on which to run as a function of the
        experiment type, which is deduced from the availabe grids. If the atmospheric
        grid is available but its land-sea mask is not, the atmospheric variables
        requiring land or sea masking are removed as well.

        Args:
            support_dictionary: Supporter() object, with the available files, areas and masks
            reference (dict, optional): Reference (global mean) or climatology (performance indices)
                dictionary, used to find which variables need the land-sea mask. If None, no variable
                is removed for a missing mask.
        """

        # the atmospheric grid is there, but the land-sea mask is missing
        missing_atm_mask = reference is not None and support_dictionary.atmareafile and support_dictionary.atmmask is None

        # check if we can run the performance indices
        if self.funcname == "PerformanceIndices":
            if not support_dictionary.oceareafile and not support_dictionary.ocemaskfile:
                loggy.warning("No oceanic file available, assuming this is an AMIP run without oceanic variables.")
                oceanic_vars = self.field_oce + self.field_ice
                self.field_all = self._remove_variables(self.field_all, oceanic_vars)
            if not support_dictionary.atmareafile and not support_dictionary.atmmaskfile:
                loggy.warning("No atmospheric file found, assuming this is an OMIP run without atmospheric variables.")
                atmospheric_vars = self.field_atm2d + self.field_atm3d
                self.field_all = self._remove_variables(self.field_all, atmospheric_vars)
            if missing_atm_mask:
                self.field_all = self._remove_masked_variables(self.field_all, self.field_atm2d, reference)
            if self.field_all == []:
                raise ValueError("No variables to process due to missing area/mask files, check your configuration file!")

        if self.funcname == "GlobalMean":
            if not support_dictionary.oceareafile and not support_dictionary.ocemaskfile:
                loggy.warning("No oceanic file available, assuming this is an AMIP run without oceanic variables.")
                oceanic_vars = self.var_oce + self.var_ice
                self.var_all = self._remove_variables(self.var_all, oceanic_vars)
            if not support_dictionary.atmareafile and not support_dictionary.atmmaskfile:
                loggy.warning("No atmospheric file found, assuming this is an OMIP run without atmospheric variables.")
                self.var_all = self._remove_variables(self.var_all, self.var_atm)
            if missing_atm_mask:
                self.var_all = self._remove_masked_variables(self.var_all, self.var_atm, reference)
            if self.var_all == []:
                raise ValueError("No variables to process due to missing area/mask files, check your configuration file!")
