"""
Common functions for reference and climatology creation.
"""

import logging

loggy = logging.getLogger(__name__)


# Climatology file prefixes
CLIMATOLOGY_PREFIXES = ["climate_variance_", "climate_average_", "seasons_variance_", "seasons_average_"]

# supported climatology
SUPPORTED_REFERENCE = ["EC23", "EC26-PDAY", "EC26-CMIP", "EC26-HIST"]
SUPPORTED_CLIMATOLOGY = ["EC23", "EC24", "EC26-HIST", "EC26-CMIP", "EC26-CMIP-HR"]

# deprecated climatology: their CMIP6 reference PIs for 3D fields are biased (#150)
DEPRECATED_CLIMATOLOGY = ["EC23", "EC24"]
DEPRECATED_REFERENCE = ["EC23"]

# horizontal resolutions on which the climatologies are available
SUPPORTED_RESOLUTION = ["r360x180"]

# defaults, used when not specified by the user or in the configuration file
DEFAULT_CLIMATOLOGY = "EC26-CMIP"
DEFAULT_REFERENCE = "EC26-CMIP"
DEFAULT_RESOLUTION = "r360x180"
