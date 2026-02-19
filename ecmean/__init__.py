"""ecmean module"""

__version__ = '0.2'

# functions to be accessible everywhere
from ecmean.libs.diagnostic import Diagnostic
from ecmean.libs.support import Supporter
from ecmean.libs.units import UnitsHandler
from ecmean.global_mean import GlobalMean, global_mean
from ecmean.performance_indices import PerformanceIndices, performance_indices

__all__ = ["GlobalMean", "global_mean", "PerformanceIndices",
           "performance_indices", "Diagnostic", "Supporter", "UnitsHandler"]
