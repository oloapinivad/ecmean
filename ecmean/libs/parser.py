"""
Shared functions for ecmean: parsers arguments from command line
"""

import argparse

from ecmean import __version__
from ecmean.libs.climatology import DEFAULT_CLIMATOLOGY, DEFAULT_REFERENCE, SUPPORTED_CLIMATOLOGY, SUPPORTED_REFERENCE


def parse_arguments(args, script):
    """Parse CLI arguments for global mean

    Args:
        args: command line argument
        script: script for which parsing is used, i.e. 'gm' or 'pi'
    Return:
        Parsed arguments
    """

    # common configuration to be parsed
    parser = argparse.ArgumentParser(description="ECmean global mean diagnostics for Global Climate models")
    parser.add_argument("exp", metavar="EXP", type=str, help="experiment ID")
    parser.add_argument("year1", metavar="Y1", type=int, help="starting year")
    parser.add_argument("year2", metavar="Y2", type=int, help="final year")
    parser.add_argument("-i", "--interface", type=str, default="", help="interface (overrides config.yml)")
    parser.add_argument("-c", "--config", type=str, default="", help="config file (default: ./config.yml)")
    parser.add_argument("-j", dest="numproc", type=int, default=1, help="number of processors to use")
    parser.add_argument("-m", "--model", type=str, default="", help="model name")
    parser.add_argument("-e", "--ensemble", type=str, default=None, help="variant label (ripf number for cmor)")
    parser.add_argument("--consortium", type=str, default=None, help="consortium name (cmor institution_id)")
    parser.add_argument("--mip", type=str, default=None, help="MIP name (cmor activity_id, e.g. CMIP, HighResMIP)")
    parser.add_argument("-l", "--loglevel", type=str, default="WARNING", help="define the level of logging.")
    parser.add_argument("-o", "--outputdir", type=str, help="force the output directory")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    # specific to global mean
    if script == "gm":
        parser.add_argument("--trend", action="store_true", help="compute trends")
        parser.add_argument("--line", action="store_true", help="appends also single line to a table")
        parser.add_argument(
            "--reference",
            type=str,
            default=DEFAULT_REFERENCE,
            help=f"reference climatology to be compared. default: {DEFAULT_REFERENCE}. Options: {SUPPORTED_REFERENCE}",
            choices=SUPPORTED_REFERENCE,
        )
        parser.add_argument("--addnan", action="store_true", help="provide figures also where observations are missing")

    # specific to performance indices
    if script == "pi":
        parser.add_argument(
            "--climatology",
            type=str,
            default=DEFAULT_CLIMATOLOGY,
            help=f"climatology to be compared. default: {DEFAULT_CLIMATOLOGY}. Options: {SUPPORTED_CLIMATOLOGY}",
            choices=SUPPORTED_CLIMATOLOGY,
        )

    return parser.parse_args(args)
