"""Tests for performance indices functions"""

# test for PIs: run on ECE4 test data, both amip and coupled, and on CMIP6 EC-Earth3 data.
# mostly run on EC26 climatologies, with a couple of checks on the deprecated EC24
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr
import yaml

from ecmean import PerformanceIndices, performance_indices
from ecmean.libs.climatology import DEPRECATED_CLIMATOLOGY
from ecmean.libs.general import are_dicts_equal
from ecmean.libs.interpolators import cdo_available
from ecmean.libs.ncfixers import xr_preproc

# from ecmean.libs.plotting import heatmap_comparison_pi, prepare_clim_dictionaries_pi

# set TOLERANCE
TOLERANCE = 1e-1
# set CLEANUP flag to control removal of generated YML files after tests
CLEANUP = True
OUTPUTDIR = "tests/pluto"
# interpolation tools to test, CDO only when a recent enough executable is available
TOOLS = ["ESMF", pytest.param("CDO", marks=pytest.mark.skipif(not cdo_available(), reason="CDO >= 2.6.2 not available"))]

# set up coverage env var
env = {**os.environ, "COVERAGE_PROCESS_START": ".coveragerc"}


def deprecation_check(clim):
    """Expect a DeprecationWarning for deprecated climatologies"""
    if clim in DEPRECATED_CLIMATOLOGY:
        return pytest.warns(DeprecationWarning, match="deprecated")
    return contextlib.nullcontext()


# test on coupled
@pytest.mark.parametrize("clim", ["EC24", "EC26-HIST", "EC26-CMIP"])
def test_performance_indices_cpld(clim):
    """Test performance indices on coupled EC-Earth4 data."""
    with deprecation_check(clim):
        performance_indices("cpld", 1990, 1990, numproc=4, climatology=clim, config="tests/config.yml")
    file1 = "tests/table/PI_" + clim + "_cpld_EC-Earth4_r1i1p1f1_1990_1990.yml"
    file2 = "tests/table/PI_" + clim + "_cpld_1990_1990.ref"
    with open(file1, "r", encoding="utf8") as f1, open(file2, "r", encoding="utf8") as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# test on amip
@pytest.mark.parametrize("clim", ["EC26-CMIP"])
@pytest.mark.parametrize("tool", TOOLS)
def test_performance_indices_amip(clim, tool):
    """Test performance indices on AMIP EC-Earth4 data with custom output directory, with both interpolation tools."""
    performance_indices(
        "amip", 1990, 1990, numproc=1, climatology=clim, config="tests/config.yml", outputdir=OUTPUTDIR, tool=tool
    )
    file1 = "tests/pluto/YAML/PI_" + clim + "_amip_EC-Earth4_r1i1p1f1_1990_1990.yml"
    file2 = "tests/table/PI_" + clim + "_amip_1990_1990.ref"
    with open(file1, "r", encoding="utf8") as f1, open(file2, "r", encoding="utf8") as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)
        shutil.rmtree(OUTPUTDIR)


@pytest.mark.parametrize("clim", ["EC26-HIST"])
@pytest.mark.parametrize("tool", TOOLS)
def test_performance_indices_omip(clim, tool):
    """Test performance indices on OMIP EC-Earth4 data, with both interpolation tools."""
    performance_indices(
        "omip", 1990, 1990, numproc=1, climatology=clim, config="tests/config.yml", outputdir=OUTPUTDIR, tool=tool
    )
    file1 = "tests/pluto/YAML/PI_" + clim + "_omip_EC-Earth4_r1i1p1f1_1990_1990.yml"
    file2 = "tests/table/PI_" + clim + "_omip_1990_1990.ref"
    with open(file1, "r", encoding="utf8") as f1, open(file2, "r", encoding="utf8") as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)
        shutil.rmtree(OUTPUTDIR)


# test performance_indices from commnand line + debug
@pytest.mark.parametrize("clim", ["EC24", "EC26-CMIP", "EC26-HIST"])
def test_cmd_performance_indices_CMIP6(clim):
    """Test performance indices command line interface on CMIP6 historical data."""
    subprocess.run(
        [
            str(Path(sys.executable).parent / "performance_indices"),
            "historical",
            "1990",
            "1990",
            "-j",
            "2",
            "-c",
            "tests/config_CMIP6.yml",
            "--climatology",
            clim,
            "-m",
            "EC-Earth3",
            "-l",
            "debug",
        ],
        env=env,
        check=True,
    )
    file1 = "tests/table/PI_" + clim + "_historical_EC-Earth3_r1i1p1f1_1990_1990.yml"
    file2 = "tests/table/PI_" + clim + "_CMIP6_1990_1990.ref"
    with open(file1, "r", encoding="utf8") as f1, open(file2, "r", encoding="utf8") as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# test performance_indices from commnand line + debug
@pytest.mark.parametrize("clim", ["EC26-CMIP"])
def test_cmd_performance_indices_CMIP6_special(clim):
    """Test PerformanceIndices class interface with extrafigure option on CMIP6 data."""
    with open("tests/config_CMIP6_short.yml", "r", encoding="utf8") as configfile:
        configdict = yaml.safe_load(configfile)
    pi = PerformanceIndices(
        "historical", 1990, 1990, config=configdict, climatology=clim, loglevel="debug", ensemble="r1i1p1f1", extrafigure=True
    )
    pi.prepare()
    pi.run()
    pi.store()
    pi.plot()
    file1 = "tests/table/PI_" + clim + "_historical_EC-Earth3_r1i1p1f1_1990_1990.yml"
    file2 = "tests/table/PI_" + clim + "_CMIP6_1990_1990_short.ref"
    with open(file1, "r", encoding="utf8") as f1, open(file2, "r", encoding="utf8") as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# test on amip but with access from xarray dataset
@pytest.mark.parametrize("clim", ["EC26-HIST"])
def test_performance_indices_amip_xdataset(clim, tmp_path):
    """Test performance indices using xarray dataset input and plot generation."""
    file1 = "tests/table/PI_" + clim + "_amip_EC-Earth4_r1i1p1f1_1990_1990.yml"
    file2 = "tests/table/PI_" + clim + "_amip_1990_1990.ref"
    if os.path.isfile(file1):
        os.remove(file1)
    xfield = xr.open_mfdataset("tests/data/amip/output/oifs/*.nc", preprocess=xr_preproc, compat="no_conflicts")
    performance_indices("amip", 1990, 1990, numproc=2, climatology=clim, config="tests/config.yml", xdataset=xfield)
    with open(file1, "r", encoding="utf8") as f1, open(file2, "r", encoding="utf8") as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"

    # second part of the test: check that the plot can be created from the YML file
    outputfile = tmp_path / "PI_heatmap.png"
    pi = PerformanceIndices("amip", 1990, 1990, config="tests/config.yml", climatology=clim)
    pi.prepare()
    pi.plot(mapfile=outputfile)
    assert os.path.isfile(outputfile), "Plot not created."

    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)
