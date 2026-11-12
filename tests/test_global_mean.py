"""Tests for global mean fuctions"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

from ecmean.global_mean import GlobalMean, global_mean
from ecmean.libs.files import load_yaml
from ecmean.libs.general import are_dicts_equal
from ecmean.libs.ncfixers import xr_preproc

# set TOLERANCE
TOLERANCE = 1e-3
# set CLEANUP flag to control removal of generated TXT files after tests
CLEANUP = True

# set up coverage env var
env = {**os.environ, "COVERAGE_PROCESS_START": ".coveragerc"}


# Open the text file for reading
def load_gm_txt_files(textfile):
    """
    Function to the read the global mean text files and extract the model values
    in order to create a dictionary that can be used for comparison
    """

    data_dict = {}
    with open(textfile, "r", encoding="utf8") as file:
        # Read the file line by line
        for line in file:
            # Remove leading and trailing whitespace and split the line by '|'
            columns = line.strip().split("|")
            # Check if there are at least 5 columns (including the header row)
            if len(columns) >= 5:
                # Extract the first and fourth columns and remove leading/trailing whitespace
                variable = columns[1].strip()
                value = columns[4].strip()
                # Add the data to the dictionary if it's not empty
                if variable and value:
                    data_dict[variable] = value
    return data_dict


# call on coupled ECE using parser and debug mode
@pytest.mark.parametrize("clim", ["EC26-HIST"])
def test_cmd_global_mean_coupled(clim):
    """Test global_mean command line interface on coupled EC-Earth4 data."""
    file1 = "tests/table/GM_" + clim + "_cpld_EC-Earth4_r1i1p1f1_1990_1990.txt"
    file2 = "tests/table/GM_" + clim + "_cpld_1990_1990.ref"
    if os.path.isfile(file1):
        os.remove(file1)
    subprocess.run(
        [
            str(Path(sys.executable).parent / "global_mean"),
            "cpld",
            "1990",
            "1990",
            "-j",
            "2",
            "-c",
            "tests/config.yml",
            "--trend",
            "-l",
            "debug",
            "--reference",
            clim,
        ],
        env=env,
        check=True,
    )

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# call on amip ECE
@pytest.mark.parametrize("clim", ["EC26-CMIP"])
def test_global_mean_amip(clim):
    """Test global_mean on AMIP EC-Earth4 data with line plot and NaN handling."""
    file1 = "tests/table/GM_" + clim + "_amip_EC-Earth4_r1i1p1f1_1990_1990.txt"
    file2 = "tests/table/GM_" + clim + "_amip_1990_1990.ref"
    if os.path.isfile(file1):
        os.remove(file1)
    global_mean(
        exp="amip", year1=1990, year2=1990, numproc=1, config="tests/config.yml", line=True, addnan=True, reference=clim
    )

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# call on omip ECE
@pytest.mark.parametrize("clim", ["EC26-HIST"])
def test_global_mean_omip(clim):
    """Test global_mean on OMIP EC-Earth4 data."""
    file1 = "tests/table/GM_" + clim + "_omip_EC-Earth4_r1i1p1f1_1990_1990.txt"
    file2 = "tests/table/GM_" + clim + "_omip_1990_1990.ref"
    if os.path.isfile(file1):
        os.remove(file1)
    global_mean(exp="omip", year1=1990, year2=1990, numproc=1, config="tests/config.yml", reference=clim)

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# call on amip ECE using the xdataset option
@pytest.mark.parametrize("clim", ["EC26-PDAY"])
def test_global_mean_amip_xdataset_config_dict(clim):
    """Test global_mean using xarray dataset input and dictionary config."""
    file1 = "tests/table/GM_" + clim + "_amip_EC-Earth4_r1i1p1f1_1990_1990.txt"
    file2 = "tests/table/GM_" + clim + "_amip_1990_1990.ref"
    if os.path.isfile(file1):
        os.remove(file1)
    xfield = xr.open_mfdataset("tests/data/amip/output/oifs/*.nc", preprocess=xr_preproc, compat="no_conflicts")
    config = load_yaml("tests/config.yml")
    global_mean(exp="amip", year1=1990, year2=1990, numproc=4, config=config, xdataset=xfield, reference=clim)
    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


# call on historical CMIP6
@pytest.mark.parametrize("clim", ["EC26-CMIP"])
def test_global_mean_CMIP6(clim):
    """Test global_mean on CMIP6 historical data with trend calculation."""
    file1 = "tests/table/GM_" + clim + "_historical_EC-Earth3_r1i1p1f1_1990_1991.txt"
    file2 = "tests/table/GM_" + clim + "_CMIP6_1990_1991.ref"
    if os.path.isfile(file1):
        os.remove(file1)
    global_mean(
        exp="historical", year1=1990, year2=1991, numproc=2, config="tests/config_CMIP6.yml", trend=True, reference=clim
    )

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."
    if CLEANUP and os.path.isfile(file1):
        os.remove(file1)


def test_deprecated_reference():
    """Deprecated references still work but raise a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        GlobalMean("amip", 1990, 1990, config="tests/config.yml", reference="EC23")


def test_gm_plot(tmp_path):
    """Test plot generation from GlobalMean class, using the default reference."""
    outputfile = tmp_path / "GM_Heatmap.pdf"
    gm = GlobalMean("amip", 1990, 1990, config="tests/config.yml", loglevel="info")
    gm.prepare()
    gm.plot(mapfile=outputfile)
    assert os.path.isfile(outputfile), "Plot not created."


def test_global_mean_omip_line():
    """The tuning line of an OMIP run has NaN for the removed variables and as many columns as the header (#153)"""
    gm = GlobalMean("omip", 1990, 1990, config="tests/config.yml", numproc=1, reference="EC26-HIST", line=True)
    if os.path.isfile(gm.diag.linefile):
        os.remove(gm.diag.linefile)
    gm.prepare()
    gm.run()
    gm.store()
    gm.plot(storefig=False)

    with open(gm.diag.linefile, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header = lines[0].split()[5:]
    values = lines[-1].split()[5:]
    assert len(values) == len(header)
    assert "nan" in values
    if CLEANUP:
        os.remove(gm.diag.linefile)
