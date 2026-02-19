"""Tests for global mean fuctions"""

import os
import subprocess
import xarray as xr
from ecmean.global_mean import global_mean, GlobalMean
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.general import are_dicts_equal
from ecmean.libs.files import load_yaml

# set TOLERANCE
TOLERANCE = 1e-3

# set up coverage env var
env = {**os.environ, "COVERAGE_PROCESS_START": ".coveragerc"}


# Open the text file for reading
def load_gm_txt_files(textfile):
    """
    Function to the read the global mean text files and extract the model values
    in order to create a dictionary that can be used for comparison
    """

    data_dict = {}
    with open(textfile, 'r', encoding='utf8') as file:
        # Read the file line by line
        for line in file:
            # Remove leading and trailing whitespace and split the line by '|'
            columns = line.strip().split('|')
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
def test_cmd_global_mean_coupled():
    file1 = 'tests/table/global_mean_cpld_EC-Earth4_r1i1p1f1_1990_1990.txt'
    file2 = 'tests/table/global_mean_cpld_1990_1990.ref'
    if os.path.isfile(file1):
        os.remove(file1)
    subprocess.run(['global_mean', 'cpld', '1990', '1990', '-j', '2',
                    '-c', 'tests/config.yml', '--trend', '-l', 'debug'],
                   env=env, check=True)

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."


# call on amip ECE
def test_global_mean_amip():
    file1 = 'tests/table/global_mean_amip_EC-Earth4_r1i1p1f1_1990_1990.txt'
    file2 = 'tests/table/global_mean_amip_1990_1990.ref'
    if os.path.isfile(file1):
        os.remove(file1)
    global_mean(exp='amip', year1=1990, year2=1990, numproc=1, config='tests/config.yml',
                line=True, addnan=True, reference='EC23')

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."

# call on amip ECE
def test_global_mean_omip():
    file1 = 'tests/table/global_mean_omip_EC-Earth4_r1i1p1f1_1990_1990.txt'
    file2 = 'tests/table/global_mean_omip_1990_1990.ref'
    if os.path.isfile(file1):
        os.remove(file1)
    global_mean(exp='omip', year1=1990, year2=1990, numproc=1, config='tests/config.yml',
                reference='EC23')

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."


# call on amip ECE using the xdataset option
def test_global_mean_amip_xdataset_config_dict():
    file1 = 'tests/table/global_mean_amip_EC-Earth4_r1i1p1f1_1990_1990.txt'
    file2 = 'tests/table/global_mean_amip_1990_1990.ref'
    if os.path.isfile(file1):
        os.remove(file1)
    xfield = xr.open_mfdataset('tests/data/amip/output/oifs/*.nc', preprocess=xr_preproc, compat='no_conflicts')
    config = load_yaml('tests/config.yml')
    global_mean(exp='amip', year1=1990, year2=1990, numproc=4, config=config,
                xdataset=xfield, reference='EC23')
    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."

# call on historical CMIP6

def test_global_mean_CMIP6():
    file1 = 'tests/table/global_mean_historical_EC-Earth3_r1i1p1f1_1990_1991.txt'
    file2 = 'tests/table/global_mean_CMIP6_1990_1991.ref'
    if os.path.isfile(file1):
        os.remove(file1)
    global_mean(exp='historical', year1=1990, year2=1991, numproc=2,
                config='tests/config_CMIP6.yml', trend=True, reference='EC23')

    data1 = load_gm_txt_files(file1)
    data2 = load_gm_txt_files(file2)

    assert are_dicts_equal(data1, data2, TOLERANCE), "TXT files are not identical."


def test_gm_plot(tmp_path):
    outputfile = tmp_path / 'Global_Mean_Heatmap.pdf'
    gm = GlobalMean('amip', 1990, 1990, config='tests/config.yml', loglevel='info')
    gm.prepare()
    gm.plot(mapfile=outputfile)
    assert os.path.isfile(outputfile), "Plot not created."
