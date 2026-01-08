"""Tests for performance indices functions"""

# test for PIs: run on ECE4 test data, both amip and coupled, and on CMIP6 EC-Earth3 data.
# all run for EC23 climatologies
import os
import subprocess
import pytest
import xarray as xr
import yaml
from ecmean import performance_indices, PerformanceIndices
from ecmean.libs.ncfixers import xr_preproc
from ecmean.libs.general import are_dicts_equal
from ecmean.libs.ecplotter import ECPlotter
# from ecmean.libs.plotting import heatmap_comparison_pi, prepare_clim_dictionaries_pi

# set TOLERANCE
TOLERANCE = 1e-1
CLEANUP = False

# set up coverage env var
env = {**os.environ, "COVERAGE_PROCESS_START": ".coveragerc"}

# test on coupled
@pytest.mark.parametrize("clim", ['EC23', 'EC24'])
def test_performance_indices_cpld(clim):
    performance_indices('cpld', 1990, 1990, numproc=4,
                        climatology=clim, config='tests/config.yml')
    file1 = 'tests/table/PI4_' + clim + '_cpld_EC-Earth4_r1i1p1f1_1990_1990.yml'
    file2 = 'tests/table/PI4_' + clim + '_cpld_1990_1990.ref'
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"


# test on amip
@pytest.mark.parametrize("clim", ['EC23', 'EC24'])
def test_performance_indices_amip(clim):
    performance_indices('amip', 1990, 1990, numproc=1,
                        climatology=clim, config='tests/config.yml', outputdir='tests/pluto')
    file1 = 'tests/pluto/yml/PI4_' + clim + '_amip_EC-Earth4_r1i1p1f1_1990_1990.yml'
    file2 = 'tests/table/PI4_' + clim + '_amip_1990_1990.ref'
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"

def test_performance_indices_omip():
    performance_indices('omip', 1990, 1990, numproc=1,
                        climatology='EC23', config='tests/config.yml', outputdir='tests/pluto')
    file1 = 'tests/pluto/yml/PI4_' + 'EC23' + '_omip_EC-Earth4_r1i1p1f1_1990_1990.yml'
    file2 = 'tests/table/PI4_' + 'EC23' + '_omip_1990_1990.ref'
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"


# test performance_indices from commnand line + debug
@pytest.mark.parametrize("clim", ['EC23', 'EC24'])
def test_cmd_performance_indices_CMIP6(clim):
    subprocess.run(['performance_indices', 'historical', '1990', '1990', '-j', '2', '-c',
                    'tests/config_CMIP6.yml', '--climatology', clim, '-m', 'EC-Earth3', '-l', 'debug'],
                   env=env, check=True)
    file1 = 'tests/table/PI4_' + clim + '_historical_EC-Earth3_r1i1p1f1_1990_1990.yml'
    file2 = 'tests/table/PI4_' + clim + '_CMIP6_1990_1990.ref'
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"

#test performance_indices from commnand line + debug
@pytest.mark.parametrize("clim", ['EC24'])
def test_cmd_performance_indices_CMIP6_special(clim):
    with open('tests/config_EC24.yml', 'r', encoding='utf8') as configfile:
        configdict = yaml.safe_load(configfile)
    pi = PerformanceIndices('historical', 1990, 1990, config=configdict, climatology=clim, loglevel='debug',
                            ensemble='r1i1p1f1', extrafigure=True)
    pi.prepare()
    pi.run()
    pi.store()
    pi.plot()
    file1 = 'tests/table/PI4_' + clim + '_historical_EC-Earth3_r1i1p1f1_1990_1990.yml'
    file2 = 'tests/table/PI4_' + clim + '_CMIP6_1990_1990_short.ref'
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"

# test on amip but with access from xarray dataset
@pytest.mark.parametrize("clim", ['EC23'])
def test_performance_indices_amip_xdataset(clim):
    file1 = 'tests/table/PI4_' + clim + '_amip_EC-Earth4_r1i1p1f1_1990_1990.yml'
    file2 = 'tests/table/PI4_' + clim + '_amip_1990_1990.ref'
    if os.path.isfile(file1):
        os.remove(file1)
    xfield = xr.open_mfdataset('tests/data/amip/output/oifs/*.nc', preprocess=xr_preproc)
    performance_indices('amip', 1990, 1990, numproc=2, climatology=clim,
                        config='tests/config.yml', xdataset=xfield)
    with open(file1, 'r', encoding='utf8') as f1, open(file2, 'r', encoding='utf8') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)
    assert are_dicts_equal(data1, data2, TOLERANCE), f"YAML files are not identical.\nData1: {data1}\nData2: {data2}"

def test_pi_plot(tmp_path):
    outputfile = tmp_path / 'PI4_heatmap.png'
    pi = PerformanceIndices('amip', 1990, 1990, config='tests/config.yml', climatology='EC23')
    pi.prepare()
    pi.plot(mapfile=outputfile)
    assert os.path.isfile(outputfile), "Plot not created."


def test_ecplotter_title_override():
    """Test that title can be set in __init__ and overridden via kwargs."""
    # Default title when no title provided
    plotter1 = ECPlotter(
        diagnostic="performance_indices",
        modelname="EC-Earth4",
        expname="amip",
        year1=1990, year2=1991,
        regions=["Global"],
        seasons=["ALL"]
    )
    assert plotter1.title == "PERFORMANCE INDICES EC-Earth4 amip 1990 1991"
    
    # Custom title in __init__ is used
    custom_title = "My Custom Title"
    plotter2 = ECPlotter(
        diagnostic="performance_indices",
        modelname="EC-Earth4",
        expname="amip",
        year1=1990, year2=1991,
        regions=["Global"],
        seasons=["ALL"],
        title=custom_title
    )
    assert plotter2.title == custom_title

    # Title in kwargs overrides instance title
    mock_longname = "Test Variable"
    mock_data = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_cmip6 = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_longnames = [mock_longname]
    
    fig = plotter2.heatmap_comparison_pi(
        data_dict=mock_data,
        cmip6_dict=mock_cmip6,
        longnames=mock_longnames,
        storefig=False,
        title="Kwargs Override"
    )
    
    ax = fig.axes[0]
    assert ax.get_title() == "Kwargs Override"
