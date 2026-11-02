"""Unit tests for the filename handling of ecmean.libs.files."""

from types import SimpleNamespace

from ecmean.libs.files import _expand_filename, _filter_filename_by_year

TEMPLATE = "{var}_Amon_{model}_{expname}_{ensemble}_gn_{year1}??-{year2}??.nc"


def test_expand_filename_cmor_identifiers():
    diag = SimpleNamespace(
        expname="hist-1950",
        frequency="*mon",
        ensemble="r1i1p2f1",
        grid="*",
        modelname="EC-Earth3P-HR",
        version="*",
        consortium="EC-Earth-Consortium",
        mip="HighResMIP",
    )
    path = _expand_filename("CMIP6/{mip}/{consortium}/{model}/{expname}/" + TEMPLATE, "tas", diag)
    assert str(path) == (
        "CMIP6/HighResMIP/EC-Earth-Consortium/EC-Earth3P-HR/hist-1950/tas_Amon_EC-Earth3P-HR_hist-1950_r1i1p2f1_gn_*??-*??.nc"
    )


def test_filter_filename_by_year():
    files = ["a/tas_gn_199001-199012.nc", "a/tas_gn_199101-199112.nc", "a/tas_gn_1985-1994.nc"]
    assert _filter_filename_by_year(TEMPLATE, files, 1990) == [files[0], files[2]]


def test_filter_filename_by_year_skips_unparsable():
    files = ["a/tas_gn_199001-199012.nc", "a/tas_gn_clim.nc", "a/tas_gn_1990xx.nc"]
    assert _filter_filename_by_year(TEMPLATE, files, 1990) == [files[0]]


def test_filter_filename_without_years():
    files = ["a/sftlf_fx_gn.nc"]
    assert _filter_filename_by_year("sftlf_fx_{grid}.nc", files, 1990) == files
