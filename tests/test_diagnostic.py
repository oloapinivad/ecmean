"""Unit tests for the configuration checks of ecmean.libs.diagnostic."""

import copy
from types import SimpleNamespace

import pytest
import yaml

from ecmean.libs.diagnostic import Diagnostic

BASE_CONFIG = {
    "interface": "EC-Earth4",
    "model": {"name": "EC-Earth4"},
    "dirs": {"exp": "tests/data", "tab": "tests/table", "fig": "tests/figures"},
    "global_mean": {
        "variables": {"atm": ["tas"]},
        "regions": ["Global", "Tropical"],
        "seasons": ["ALL", "JJA"],
        "reference": "EC26-CMIP",
    },
    "performance_indices": {
        "variables": {"atm2d": ["tas"]},
        "regions": ["Global", "Tropical"],
        "seasons": ["ALL", "JJA"],
        "climatology": "EC26-CMIP",
    },
}


def _diagnostic(funcname, config=None, **kwargs):
    return Diagnostic("exp", 1990, 1990, BASE_CONFIG if config is None else config, funcname, **kwargs)


@pytest.mark.parametrize(
    "key, value, message",
    [("regions", ["Tropical"], "'Global' region"), ("seasons", ["JJA"], "'ALL' season")],
)
def test_global_all_mandatory_global_mean(key, value, message):
    config = copy.deepcopy(BASE_CONFIG)
    config["global_mean"][key] = value
    with pytest.raises(ValueError, match=message):
        _diagnostic("GlobalMean", config)


def test_global_all_not_required_for_performance_indices():
    config = copy.deepcopy(BASE_CONFIG)
    config["performance_indices"]["regions"] = ["NH", "SH"]
    config["performance_indices"]["seasons"] = ["JJA"]
    assert _diagnostic("PerformanceIndices", config).regions == ["NH", "SH"]


def test_versions_are_separate():
    diag = _diagnostic("GlobalMean")
    assert diag.version == "*"
    assert diag.ecmean_version != "*"


def test_resolution_default():
    assert _diagnostic("PerformanceIndices").resolution == "r360x180"


def test_resolution_from_config():
    config = copy.deepcopy(BASE_CONFIG)
    config["performance_indices"]["resolution"] = "r360x180"
    assert _diagnostic("PerformanceIndices", config).resolution == "r360x180"


@pytest.mark.parametrize("from_argument", [False, True])
def test_resolution_not_supported(from_argument):
    config = copy.deepcopy(BASE_CONFIG)
    kwargs = {}
    if from_argument:
        kwargs["resolution"] = "r90x45"
    else:
        config["performance_indices"]["resolution"] = "r90x45"
    with pytest.raises(ValueError, match="not supported"):
        _diagnostic("PerformanceIndices", config, **kwargs)


def test_default_config_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="config.yml"):
        _diagnostic("GlobalMean", config="")


def test_default_config_in_current_directory(tmp_path, monkeypatch):
    (tmp_path / "config.yml").write_text(yaml.safe_dump(BASE_CONFIG))
    monkeypatch.chdir(tmp_path)
    assert _diagnostic("GlobalMean", config="").modelname == "EC-Earth4"


MASK_REFERENCE = {
    "tas": {"mask": "global"},
    "tas_land": {"mask": "land"},
    "net_sfc": {"mask": "ocean"},
    "tos": {"mask": "ocean"},
}


@pytest.mark.parametrize(
    "funcname, section, attribute",
    [("GlobalMean", "global_mean", "var_all"), ("PerformanceIndices", "performance_indices", "field_all")],
)
def test_missing_atm_mask_drops_masked_variables(funcname, section, attribute, caplog):
    """Without atmospheric land-sea mask, only atmospheric variables requiring land/sea masking are dropped."""
    config = copy.deepcopy(BASE_CONFIG)
    atm = "atm" if funcname == "GlobalMean" else "atm2d"
    config[section]["variables"] = {atm: ["tas", "tas_land", "net_sfc"], "oce": ["tos"]}
    diag = _diagnostic(funcname, config)
    support = SimpleNamespace(atmareafile="grid.nc", atmmaskfile="", atmmask=None, oceareafile="area.nc", ocemaskfile="")

    diag.configure_amip_omip_cpld(support, MASK_REFERENCE)

    assert getattr(diag, attribute) == ["tas", "tos"]
    assert "tas_land requires a land-sea mask (mask: land)" in caplog.text
    assert "net_sfc requires a land-sea mask (mask: ocean)" in caplog.text


def test_missing_atm_mask_without_reference_keeps_variables():
    """Without reference, configure_amip_omip_cpld does not drop variables for a missing mask."""
    config = copy.deepcopy(BASE_CONFIG)
    config["global_mean"]["variables"] = {"atm": ["tas", "tas_land"]}
    diag = _diagnostic("GlobalMean", config)
    support = SimpleNamespace(atmareafile="grid.nc", atmmaskfile="", atmmask=None, oceareafile="", ocemaskfile="")

    diag.configure_amip_omip_cpld(support)

    assert diag.var_all == ["tas", "tas_land"]


def test_missing_atm_mask_only_masked_variables_raises():
    """If all variables need the missing mask, nothing is left to process."""
    config = copy.deepcopy(BASE_CONFIG)
    config["global_mean"]["variables"] = {"atm": ["tas_land"]}
    diag = _diagnostic("GlobalMean", config)
    support = SimpleNamespace(atmareafile="grid.nc", atmmaskfile="", atmmask=None, oceareafile="", ocemaskfile="")

    with pytest.raises(ValueError, match="No variables to process"):
        diag.configure_amip_omip_cpld(support, MASK_REFERENCE)


def test_cmor_identifiers_default():
    diag = _diagnostic("GlobalMean")
    assert (diag.ensemble, diag.consortium, diag.mip) == ("r1i1p1f1", "*", "CMIP")


def test_cmor_identifiers_from_config():
    config = copy.deepcopy(BASE_CONFIG)
    config["model"].update({"ensemble": "r1i1p2f1", "consortium": "EC-Earth-Consortium", "mip": "HighResMIP"})
    diag = _diagnostic("GlobalMean", config)
    assert (diag.ensemble, diag.consortium, diag.mip) == ("r1i1p2f1", "EC-Earth-Consortium", "HighResMIP")


def test_cmor_identifiers_arguments_override_config():
    config = copy.deepcopy(BASE_CONFIG)
    config["model"].update({"ensemble": "r1i1p2f1", "consortium": "EC-Earth-Consortium", "mip": "HighResMIP"})
    diag = _diagnostic("PerformanceIndices", config, ensemble="r2i1p1f1", consortium="CNRM-CERFACS", mip="CMIP")
    assert (diag.ensemble, diag.consortium, diag.mip) == ("r2i1p1f1", "CNRM-CERFACS", "CMIP")
