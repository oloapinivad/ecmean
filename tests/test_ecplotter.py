"""Tests for ECPlotter class"""

from pathlib import Path

import pytest

from ecmean.libs.ecplotter import ECPlotter, unit_to_latex
from ecmean.libs.files import load_yaml


def test_ecplotter_title_override():
    """Test that title defaults to auto-generated title and can be overridden via title parameter."""
    # Default title when no title provided
    plotter1 = ECPlotter(
        diagnostic="performance_indices",
        modelname="EC-Earth4",
        expname="amip",
        year1=1990,
        year2=1991,
        regions=["Global"],
        seasons=["ALL"],
    )
    assert plotter1.default_title == "PERFORMANCE INDICES EC-Earth4 amip 1990 1991"

    # Title defaults to auto-generated when not provided
    mock_longname = "Test Variable"
    mock_data = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_cmip6 = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_longnames = [mock_longname]

    fig1 = plotter1.heatmap_comparison_pi(data_dict=mock_data, cmip6_dict=mock_cmip6, longnames=mock_longnames, storefig=False)

    ax1 = fig1.axes[0]
    assert ax1.get_title() == "PERFORMANCE INDICES EC-Earth4 amip 1990 1991"

    # Title overrides default title
    fig2 = plotter1.heatmap_comparison_pi(
        data_dict=mock_data, cmip6_dict=mock_cmip6, longnames=mock_longnames, storefig=False, title="Custom Title Override"
    )

    ax2 = fig2.axes[0]
    assert ax2.get_title() == "Custom Title Override"


def test_ecplotter_global_mean_title():
    """Test that title parameter works for global mean plots."""
    plotter = ECPlotter(
        diagnostic="global_mean",
        modelname="EC-Earth4",
        expname="amip",
        year1=1990,
        year2=1991,
        regions=["Global"],
        seasons=["ALL"],
    )

    mock_longname = "Test Variable [units]"
    mock_data = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_mean = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_std = {mock_longname: {"ALL": {"Global": 0.1}}}
    mock_units = ["W/m2"]

    # Test without reference
    fig1 = plotter.heatmap_comparison_gm(
        data_dict=mock_data,
        mean_dict=mock_mean,
        std_dict=mock_std,
        units_list=mock_units,
        storefig=False,
        title="Custom Global Mean Title",
    )

    ax1 = fig1.axes[0]
    assert ax1.get_title() == "Custom Global Mean Title"
    # units are shown in LaTeX notation (#129)
    assert ax1.get_yticklabels()[0].get_text().endswith("[W\u00a0m$^{-2}$]")

    # Test with reference - colorbar label should include reference
    fig2 = plotter.heatmap_comparison_gm(
        data_dict=mock_data,
        mean_dict=mock_mean,
        std_dict=mock_std,
        units_list=mock_units,
        storefig=False,
        title="Custom Global Mean Title",
        reference="EC26-CMIP",
    )

    ax2 = fig2.axes[0]
    assert ax2.get_title() == "Custom Global Mean Title"

    # Check colorbar label includes reference
    cbar = fig2.axes[-1]  # Colorbar is typically the last axis
    cbar_label = cbar.get_ylabel()
    assert "EC26-CMIP" in cbar_label, f"Expected 'EC26-CMIP' in colorbar label, got: {cbar_label}"


def test_heatmap_plot_yaml_paths():
    """data and base can be given as paths to YAML files (str or Path), with the same result as dictionaries (#154)"""
    datafile = "tests/table/PI_EC26-CMIP_cpld_1990_1990.ref"
    basefile = Path("ecmean/climatology/EC26-CMIP/pi_climatology_EC26-CMIP.yml")
    variables = list(load_yaml(datafile))
    plotter = ECPlotter(
        diagnostic="performance_indices",
        modelname="EC-Earth4",
        expname="cpld",
        year1=1990,
        year2=1990,
        regions=["Global", "North Midlat", "Tropical", "South Midlat"],
        seasons=["ALL", "JJA"],
    )

    fig_paths = plotter.heatmap_plot(data=datafile, base=basefile, variables=variables, storefig=False)
    fig_dicts = plotter.heatmap_plot(data=load_yaml(datafile), base=load_yaml(basefile), variables=variables, storefig=False)

    labels = [[t.get_text() for t in fig.axes[0].get_yticklabels()] for fig in (fig_paths, fig_dicts)]
    # one row per variable, plus the Total PI row
    assert labels[0] == labels[1]
    assert len(labels[0]) == len(variables) + 1


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("W/m2", "W m$^{-2}$"),
        ("mm/day", "mm day$^{-1}$"),
        ("m/century", "m century$^{-1}$"),
        ("Mm^2", "Mm$^{2}$"),
        ("kg m**-2 s-1", "kg m$^{-2}$ s$^{-1}$"),
        ("W/(m2 s)", "W m$^{-2}$ s$^{-1}$"),
        ("celsius", "celsius"),
        ("frac", "frac"),
        ("%", "$\\%$"),
        ("1", "1"),
        ("", ""),
        ("W m$^{-2}$", "W m$^{-2}$"),
    ],
)
def test_unit_to_latex(unit, expected):
    """Units are converted to LaTeX notation for plot labels, existing LaTeX is preserved (#129)"""
    assert unit_to_latex(unit) == expected
