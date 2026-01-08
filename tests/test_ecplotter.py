"""Tests for ECPlotter class"""

from ecmean.libs.ecplotter import ECPlotter


def test_ecplotter_title_override():
    """Test that title defaults to auto-generated title and can be overridden via kwargs."""
    # Default title when no title provided
    plotter1 = ECPlotter(
        diagnostic="performance_indices",
        modelname="EC-Earth4",
        expname="amip",
        year1=1990, year2=1991,
        regions=["Global"],
        seasons=["ALL"]
    )
    assert plotter1.default_title == "PERFORMANCE INDICES EC-Earth4 amip 1990 1991"

    # Title defaults to auto-generated when not provided in kwargs
    mock_longname = "Test Variable"
    mock_data = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_cmip6 = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_longnames = [mock_longname]
    
    fig1 = plotter1.heatmap_comparison_pi(
        data_dict=mock_data,
        cmip6_dict=mock_cmip6,
        longnames=mock_longnames,
        storefig=False
    )
    
    ax1 = fig1.axes[0]
    assert ax1.get_title() == "PERFORMANCE INDICES EC-Earth4 amip 1990 1991"

    # Title in kwargs overrides default title
    fig2 = plotter1.heatmap_comparison_pi(
        data_dict=mock_data,
        cmip6_dict=mock_cmip6,
        longnames=mock_longnames,
        storefig=False,
        title="Kwargs Override"
    )
    
    ax2 = fig2.axes[0]
    assert ax2.get_title() == "Kwargs Override"