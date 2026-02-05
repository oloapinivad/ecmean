"""Tests for ECPlotter class"""

from ecmean.libs.ecplotter import ECPlotter


def test_ecplotter_title_override():
    """Test that title defaults to auto-generated title and can be overridden via title parameter."""
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

    # Title defaults to auto-generated when not provided
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

    # Title overrides default title
    fig2 = plotter1.heatmap_comparison_pi(
        data_dict=mock_data,
        cmip6_dict=mock_cmip6,
        longnames=mock_longnames,
        storefig=False,
        title="Custom Title Override"
    )
    
    ax2 = fig2.axes[0]
    assert ax2.get_title() == "Custom Title Override"


def test_ecplotter_global_mean_title():
    """Test that title parameter works for global mean plots."""
    plotter = ECPlotter(
        diagnostic="global_mean",
        modelname="EC-Earth4",
        expname="amip",
        year1=1990, year2=1991,
        regions=["Global"],
        seasons=["ALL"]
    )
    
    mock_longname = "Test Variable [units]"
    mock_data = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_mean = {mock_longname: {"ALL": {"Global": 1.0}}}
    mock_std = {mock_longname: {"ALL": {"Global": 0.1}}}
    mock_units = ["units"]
    
    # Test without reference
    fig1 = plotter.heatmap_comparison_gm(
        data_dict=mock_data,
        mean_dict=mock_mean,
        std_dict=mock_std,
        units_list=mock_units,
        storefig=False,
        title="Custom Global Mean Title"
    )
    
    ax1 = fig1.axes[0]
    assert ax1.get_title() == "Custom Global Mean Title"
    
    # Test with reference - colorbar label should include reference
    fig2 = plotter.heatmap_comparison_gm(
        data_dict=mock_data,
        mean_dict=mock_mean,
        std_dict=mock_std,
        units_list=mock_units,
        storefig=False,
        title="Custom Global Mean Title",
        reference="EC23"
    )
    
    ax2 = fig2.axes[0]
    assert ax2.get_title() == "Custom Global Mean Title"
    
    # Check colorbar label includes reference
    cbar = fig2.axes[-1]  # Colorbar is typically the last axis
    cbar_label = cbar.get_ylabel()
    assert "EC23" in cbar_label, f"Expected 'EC23' in colorbar label, got: {cbar_label}"