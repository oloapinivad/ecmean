"""Tests for ECPlotter class"""

from ecmean.libs.ecplotter import ECPlotter


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