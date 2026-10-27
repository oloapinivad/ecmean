Global Mean
===========

Main concepts
^^^^^^^^^^^^^

The ``global_mean`` command is based on the ``global_mean.py`` script which computes the global averages for many dynamical and physical fields.
It compares the output against a set of pre-computed climatological values defined in the reference climatology (see below) and produces a table with the comparison.
This includes the most important dynamical and physical fields for both the atmosphere and the ocean (e.g. land temperature, salinity, etc.).
Different datasets are taken in consideration, providing an estimate of the interannual variability in the form of standard deviation.

Most importantly, it provides estimates for the radiative budget (including clouds radiative forcing) and for the hydrological cycle (including integrals over land and ocean)
and other quantities useful for fast model assessment and for model tuning.

Usage
^^^^^

Running the global mean evaluation is rather simple::

   global_mean EXP Y1 Y2


Positional Arguments
--------------------

  EXP
    Experiment identification.

  Y1
    Starting year of analysis.

  Y2
    Final year of analysis.

Optional Arguments
------------------

.. option:: -h, --help

   Show this help message and exit.

.. option:: --version

   Show |ecmean|'s version number and exit.

.. option:: -l LOGLEVEL, --loglevel LOGLEVEL

   Define the level of logging. The default is 'warning'.

.. option:: -j NUMPROC

   Specify the number of processors to use.

.. option:: -c CONFIG, --config CONFIG

   Set up a specific configuration file. The default is ``config.yml`` in the current directory.

.. option:: -i INTERFACE, --interface INTERFACE

   Set up a specific interface file, overriding the configuration specified in ``config.yml``.

.. option:: -m MODEL, --model MODEL

   Specify the model name, overriding the configuration specified in ``config.yml``.

.. option:: -e ENSEMBLE, --ensemble ENSEMBLE

   Specify the variant label (ripf number for cmor), overriding the configuration specified in ``config.yml``. Default is r1i1p1f1.

.. option:: --consortium CONSORTIUM

   Specify the consortium name (CMOR institution_id, e.g. EC-Earth-Consortium), overriding the configuration specified in ``config.yml``. Default is any (``*``).

.. option:: --mip MIP

   Specify the MIP name (CMOR activity_id, e.g. CMIP, HighResMIP), overriding the configuration specified in ``config.yml``. Default is CMIP.

.. option:: -o DIR, --outputdir DIR

   Specify the path of the output directory. This will create YAML and PDF folders for tables and figures.

.. option:: --reference REFERENCE

   Specify the reference dataset to use for comparison. The default is EC26-CMIP. Other options include EC26-HIST, EC26-PDAY and the deprecated EC23.

.. option:: --trend

   Compute trends on multiple years. This option is only available in table format.

.. option:: --line

   Append a single line to the table.

.. option:: --addnan

   Activate to plot also in the heatmap also fields which does not have a comparison against observations. Default is False.

Example
^^^^^^^

Usage example for CMIP5::

   global_mean historical 1990 1999 -j 12 -m EC-EARTH -e r1i1p1 -i CMIP5

This will compute the global mean for member r1i1p1 of the EC-EARTH model in the CMIP5 historical experiment.

Output
^^^^^^

Data are stored in machine-readable format in a YAML file, which includes much more details such as the global and regional mean over different seasons.
In addition, |ecmean| produces also a figure including a more detailed comparison for different seasons and regions.
This is available only for the datasets for which we have access to a gridded dataset.

.. figure:: _static/globaltestfigure.png
   :align: center
   :width: 600px
   :alt: Global mean figure for EC-Earth3

   An example for a single year of the EC-Earth3 historical r1i1p1f1 simulation. Colors indicate the model bias as standard deviation of the interannual variability from observations.
   Blue implies negative bias, red positive bias. In each of the tiles the larger number shows the model value, while the smaller one is the reference value.

Finally, a txt table including the comparison with some predefined dataset, for the global mean yearly averages.

References available
^^^^^^^^^^^^^^^^^^^^

Currently, two main different references for climatological values are available, EC26 and EC23, covering different observation periods.

.. warning::
   EC23 is deprecated and will be removed in a future release. Please use EC26-CMIP, EC26-HIST or EC26-PDAY.

EC26
----
EC26 is the updated reference framework for global mean observational datasets, which provide a flexible framework for evaluating climate models.
It provides temporally consistent baselines tailored to different model forcing configurations, declined in three configurations (CMIP, HIST, PDAY) to match the intended timeframes.
It also includes a global near-surface air temperature target, which was not available in the older reference.
The variables are derived from a combination of observational and reanalysis products (e.g. CRU v4.09, Berkeley Earth, ERA5, MSWEP v2.80, CERES EBAF v4.2.1, ESA-CCI), depending on the physical quantity considered.

All configurations share the same variable definitions and masking strategy (global, land, ocean), but differ in their temporal averaging window to ensure consistency with the intended model experiments.
All metadata (datasets, masks, periods and other properties) are defined in the corresponding YAML configuration file.

EC26-CMIP (1985–2014)
~~~~~~~~~~~~~~~~~~~~~
This reference dataset is designed for the evaluation of CMIP6 historical simulations against a consistent observational baseline.
It is aligned with the CMIP6 historical period (1985–2014), ensuring temporal consistency between model climatologies and observational targets.
Radiative fluxes are restricted to the satellite era (2000–2014), while ocean salinity follows its specific observational availability window.


.. include:: tables/reference_EC26-CMIP.rst

EC26-HIST (1981–2010)
~~~~~~~~~~~~~~~~~~~~~
This configuration is designed for comparison with model simulations using historical forcing or present-day forcing fixed around 1990.
The reference period spans 1981–2010 where observational coverage allows. Radiative fluxes follow the CERES satellite era (starting in 2000), while other variables use the 1981–2010 window.


.. include:: tables/reference_EC26-HIST.rst

EC26-PDAY (2000–2024)
~~~~~~~~~~~~~~~~~~~~~
This configuration is intended for evaluation of model simulations using present-day forcing conditions representative of the 2010–2012 period.
The reference period spans 2000–2024 (or the maximum available year depending on dataset availability; e.g. 2023 for ESA-CCI-L4 products).
By using a more recent averaging window, EC26-PDAY reflects contemporary radiative balance and hydrological cycle conditions, making it suitable for fixed present-day forcing experiments.

.. include:: tables/reference_EC26-PDAY.rst

EC23 (deprecated)
-----------------
This is from the old version of |ecmean| and does not include values for global tas.
This reference dataset collects global mean observational targets used by the global_mean.py script to compute global averages.
The variables are derived from a combination of observational and reanalysis products (e.g. CRU, ERA5, MSWEP, CERES-EBAF, ESA-CCI, Wild 2020), depending on the physical quantity considered.
Most fields are defined over the 1991–2020 period, while other variables use shorter observational windows due to data availability.
All metadata (datasets, masks, periods and other properties) are defined in the corresponding YAML configuration file.

.. include:: tables/reference_EC23.rst


Reference climatology computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reference climatology are computed by the ``ecmean/utils/reference-create.py`` script, which is included in the repository for documentation.
It is based on a YML file which is tells the script where to retrieve the data, identifying all the required data folder and names.
Of course, in the remote case you would like to develop a new climatology, you can create your own YML file and run the script to produce the reference climatology
Examples are the `create-reference-wilma-EC26.yml` and `create-reference-wilma-EC23.yml` files, which are used to produce the EC26 and EC23 reference climatology, respectively.
The results are produced into a YML file for in ``ecmean/reference/gm_reference_EC**.yml`` which includes the global and regional mean
over different seasons as well the interannual standard deviation. Full details on the datasets used are found there.
