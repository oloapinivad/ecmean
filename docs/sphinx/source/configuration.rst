Configuration
=============

Configuration file
------------------
A template configuration file is included in the repository, with the folder definitions and all the details that can be manually adjusted.
You will need to copy the original template to create your own local configuration file ::

    > cp config.tmpl config.yml

and then edit the required folders before running the script.

General configuration
---------------------

These are the mandatory properties to be set up before running ecmean.

interface
	The interface to access the model files (see below). So far supported interfaces are ``EC-Earth4``, ``CMIP6``, ``CMIP5``.
model
	The model name: it could be ``EC-Earth4`` if running with the same interface, but could be any of the CMIP5/CMIP6 models.
dirs: exp
	Where the data from the experiments are
dirs: tab
	Where the output should be placed
dirs: fig
	Where the output figures should be placed (for ``performance_indices`` only)

.. note::
	You could call a specific configuration file with ``-c config_myconfig.yml`` when executing the |ecmean| commands (see Usage section), so that you can have multiple models on the same machine.

Global Mean configuration
-------------------------

The configuration file also defines the details of the global mean.
Modifications to this are not necessary, but please be aware that the variable convention follows a CMOR-like definition.

**variables**:
	the main block to control variables. It is divided in three sub-blocks: ``atm``, ``oce`` and ``ice``. Each of them contains the list of fields to be processed.
	``atm`` includes atmospheric fields, ``oce`` includes oceanic fields and ``ice`` includes sea-ice fields.
	``tab`` includes the list of fields to be reported in the output table.

**regions**:
	the list of regions on which compute the PI. Four regions are supported. Supported regions are: ``Global`` (90S-90N), ``North Midlat`` (30N-90N), ``Tropical`` (30S-30N), ``South Midlat`` (90S-30S).
	For EC26 reference, other regions are also available: ``NH`` (20N-90N), ``SH`` (20S-90S), ``Equatorial`` (20S-20N), ``North Pole`` (60N-90N), ``South Pole`` (60S-90S).

**seasons**:
	the list of seasons on which to compute the PI. Four standard seasons are supported, expressed as 3-character strings (e.g. ``DJF``). ``ALL`` defines the yearly average. Default includes yearly, winter and summer.
	``ALL`` and the ``Global`` region are mandatory for the global mean, since its tables and plots rely on them: an error is raised if they are missing.

**reference**:
	the global mean reference climatology. EC26-CMIP (default, for the period 1985-2014), EC26-HIST (for the period 1981-2010) and EC26-PDAY (for the period 2000-2024) are available. EC23 (without the tas global variable and for the period 1990-2020) is deprecated.

Performance indices configuration
---------------------------------

The configuration file also defines the details of the performance indices.
Modifications to this are not necessary, but please be aware that the variable convention follows a CMOR-like definition.

**variables**:
	The main block to control variables. All blocks should be lists. It is divided into four sub-blocks: ``atm2d``, ``atm3d``, ``oce`` and ``ice``. Each of them contains the list of fields to be processed.
	``atm2d`` includes atmospheric 2d fields, ``atm3d`` includes atmospheric 3d fields, ``oce`` includes oceanic fields and ``ice`` includes sea-ice fields.

**regions**:
	the list of regions on which to compute the PI. Four regions are supported. Default regions are: ``Global`` (90S-90N), ``North Midlat`` (30N-90N), ``Tropical`` (30S-30N), ``South Midlat`` (90S-30S).
	For EC24 and EC26 climatologies, other regions are also available: ``NH`` (20N-90N), ``SH`` (20S-90S), ``Equatorial`` (20S-20N), ``North Pole`` (60N-90N), ``South Pole`` (60S-90S).

**seasons**:
	the list of seasons on which to compute the PI. ``ALL`` defines the yearly average. Also ``DJF`` and ``JJA`` are supported.

**climatology**:
	the climatology to be used. EC26-CMIP (default, for the period 1985-2014) and EC26-HIST (for the period 1981-2010) are available. EC24 and EC23 are deprecated.

**resolution**:
	the horizontal resolution of the climatology. Only ``r360x180`` (default) is currently available, an error is raised otherwise.


Interface files
---------------

Conversion from model variables - as well as the correspondent file structure - is handled by an interface-dependent file ``ecmean/interfaces/interface_*.yml`` that can support EC-Earth4, CMIP5 and CMIP6 models.
New interface files can be developed exploiting the flexible file handling to increase the range of supported models.

It is important to provide land-sea masks for both atmospheric and oceanic grids, and when possible - recommended for non-regular grids - it is important to also provide files indicating grid cell areas.
These latter are used for interpolation and weighted averages. ecmean has a few routines that try to compute each grid cell area from lon/lat boundaries, but this can fail for non-regular grids.

If the atmospheric land-sea mask (``maskfile``) is not available, ecmean still runs using the ``areafile`` or ``gridfile``, but the atmospheric variables that require land or ocean masking (e.g. ``pr_land``, ``pr_oce`` in the global mean) are skipped with a warning.

.. note::
	It is not necessary to modify the interface file, but it could be required if - for example - your CMIP5/6 directory tree does not reflect exactly the one available on ESGF.

Of course, if some specific management of the model grid is requested - for example if the model grid does not follow standard naming conventions - it may be necessary to intervene directly within the code.
A function named ``xr_preproc()``  within ``ecmean/libs/ncfixers.py``  might be adjusted in this case to allow for a correct interpretation of the model grid.


Variables in the interface file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each entry of the ``variables`` block is named with the CMOR-like name used by |ecmean| (e.g. ``tas``), which is also the variable
looked for in the model files, and accepts the following keys:

varname
	A human-readable description of the variable.
filetype
	One of the entries of the ``filetype`` block, which defines where the files are and which component (and thus domain, atmosphere or ocean) the variable belongs to.
units
	Optional. The units of the model data. If missing, the ``units`` attribute of the netCDF variable is used (for a derived variable, the one of the first variable of the formula);
	if also this is missing, the variable is assumed to be a fraction. It is converted to the units of the reference/climatology.
direction
	Optional, ``up`` or ``down`` (default). It is the sign convention of the fluxes in the model, compared with the one of the reference to flip the sign when needed.
derived
	Optional. A formula to compute the variable from other variables stored in the same filetype (see below).

Derived variables
^^^^^^^^^^^^^^^^^

Variables that are not available in the model output can be computed from other variables with the ``derived`` key.
The formula is written in terms of variable names, and all the variables it contains are read from the files of the ``filetype`` of the derived variable.
For example, from ``ecmean/interfaces/interface_CMIP6.yml``::

   net_toa:
     varname: 'Net TOA'
     units: W/m2
     derived: 'rsdt-rsut-rlut'
     filetype: atm2d
     direction: down

   net_sfc:
     varname: 'Net Surface'
     units: W/m2
     filetype: atm2d
     direction: down
     derived: 'rsds-rsus+rlds-rlus-hfls-hfss-prsn*334000'

The supported syntax is: variable names, numbers (decimals and scientific notation, e.g. ``334000`` or ``3.34e5``),
the four operations ``+ - * /``, unary minus, parentheses and spaces. Anything else (e.g. functions) raises a ``ValueError``.
The formulas are evaluated with the ``_eval_formula()`` function of ``ecmean/libs/formula.py``.

.. note::
	For a derived variable, ``units`` and ``direction`` describe the result of the formula, so set ``units`` explicitly when the first variable of the formula is not in the units of the result.

CMOR compatibility
------------------

It is possible to use |ecmean| tools also to analyze CMOR-like files for CMIP5 or CMIP6. This assumes a standard ESGF directory structure but you can change it by modifying the corresponding interface files ``ecmean/interfaces/interface_CMIP5.yml`` and ``ecmean/interfaces/interface_CMIP6.yml``.
In order to allow masking and interpolation you will need some files to guide |ecmean|. ``fx`` variables ``sftlf`` and ``areacella`` are required for the atmospheric domain,
while ``Ofx`` variables ``sftof`` and ``areacello`` are required for the oceanic domain. At least one of these files is required for each domain, but it is recommended to provide both.
