Configuration
=============

Configuration file
------------------
A template configuration file is included in the repository, including the folder definition and all the details that can be manually adjusted. 
You will need to copy the original template to create your own local configuration file ::
	
    > cp config.tmpl config.yml 

and then edit the required folder before running the script. 

General configuration
---------------------

These are the mandatory properties to be set up before running ecmean.

interface
	The interface to access the model files (see below). So far supported interfaces are ``EC-Earth4``, ``CMIP6``, ``CMIP5``. 
model	
	The mode name: it could be ``EC-Earth4`` if running with the same interface, but could be any of the CMIP5/CMIP6 models.
dirs: exp
	Where the data from the experiments are
dirs: tab
	Where the output should be placed
dirs: fig
	Where the output figures should be placed (for ``performance_indices`` only)
dirs: clm
	Where the ecmean climatology is installed, i.e. the ``ecmean/climatology`` folder

.. note::
	You could call a specific configuration file with ``-c config_myconfig.yml`` when executing the ecmean commands (see Usage section), so that you can have multiple models on the same machine.

Global Mean configuration
-------------------------

The configuration files defines also the details of the global mean. 
Modification to this is not necessary, but please be aware that the variable convention follow a CMOR-like definition.

**variables**: 
	the main block to control variables. It is divided in three sub-blocks: ``atm``, ``oce`` and ``ice``. Each of them contains the list of fields to be processed. 
	``atm`` includes atmospheric fields, ``oce`` includes oceanic fields and ``ice`` includes sea-ice fields. 
	``tab`` includes the list of fields to be reported in the output table.

**regions**: 
	the list of regions on which compute the PI. Four regions are supported. Supported regions are: ``Global`` (90S-90N), ``North Midlat`` (30N-90N), ``Tropical`` (30S-30N), ``South Midlat`` (90S-30S) 

**seasons**:
	the list of seasons on which compute the PI. Four standard seasons are supported expressed as 3-string character (e.g. ``DJF``). ``ALL`` defines the yearly average. Default includes yearly, winter and summer.

**reference**:
	the global mean reference climatology. Only EC23 is currently available

Performance indices configuration
---------------------------------

The configuration files defines also the details of the performance indices. 
Modification to this is not necessary, but please be aware that the variable convention follow a CMOR-like definition.

**variables**:
	The main block to control variables. All blocks should be list. It is divided in four sub-blocks: ``atm2d``, ``atm3d``, ``oce`` and ``ice``. Each of them contains the list of fields to be processed. 
	``atm2d`` includes atmospheric 2d fields, ``atm3d`` includes atmospheric 3d fields, ``oce`` includes oceanic fields and ``ice`` includes sea-ice fields.

**regions**: 
	the list of regions on which compute the PI. Four regions are supported. Default regions are: ``Global`` (90S-90N), ``North Midlat`` (30N-90N), ``Tropical`` (30S-30N), ``South Midlat`` (90S-30S).
	For EC24 climatology, other regions are also available: ``NH`` (20N-90N), ``SH`` (20S-90S), ``Equatorial`` (20S-20N), ``North Pole`` (60N-90N), ``South Pole`` (60S-90S).

**seasons**:
	the list of seasons on which compute the PI. ``ALL`` defines the yearly average. Also ``DJF`` and ``JJA`` are supported.

**climatology**:
	the climatology to be used. EC23 and EC24 are avaiable. 


Interface files
---------------

Conversion from model variables - as well as the correspondent file structure - is handled by a interface-dependent file ``ecmean/interfaces/interface_*.yml`` that can support EC-Earth4, CMIP5 and CMIP6 models. 
New interface files can be developed exploiting of the flexible file handling increasing the range of supported models. 

It is important to provide land-sea mask for both atmospheric and oceanic grid, and when possible - recommended for non-regular grid - it is important to provide also files indicating grid cell areas.
These latter are used for interpolation and weighted averages. ecmean has a few routines that tries to compute each grid cell area from lon/lat boundaries, but this can fail for non-regular grids.

.. note::
	It is not necessary to modify the interface file, but it could be required if - for example - your CMIP5/6 directory tree does not reflect exactly the one available on ESGF. 

Of course, if some specific management of the model grid is requested - for example if the model grid does not follow standard naming convention - it might be necessary to intervene directly within the code. 
A function named ``xr_preproc()``  within ``ecmean/libs/ncfixers.py``  might be adjusted in the case to allow for a correct interpretation of the model grid.


CMOR compatibility
------------------

It is possible to use ecmean tools also to analyze CMOR-like files for CMIP5 or CMIP6. This assumes a standard ESGF directory structure but you can change it by modifying the corresponding interface files ``ecmean/interfaces/interface_CMIP6.yml`` and ``ecmean/interfaces/interface_CMIP6.yml``.
In order to allow masking and interpolation you will need some files to guide ECmean. `fx` variables ``sftlf`` and ``areacella`` are requried for atmospheric domain,
while `Ofx` variables ``sftof``  and ``areacello``are required for oceanic domain. At least one of these files is required for each domain, but it is recommended to provide both.


