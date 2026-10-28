# Changelog

All notable changes to ecmean will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

Unreleased is the current development version (target v0.3.0), which is currently lying in `main` branch.

*IMPORTANT*: 
1. 3D performance indices (ta, ua, va, hus) change values
2. The new default PI climatology is EC26-CMIP6, and the new GM climatology is now EC26-CMIP. Results cannot be compared against previous ecmean version

- Fix vertical weights for 3D performance indices with descending pressure levels (#150)
- CMIP6 reference PIs for 3D fields regenerated for EC26-CMIP and EC26-HIST; 2D references unchanged (#150)
- Default PI climatology is now EC26-CMIP; EC23 and EC24 are deprecated (#150)
- Default global mean reference is now EC26-CMIP; EC23 is deprecated (#150)
- A variable failing in a worker no longer drops the other variables of the same worker: it is logged and stored as NaN, and dead workers raise an explicit error (#151)
- `addnan` option and `--addnan` flag of global mean are now forwarded to the plot, so variables without reference are hidden by default (#152)
- `global_mean --line` no longer crashes when variables of the tuning table are missing, as in OMIP/AMIP runs: they are written as NaN (#153)
- Remove the unused `silent` argument of `GlobalMean`, `PerformanceIndices`, `Diagnostic` and of the wrapper functions, and the `-s/--silent` command line flag: it never had any effect (breaking for calls passing `silent`)
- Documentation: document the variable keys and the derived variables of the interface files; complete docstrings of `global_mean()`, `performance_indices()` and `Diagnostic`, add Python API usage to the Sphinx docs and remove the non-working `./script.py` instructions
- Workers split into per-variable functions, with variable loading shared between GM and PI in `load_variable()` (#151)
- Formulas of derived variables are parsed with Python's `ast` module instead of a hand-written tokenizer, with the same supported syntax (#157)
- Performance: input files are opened once per variable, globbed once per variable instead of once per year, and variables are split across workers by total load (#155)
- `consortium` and `mip` can be used in the interface files (`--consortium`/`--mip` command line flags, `consortium`/`mip` arguments, `model` section of the config); `ensemble`, `consortium` and `mip` are read from the config when not given, the CMIP6 interfaces use them in `basedir`, the esgpull interface accepts any month in the filenames, and filenames whose years cannot be parsed are skipped with a warning (#XXX)
- Development: ruff pinned to 0.17.1 (pyproject, pre-commit and AGENTS.md)
- Robustness fixes in `libs/support.py` (#156): `identify_grid` checks `lat.ndim` for curvilinear grids and compares latitude spacings with `np.isclose`; `check_file_exist` raises a clear error for empty strings and for patterns matching no file instead of `IndexError`
- The atmospheric land-sea mask is no longer mandatory: without `maskfile`, `Supporter` sets `atmmask` to `None` and `Diagnostic.configure_amip_omip_cpld()` skips, with a warning, the atmospheric variables that require land/ocean masking (#132)
- *IMPORTANT*: global mean and performance indices compute the time means of all seasons in a single pass over the data with the new `season_means()` (`libs/general.py`), loading one block of whole time chunks at a time (xarray/dask) instead of the whole record: worker memory no longer grows with the length of the record (PI `ta` of BCC-CSM2-HR over 30 years: 9.6 GB to 2.0 GB; GM `tas` of CMCC-CM2-VHR4 over 30 years: 4.4 GB to 1.1 GB, 3.2 GB to 1.1 GB with trends). Means are accumulated in float64, so results move at the last printed decimal (PIs by at most 0.001) (#168)
- New `CMIP6_highresmip` interface, using a data file (`tas`/`tos`) as `gridfile` for HighResMIP models without fx/Ofx files; `Supporter` loads only the first time step of the area/grid file (#139)
- `xr_preproc` no longer renames a coordinate onto an existing name (e.g. a 2D `latitude` auxiliary coordinate next to the 1D `lat`, as in BCC-CSM2-HR ocean files), which raised `ValueError: the new name 'lat' conflicts` (#139)
- Unstructured atmospheric grids (e.g. CESM1-CAM5-SE-HR spectral element) are remapped with nearest neighbour weights, as the oceanic ones, instead of failing on a bilinear regridder (#139)
- New `EC26-CMIP-HR` PI climatology (same observations as EC26-CMIP), whose reference PIs are computed on HighResMIP `hist-1950` models by `utils/cmip6-clim-evaluate.py -c EC26 -t CMIP --highresmip`; the script now averages each variable over all the models providing it, instead of only the variables of the first model (#139)
- `Supporter` initialises `atmmask`/`atmarea` to `None`, so an OMIP run with remapping and no ocean mask no longer fails with `AttributeError` (#156)
- `Diagnostic` raises a clear `ValueError` if the global mean regions do not include `Global` or the seasons do not include `ALL`, which its tables and plots require (#156)
- The ecmean version printed in the welcome message is now stored in `Diagnostic.ecmean_version`, so `Diagnostic.version` only holds the `*` CMOR filename wildcard (#156)
- `performance_indices.resolution` is now read from the config file (argument > config > default `r360x180`) and a `ValueError` is raised if it is not in `SUPPORTED_RESOLUTION` (defined in `libs/climatology.py`) (only `r360x180` for now). `Diagnostic.resolution` is now `None` for `GlobalMean`, where it was never used (#156)
- Without `-c`/`--config`, `config.yml` is now searched in the current directory instead of next to the source tree (`libs/../../config.yml`), with a clear `FileNotFoundError` if it is missing (#156)
- `ECPlotter.heatmap_plot()` loads `data` and `base` from YAML files when given as paths (`str` or `Path`), as documented, instead of parsing the path string itself (#154)
- Units in the global mean heatmap are shown in LaTeX notation (e.g. `W/m2` -> W m$^{-2}$) through `ecplotter.unit_to_latex()`, consistently with AQUA; text and YAML tables are unchanged (#129)
- Python 3.9 is no longer supported: the minimum version is now 3.10

## [v0.2.2]
- Remove a wrong dependency with esmpy

## [v0.2.1]
- Improve the netcdf fixers to make it easier process multiple data (#146)

## [v0.2.0]

*IMPORTANT*: New project name is now `ecmean` 

- Ruff is now enable as formatter/checker for the entire project (#145)
- Expand tests for new climatologies (#145)
- New climatologies EC26 for both global mean and performance indices (#135, #143, #144)
- Various speed up improvement for ECplotter and multiprocessing (#137)
- Support for python==3.14 (#136)
- Refactored colorbar label in performance indices plots; removed kwargs direct title propagation (#133)

## [v0.1.16]

- Allow title override in global mean and performance indices plots (#130)
- Support for EC-Earth4 regular grid from OIFS

## [v0.1.15]

- Fixed a bug in the heatmap production for global mean

## [v0.1.14]

- Cleaner command line arguments, notably loglevel -l instead -v (#120)
- Refactor of plotting routines in to the `ECPlotter()` class (#120)
- Output file in lower cases, e.g `pdf` instead of `PDF` (#120)
- Safe checks in case of missing area/mask files to activate only AMIP/OMIP runs (#122)
- Minor code cleaning using more pythonic approach (#121)
- Cleaner configuration files (#115)
- New EC24 climatology introduced and removal of RK08 climatology (#116)
- New region definitions, available in EC24 climatology (#116) 

## [v0.1.13]

- Support for python==3.13

## [v0.1.12]

- Allowing for configuration file as dictionary (#106)
- GlobalMean and PerformanceIndices classes introduced (#110, #111)

## [v0.1.11]

- Pinning python<3.13

## [v0.1.10]

- Expand using first year for grdfile and areafile (#100)
- Remove pin on numpy and extend documentation on EC23 climatology (#101)

## [v0.1.9]

- Specify numpy<2.0 to ensure all funcionalities

## [v0.1.8]

- Support for python 3.12 (#97)
- Minor code improvements, plus update docs and benchmarking  (#98)

## [v0.1.7]

- global_mean can plot also values where observation are missing through the `--addnan` option (#95)
- Verbosity level reduced in both global mean and performance indices (#95)
- New colorscale `vlag` for global_mean (#95)

## [v0.1.6]

- Remove support for python 3.8
- Refactor tests so that comparison is element-wise and not file-wise (Allow xesmf>0.8 and refactor the tests #93)
- Simple support for allow running the code on MacOS and pinning of xesmf<0.8 to avoid code slowdown (support for macos #86)
- Matplotlib pinning for solve temporary seaborn bug (Various updates and fixing the issue of matplotlib #89)

## [v0.1.5]

- Specifying output directory as an argument (Support more general output directory #87)

## [v0.1.4]

- Added the `changelog` file
- Better representation of mask file search and loading (#80)
- Fix of global mean temperature computaton (#84)
- Minor EC-Earth4 related fixes

## [v0.1.3]

- New logging facility
- Support for python 3.11

## [v0.1.0]

First version of ECmean4 published on pypi
