# Changelog

All notable changes to ecmean will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

Unreleased is the current development version, which is currently lying in `main` branch.

Fixed
- Importing ecmean no longer switches matplotlib to the `Agg` backend (figures were not shown in notebooks); only the `global_mean` and `performance_indices` CLIs set it

## [v0.3.0]

*IMPORTANT*:
1. 3D performance indices (ta, ua, va, hus) change values (#150)
2. New default climatologies: EC26-CMIP for PI and GM (EC23/EC24 deprecated). Results are not comparable with previous versions (#150)
3. Time means are now accumulated in float64 in a single pass: results may move at the last printed decimal (#168)
4. Breaking: the unused `silent` argument and `-s/--silent` flag are removed; Python 3.9 is no longer supported (minimum 3.10)

Added
- HighResMIP support: `CMIP6_highresmip` interface, nearest-neighbour remapping of unstructured atmospheric grids and of ocean grids with placeholder cells, and `EC26-CMIP-HR` PI climatology with per-model exclusions (#139)
- `consortium` and `mip` options in interface files, CLI and config (#139)
- Atmospheric land-sea mask is now optional (#132)
- LaTeX notation for units in the global mean heatmap (#129)
- Benchmark script `utils/benchmarking.py` with memory sampling and run comparison (#169)
- Python 3.15 support (#164); `AGENTS.md` for contributors and coding agents (#163)

Changed
- Seasonal means computed in a single pass with bounded memory: up to ~5x less memory per worker (#168)
- Performance: files opened and globbed once per variable, workers balanced by load (#155)
- Workers split per variable and shared GM/PI helpers; derived-variable formulas parsed with `ast` (#151, #157)
- `resolution` and config file are read more robustly, with clear errors (#156)
- Documentation: API docs, derived variables, Python usage
- Development: ruff pinned to 0.17.1, Dependabot for GitHub Actions, `pip`/`certifi` minimum versions

Fixed
- Vertical weights for 3D PIs with descending pressure levels (#150)
- A failing variable no longer drops the other variables of its worker (#151)
- `--addnan` forwarded to GM plots; `--line` works with missing variables (#152, #153)
- `heatmap_plot()` loads `data`/`base` from YAML paths (#154)
- Grid detection, empty/missing file patterns and AMIP/OMIP without ocean mask (#156)
- `xr_preproc` no longer fails on conflicting coordinate names and renames `air_pressure` `lev` axes to `plev` (#139)

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
