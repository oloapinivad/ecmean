# Changelog

All notable changes to ECmean4 will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

Unreleased is the current development version, which is currently lying in `main` branch.

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
