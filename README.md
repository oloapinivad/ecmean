[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/oloapinivad/ecmean/graphs/commit-activity)
[![Documentation Status](https://readthedocs.org/projects/ecmean/badge/?version=latest)](https://ecmean.readthedocs.io/en/latest/?badge=latest)
[![PyTest](https://github.com/oloapinivad/ecmean/actions/workflows/mambatest.yml/badge.svg)](https://github.com/oloapinivad/ecmean/actions/workflows/mambatest.yml)
[![Coverage Status](https://coveralls.io/repos/github/oloapinivad/ecmean/badge.svg)](https://coveralls.io/github/oloapinivad/ecmean)
[![PyPI version](https://badge.fury.io/py/ecmean.svg)](https://badge.fury.io/py/ecmean)
[![DOI](https://zenodo.org/badge/460093944.svg)](https://zenodo.org/doi/10.5281/zenodo.13834627)

![ecmean](docs/ecmean_smallest.png)

*A lightweight climate model evaluation tool*

ecmean is a lightweight parallelized tool for evaluation of basic properties of Global Climate Models, such as global mean and climate model performance indices.

It builds on the original ECmean which has been used for EC-Earth2 and EC-Earth3 evaluation, but it uses Python3 as a scripting language to perform lazy calls with ``Xarray``+``Dask`` and makes use of YML configuration files, with parallelization support with ``Multiprocess``. 
It works both on raw EC-Earth4 output and on CMOR model output from CMIP5 and CMIP6.

ecmean is under development, so please use it with caution!

A complete [ReadTheDocs Documentation](https://ecmean.readthedocs.io/en/latest/index.html) is available, please refer to this.
