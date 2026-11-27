Installation
============


|ecmean| is a lightweight python package, but it depends on some binaries for interpolation and netcdf/grib data access, thus both installation options require conda/mamba.
We recommend to use `mamba <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`_ since it provides a lighter solution and deals better with dependencies.

.. warning::

     Since |ecmean| v0.2, the package is called ecmean instead of ECmean4.
     The old name is still available on PyPi but it will not be updated anymore, thus we recommend to use the new name for installation and usage.

Using Conda/Mamba
-----------------

This is the recommended installation method, since it will install all the dependencies and binaries needed for |ecmean| to work properly.
You can create a conda/mamba environment which includes python with

    > mamba create -n ecmean ecmean

Using PyPi
----------

.. note::

	Please note that although |ecmean| is distributed via PyPi, it depends on packages that currently are available only on conda-forge and on configuration files available from the GitHub repository. Therefore, the installation via pip requires the creation of a conda environment as well as the clone from the repository.


It will bring you the latest version available on PyPi.
You can create a conda/mamba environment which includes python, `eccodes <https://github.com/ecmwf/eccodes-python>`_ and `xESMF <https://xesmf.readthedocs.io/en/latest/>`_ dependencies, and then install ecmean.
However, you should start by cloning the repository from GitHub, since the configuration files used for running ecmean are placed there ::

    > git clone https://github.com/oloapinivad/ecmean.git
    > mamba create -n ecmean "python>=3.10" "xesmf>=0.8" eccodes
    > mamba activate ecmean
    > pip install ecmean


Using GitHub
------------

This method will allow you to have access to the most recent ecmean version but it requires a bit more effort.

As before, should clone from the Github Repository ::

    > git clone https://github.com/oloapinivad/ecmean.git

.. note ::

    Please note that if you clone with HTTPS you will not be able to contribute to the code, even if you are listed as a collaborator.
    If you want to be a developer you should clone with SSH and you should add your own SSH key on the GitHub portal:
    please check the `procedure on the Github website <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_ .

Then you can go through the ecmean folder ::

    > cd ecmean

and then you can set up the conda/mamba environment ::

    > mamba env create --name ecmean -f environment.yml

Then you should activate the environment ::

    > mamba activate ecmean


Checking everything is ok
-------------------------

From now on the two command line functions of |ecmean| (``global_mean`` and ``performance_indices``) should be available in your environment.
You can test by running in the shell command line and you should see an output like::

    > global_mean
    > usage: global_mean [-h] [-s] [-t] [-l] [-o FILE] [-m MODEL] [-c CONFIG] [-v LOGLEVEL] [-j NUMPROC] [-e ENSEMBLE] [-i INTERFACE] EXP Y1 Y2
    > global_mean: error: the following arguments are required: EXP, Y1, Y2

You can also run tests by simply calling ``pytest`` - as long as you have the corresponding Python package installed - from the ecmean folder ::

    > python -m pytest

Requirements
------------

The required packages are listed in ``environment.yml`` and in ``pyproject.toml``.
A secondary environment is available in ``environment-dev.yml`` and can be used for development, including testing capabilities and jupyter notebooks.

.. note::
	Both Unix and MacOS are supported. Python >=3.10 is requested.
