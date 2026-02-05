Installation
============


ECmean4 is a lightweight python package, but it depends on some binaries for interpolation and netcdf/grib data access, thus both installation options requires conda/mamba. 
We recommend to use `mamba <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`_ since it provides a lighter and deal in a better way with dependencies.

Using PyPi
----------

.. warning::

	Please note that although ECmean4 is distributed via PyPi, it depends on packages that currently are available only on conda-forge and on configuration files available from the GitHub repository. Therefore, the installation via pip requires the creation of a conda environment as well as the clone from the repository.


It will bring you the last version available on PyPi.
You can create a conda/mamba environment which incudes the python, `eccodes <https://github.com/ecmwf/eccodes-python>`_ and `xESMF <https://xesmf.readthedocs.io/en/latest/>`_ dependencies, and then install ECmean4.
However, you should start by cloning the repository from GitHub, since the configuration files used for running ECmean4 are placed there ::

    > git clone https://github.com/oloapinivad/ECmean4.git
    > mamba create -n ecmean python xesmf eccodes
    > mamba activate ecmean
    > pip install ECmean4


Using GitHub
------------

This method will allow you to have access at the most recent ECmean4 version but it requires a bit more of effort.

As before, should clone from the Github Repository ::

    > git clone https://github.com/oloapinivad/ECmean4.git
    
.. note ::

    Please note that if you clone with HTTPS you will not be able to contribute to the code, even if you are listed as collaborator.
    If you want to be a developer you should clone with SSH and you should add your own SSH key on the GitHub portal: 
    please check the `procedure on the Github website <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_ .

Then you can through the ECmean4 folder ::

    > cd ECmean4

and then you can set up the conda/mamba environment ::

    > mamba env create --name ecmean -f environment.yml

Then you should activate the environment ::

    > mamba activate ecmean


Checking everything is ok
-------------------------

From now on the two command line function of ECmean4 (``global_mean`` and ``performance_indices``) should be available in your environment. 
You can test by running in shell command line and you should and output as::

    > global_mean
    > usage: global_mean [-h] [-i INTERFACE] [-c CONFIG] [-j NUMPROC] [-l LOGLEVEL] [-o OUTPUTDIR] [--model MODEL] [--ensemble ENSEMBLE] [--consortium CONSORTIUM] [--mip MIP] [-s] [--version] [--trend] [--line] [--reference {EC23}] [--addnan] EXP Y1 Y2 [-i INTERFACE] EXP Y1 Y2 
    > global_mean: error: the following arguments are required: EXP, Y1, Y2

You can also run tests by simply calling ``pytest`` - as long as you have the corresponding Python package installed - from the ECmean4 folder ::

    > python -m pytest

Requirements
------------

The required packages are listed in ``environment.yml`` and in ``pyproject.toml``.
A secondary environment available in  ``dev-environment.yml`` can be used for development, including testing capabilities and jupyter notebooks. 

.. note::
	Both Unix and MacOS are supported. Python >3.9 is requested.




