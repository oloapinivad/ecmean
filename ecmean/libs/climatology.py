#!/usr/bin/env python3
'''
Shared functions for Xarray to create climatology
'''

import logging
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

loggy = logging.getLogger(__name__)


SUPPORTED_REFERENCE = ['EC23', 'EC25', 'EC26-PDAY', 'EC26-CMIP', 'EC26-HIST']
SUPPORTED_CLIMATOLOGY = ['EC23', 'EC24', 'EC26-HIST', 'EC26-CMIP']

def variance_threshold(xvariance):
    """this defines the two thresholds (high and low) for filtering the dataset
    So far it is done on the 5-sigma of the log10 distribution"""
    f = np.log10(xvariance.where(xvariance > 0))
    m = float(np.mean(f).values)
    s = float(np.std(f).values)
    low = 10**(m - 5 * s)
    high = 10**(m + 5 * s)
    return low, high

# function to set absurd value from a specific dataset


def fix_specific_dataset(var, dataset, xfield):
    """in the case some dataset show unexpected values this can be filtered here"""
    # variance of SST under sea ice is almost zero. We need to get rid of those points
    if var == 'tos' and dataset == 'ESA-CCI-L4':
        # xfield = xfield.where(xfield > 271.15)
        xfield = xfield.where(xfield > 1 * 10**-2)
    return xfield


def full_histogram(field, figname, n_bins=100):
    """Compute the histogram of the full field before it is processed.
    this is done to check the presence of irrealistic values within the dataset
    dask.array.histogram is used to speed up the computation."""

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(15, 5))

    # test using underlying dask array
    mmm = da.nanmin(field.data).compute()
    xxx = da.nanmax(field.data).compute()
    extra = (xxx - mmm) / 20
    hist, bins = da.histogram(field.data, bins=n_bins, range=[mmm - extra, xxx + extra])
    x = 0.5 * (bins[1:] + bins[:-1])
    width = np.diff(bins)
    axs.bar(x, hist.compute(), width, log=True)
    axs.title.set_text('Complete original values ' + field.name)
    fig.savefig(figname)


def check_histogram(ymean, yvar, yvar_filtered, figname, n_bins=100):
    """Four histograms made for inspection of mean and variance of the field
    Mean field, variance and variance after filtering are passed and then plotted
    using histograms. log10 scales is used to highlight outliers."""

    fig, axs = plt.subplots(4, 1, sharey=True, tight_layout=True, figsize=(20, 15))

    # log 10 fields
    f = np.log10(yvar.where(yvar > 0))
    g = np.log10(yvar_filtered.where(yvar > 0))

    # stats
    avg = f.mean()
    sss = 5 * f.std()
    qqq = f.quantile([0.25, 0.75])
    iqr = qqq[1] - qqq[0]
    iqleft = qqq[0] - 1.5 * iqr
    iqright = qqq[1] + 1.5 * iqr
    left = np.min([avg - sss, f.min(skipna=True)])
    right = np.max([avg + sss, f.max(skipna=True)])
    extra = abs(left - right) / 20
    # print([avg, sss, left, right, left - extra, right - extra])

    # mean and variance field
    ymean.plot.hist(ax=axs[0], bins=n_bins, yscale='log', color="goldenrod")
    axs[0].title.set_text('Original Mean ' + yvar.name)
    yvar.plot.hist(ax=axs[1], bins=n_bins, yscale='log')
    axs[1].title.set_text('Original variance ' + yvar.name)

    # log10 plots
    f.plot.hist(ax=axs[2], bins=n_bins, yscale='log', xlim=[left - extra, right + extra])
    axs[2].title.set_text('Original variance log10 ' + yvar.name)
    g.plot.hist(ax=axs[3], bins=n_bins, yscale='log', color='red', xlim=[left - extra, right + extra])
    axs[3].title.set_text('Filtered variance log10 ' + yvar.name)
    for k in [2, 3]:
        axs[k].axvline(avg, color='k', linewidth=1)
        axs[k].axvline(avg - sss, color='k', linestyle='dashed', linewidth=1)
        axs[k].axvline(avg + sss, color='k', linestyle='dashed', linewidth=1)
        axs[k].axvline(iqleft, color='r', linestyle='dashed', linewidth=1)
        axs[k].axvline(iqright, color='r', linestyle='dashed', linewidth=1)

    fig.savefig(figname)


# get domain of the variable from the fraction of NaN: UNDER TESTING
def mask_from_field(xfield):
    """get the domain to be passed to the climatology .yml file from the number of
    missing point. Special treatment for sea ice. Use with caution."""
    ratio = float(xfield.count() / np.prod(np.array(xfield.shape)))
    loggy.info(ratio)
    if ratio < 0.2:  # this is a special case for ice, need to be double checked
        mask = 'ocean'
    elif 0.2 < ratio < 0.3:
        mask = 'land'
    elif 0.55 < ratio < 0.7:
        mask = 'ocean'
    elif ratio > 0.95:
        mask = 'global'
    else:
        mask = 'undefined'
        raise ValueError('ERROR: cant recognize mask')

    loggy.debug(mask)
    return mask
