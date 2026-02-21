"""Class to provide a complete plotting solution for the ECMean package."""
import textwrap
import logging
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import  TwoSlopeNorm, ListedColormap #, LogNorm
import seaborn as sns
import numpy as np
from ecmean.libs.general import dict_to_dataframe, init_mydict
from ecmean.libs.climatology import SUPPORTED_CLIMATOLOGY, SUPPORTED_REFERENCE

 # OPTIMIZATION: Use non-interactive backend (much faster)
matplotlib.use('Agg')

loggy = logging.getLogger(__name__)

class ECPlotter:
    """Class to provide a complete plotting solution for the ECMean package.

    This class is designed to handle all plotting needs for the ECMean package.
    """

    def __init__(self, diagnostic,
                 modelname, expname, year1, year2,
                 regions=None, seasons=None):
        """Initialize the ECPlotter class.

        Args:
            diagnostic (str): Type of diagnostic to plot, either "performance_indices" or "global_mean".
            modelname (str): Name of the model.
            expname (str): Name of the experiment. 
            year1 (int): Start year of the data.
            year2 (int): End year of the data.
            regions (list, optional): List of regions to consider. Defaults to None, only for global mean.
            seasons (list, optional): List of seasons to consider. Defaults to None, only for global mean.
        """
        
        if diagnostic not in ["performance_indices", "global_mean"]:
            raise ValueError("Invalid diagnostic type. Choose 'performance_indices' or 'global_mean'.")
        self.diagnostic = diagnostic
        self.modelname = modelname
        self.expname = expname
        self.year1 = year1
        self.year2 = year2
        self.regions = regions
        self.seasons = seasons
        self.default_title = f"{diagnostic.replace('_', ' ').upper()} {self.modelname} {self.expname} {self.year1} {self.year2}"

    def _save_and_close(self, fig, filemap):
        """Helper function to save and close the figure."""
        loggy.info("Saving heatmap to %s", filemap)
        fig.savefig(filemap, bbox_inches="tight")
        plt.close(fig)
        

    def heatmap_plot(self, data, base, variables, filename=None, storefig=True,
                     climatology="EC23", addnan=False, title=None, reference="EC23"):
        """
        Prepare data for plotting performance indices or global mean.

        Args:
            data (dict or str): Dictionary with data to plot or path to a YAML file.
            base (dict or str): Dictionary with base benchmark data or path to a YAML file.
            variables (list): List of variable short names to plot.
            filename (str, optional): Path to save the plot. Defaults to None, it would be derived automatically.
            storefig (bool, optional): Whether to save the figure. Defaults to True.
            climatology (str, optional): Type of PI climatology, either any of the supported ones. Defaults to "EC23".
            addnan (bool, optional): Whether to add NaN values in the final plots. Defaults to False, only for global mean.
            title (str, optional): Title of the plot, overrides default title. Defaults to None.
            reference (str, optional): Name of the GM reference (e.g., "EC23"). Defaults to None, only for global mean.

        Returns:
            fig: The generated matplotlib figure object, if requested.
        """
        title = title if title is not None else self.default_title
        climatology = climatology if climatology is not None else "EC23"

        if climatology not in SUPPORTED_CLIMATOLOGY:
            raise ValueError(f"Invalid climatology type. Choose from {SUPPORTED_CLIMATOLOGY}.")
        if reference is not None and reference not in SUPPORTED_REFERENCE:
            raise ValueError(f"Invalid reference type. Choose from {SUPPORTED_REFERENCE}.")
        loggy.debug("Data is: %s", data)
        if isinstance(data, str):
            data = yaml.safe_load(data)
        if isinstance(base, str):
            base = yaml.safe_load(base)
        if self.diagnostic == "performance_indices":
            data2plot, cmip6, longnames = self.prepare_clim_dictionaries_pi(data, base, variables)
            fig = self.heatmap_comparison_pi(
                data_dict=data2plot, cmip6_dict=cmip6,
                longnames=longnames, filemap=filename,
                storefig=storefig, title=title, climatology=climatology)
        elif self.diagnostic == "global_mean":
            obsmean, obsstd, data2plot, units_list = self.prepare_clim_dictionaries_gm(data, base,
                                                                        variables, self.seasons, self.regions)
            fig = self.heatmap_comparison_gm(
                data_dict=data2plot, mean_dict=obsmean, std_dict=obsstd,
                units_list=units_list, storefig=storefig,
                filemap=filename, addnan=addnan, title=title, reference=reference)
        else:
            loggy.error("Invalid diagnostic type %s. Choose 'performance_indices' or 'global_mean'.", self.diagnostic)
            raise ValueError(f"Invalid diagnostic type {self.diagnostic}. Choose 'performance_indices' or 'global_mean'.")
        return fig


    def heatmap_comparison_pi(self, data_dict, cmip6_dict,
            longnames, storefig=True, filemap=None, size_model=14,
            title=None, climatology=None
        ):
        """
        Function to produce a heatmap - seaborn based - for Performance Indices
        based on CMIP6 ratio

        Args:
            data_dict (dict): dictionary of absolute performance indices
            cmip6_dict (dict): dictionary of CMIP6 performance indices
            longnames (list): list of long names for variables
            filemap (str, optional): path to save the plot. Defaults to None.
            storefig (bool, optional): Whether to save the figure. Defaults to True.
            size_model (int, optional): size of the PIs in the plot. Defaults to 14.
            title (str, optional): title of the plot, overrides default title. Defaults to None.
            climatology (str, optional): climatology used, either "EC23" or "EC24". Defaults to None.
        """
        # convert output dictionary to pandas dataframe
        data_table = dict_to_dataframe(data_dict)
        loggy.debug("Data table")
        loggy.debug(data_table)

        # relative pi with re-ordering of rows
        cmip6_table = dict_to_dataframe(cmip6_dict).reindex(longnames)
        relative_table = data_table.div(cmip6_table)

        # compute the total PI mean
        relative_table.loc['Total PI'] = relative_table.mean()

        # reordering columns if info is available
        lll = [(x, y) for x in self.seasons for y in self.regions]
        relative_table = relative_table[lll]
        loggy.debug("Relative table")
        loggy.debug(relative_table)

        # defining plot size
        myfield = relative_table
        xfig = len(myfield.columns)
        yfig = len(myfield.index)

        # real plot
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(xfig + 5, yfig + 2))

        # set title
        title = title if title is not None else self.default_title

        # set climatology for cbar_label
        climatology = climatology if climatology is not None else "EC23"

        thr = [0, 1, 5]
        tictoc = [0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5]

        tot = len(myfield.columns)
        # Extract the region (second element) from each column tuple
        sss = len({region for _, region in myfield.columns})
        divnorm = TwoSlopeNorm(vmin=thr[0], vcenter=thr[1], vmax=thr[2])
        pal = sns.color_palette("Spectral_r", as_cmap=True)
        chart = sns.heatmap(myfield, norm=divnorm, cmap=pal,
                            cbar_kws={"ticks": tictoc, 
                            'label': f"CMIP6 RELATIVE PI for {climatology} climatology"},
                            ax=axs, annot=True, linewidth=0.5, fmt='.2f',
                            annot_kws={'fontsize': size_model, 'fontweight': 'bold'})

        chart = chart.set_facecolor('whitesmoke')
        axs.set_title(title, fontsize=25)
        axs.vlines(list(range(sss, tot + sss, sss)), ymin=-1, ymax=len(myfield.index), colors='k')
        axs.hlines(len(myfield.index) - 1, xmin=-1, xmax=len(myfield.columns), colors='purple', lw=2)
        names = [' '.join(x) for x in myfield.columns]
        axs.set_xticks([x + .5 for x in range(len(names))], names, rotation=45, ha='right', fontsize=16)
        axs.set_yticks([x + .5 for x in range(len(myfield.index))], myfield.index, rotation=0, fontsize=16)
        axs.figure.axes[-1].tick_params(labelsize=15)
        axs.figure.axes[-1].yaxis.label.set_size(15)
        axs.set(xlabel=None)

        if filemap is None:
            filemap = 'PI4_heatmap.pdf'

        # save and close
        if storefig:
            self._save_and_close(fig, filemap)
        return fig

    def heatmap_comparison_gm(self, data_dict, mean_dict, std_dict, units_list, filemap=None,
                              addnan=True, storefig=True, size_model=14, size_obs=8, title=None, reference=None):
        """
        Function to produce a heatmap - seaborn based - for Global Mean
        based on season-averaged standard deviation ratio

        Args:
            data_dict (dict): table of model data
            mean_dict (dict): table of observations
            std_dict (dict): table of standard deviation
            units_list (list): list of units
            filemap (str, optional): path to save the plot. Defaults to None.
            addnan (bool, optional): add to the final plots also fields which cannot be compared against observations. Defaults to True.
            storefig (bool, optional): Whether to save the figure. Defaults to True.
            size_model (int, optional): size of the model values in the plot. Defaults to 14.
            size_obs (int, optional): size of the observation values in the plot. Defaults to 8.
            title (str, optional): title of the plot, overrides default title. Defaults to None.
            reference (str, optional): name of the GM reference climatology (e.g., "EC23"). Defaults to None.
        """
        # convert the three dictionary to pandas and then add units
        data_table = dict_to_dataframe(data_dict)
        mean_table = dict_to_dataframe(mean_dict)
        std_table = dict_to_dataframe(std_dict)
        for table in [data_table, mean_table, std_table]:
            table.index = table.index + ' [' + units_list + ']'

        loggy.debug("Data table")
        loggy.debug(data_table)

        # define array
        ratio = (data_table - mean_table) / std_table
        if addnan:
            mask = data_table[('ALL', 'Global')].notna()
        else:
            mask = ratio[('ALL', 'Global')].notna()
        clean = ratio[mask]

        # for dimension of plots
        xfig = len(clean.columns)
        yfig = len(clean.index)
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(xfig + 5, yfig + 2))
        
        # set title
        title = title if title is not None else self.default_title

        # set color range and palette
        thr = 10
        tictoc = range(-thr, thr + 1)
        pal = ListedColormap(sns.color_palette("vlag", n_colors=21))
        tot = len(clean.columns)
        sss = len({tup[1] for tup in clean.columns})

        # add reference if declared
        ref_str = f" against {reference} reference" if reference is not None else ""
        cbar_label = (f"Model Bias{ref_str}\n"
                      "(standard deviation of interannual variability from observations)")

        if addnan:
            empty_data = np.where(clean.isna(), 0, np.nan)
            empty_data = np.where(data_table[mask] == 0, np.nan, empty_data)
            empty_mean = np.where(clean.isna(), 0, np.nan)
            empty_mean = np.where(mean_table[mask].isna(), np.nan, empty_mean)

        # Original plotting method preserved, but optimized
        chart = sns.heatmap(clean, annot=data_table[mask], vmin=-thr - 0.5, vmax=thr + 0.5, center=0,
                            annot_kws={'va': 'bottom', 'fontsize': size_model},
                            cbar_kws={'ticks': tictoc, "shrink": .5, 'label': cbar_label},
                            fmt='.2f', cmap=pal)
        if addnan:
            chart = sns.heatmap(empty_data, annot=data_table[mask], fmt='.2f',
                                vmin=-thr - 0.5, vmax=thr + 0.5, center=0,
                                annot_kws={'va': 'bottom', 'fontsize': size_model, 'color': 'dimgrey'}, cbar=False,
                                cmap=ListedColormap(['none']))
        chart = sns.heatmap(clean, annot=mean_table[mask], vmin=-thr - 0.5, vmax=thr + 0.5, center=0,
                            annot_kws={'va': 'top', 'fontsize': size_obs, 'fontstyle': 'italic'},
                            fmt='.2f', cmap=pal, cbar=False)
        if addnan:
            chart = sns.heatmap(empty_mean, annot=mean_table[mask], vmin=-thr - 0.5, vmax=thr + 0.5, center=0,
                                annot_kws={'va': 'top', 'fontsize': size_obs, 'fontstyle': 'italic', 'color': 'dimgrey'},
                                fmt='.2f', cmap=ListedColormap(['none']), cbar=False)

        chart = chart.set_facecolor('whitesmoke')
        axs.set_title(title, fontsize=25)
        axs.vlines(list(range(sss, tot + sss, sss)), ymin=-1, ymax=len(clean.index), colors='k')
        names = [' '.join(x) for x in clean.columns]
        axs.set_xticks([x + .5 for x in range(len(names))], names, rotation=45, ha='right', fontsize=16)
        axs.set_yticks([x + .5 for x in range(len(clean.index))], clean.index, rotation=0, fontsize=16)
        axs.set_yticklabels(textwrap.fill(y.get_text(), 28) for y in axs.get_yticklabels())
        axs.figure.axes[-1].tick_params(labelsize=15)
        axs.figure.axes[-1].yaxis.label.set_size(15)
        axs.set(xlabel=None)

        if filemap is None:
            filemap = 'Global_Mean_Heatmap.pdf'

        # save and close
        if storefig:
            self._save_and_close(fig, filemap)
        return fig

    @staticmethod
    def prepare_clim_dictionaries_pi(data, clim, shortnames):
        """
        Prepare dictionaries for plotting
        Args:
            data: dictionary with data
            clim: dictionary with climatology
            shortnames: list of shortnames
        Returns:
            data2plot: dictionary with data for plotting
            cmip6: dictionary with CMIP6 data
            longnames: list of longnames
        """

        # uniform dictionaries
        filt_piclim = {}
        for k in clim.keys():
            filt_piclim[k] = clim[k]['cmip6']
            for f in ['models', 'year1', 'year2']:
                if f in filt_piclim[k]:
                    del filt_piclim[k][f]

        # set longname, reorganize the dictionaries
        data2plot = {clim[var]['longname']: data[var] for var in shortnames}
        cmip6 = {clim[var]['longname']: filt_piclim[var] for var in shortnames}
        longnames = [clim[var]['longname'] for var in shortnames]

        return data2plot, cmip6, longnames

    @staticmethod
    def prepare_clim_dictionaries_gm(data, clim, shortnames, seasons, regions):
        """
        Prepare dictionaries for global mean plotting

        Args:
            data: dictionary with the data
            clim: dictionary with the climatology
            shortnames: list of shortnames
            seasons: list of seasons
            regions: list of regions

        Returns:
            obsmean: dictionary with the mean
            obsstd: dictionary with the standard deviation
            data2plot: dictionary with the data to plot
            units_list: list of units
        """

        # loop on the variables to create obsmean and obsstd
        obsmean = {}
        obsstd = {}
        for var in shortnames:
            gamma = clim[var]
            obs = gamma['obs']

            # extract from yaml table for obs mean and standard deviation
            mmm = init_mydict(seasons, regions)
            sss = init_mydict(seasons, regions)
            # if we have all the obs/std available
            if isinstance(gamma['obs'], dict):
                for season in seasons:
                    for region in regions:
                        mmm[season][region] = obs[season][region]['mean']
                        sss[season][region] = obs[season][region]['std']
            # if only global observation is available
            else:
                mmm['ALL']['Global'] = gamma['obs']

            # Assign to obsmean and obsstd using longname as the key
            obsmean[gamma['longname']] = mmm
            obsstd[gamma['longname']] = sss

        # set longname, get units
        data2plot = {clim[var]['longname']: data[var] for var in shortnames}
        units_list = [clim[var]['units'] for var in shortnames]

        return obsmean, obsstd, data2plot, units_list

    # @staticmethod
    # def plot_xarray(data_dict: dict, filename: str, cmap: str = "viridis", log_scale: bool = False):
    #     """
    #     Plots multiple 2D xarray DataArrays from a dictionary and saves them as a multi-panel PDF.
        
    #     Parameters:
    #         data_dict (dict[str, xr.DataArray]): Dictionary of 2D data arrays.
    #         filename (str): Output PDF filename.
    #         cmap (str, optional): Colormap for the plots. Defaults to 'viridis'.
    #     """
    #     num_plots = len(data_dict)
    #     cols = int(np.ceil(np.sqrt(num_plots)))
    #     rows = int(np.ceil(num_plots / cols))
    #     vmin = 10**-4
    #     vmax = 10**4
        
    #     fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True)
        
    #     if num_plots == 1:
    #         axes = [axes]
    #     else:
    #         axes = axes.flatten()
        
    #     for ax, (name, data_array) in zip(axes, sorted(data_dict.items())):
    #         if data_array is not None:
    #             if data_array.ndim != 2:
    #                 raise ValueError(f"DataArray '{name}' must be 2D.")
                
    #             norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    #             im = ax.pcolormesh(data_array, cmap=cmap, norm=norm)
    #             ax.set_title(name)
    #             ax.set_xlabel(str(data_array.dims[1]))
    #             ax.set_ylabel(str(data_array.dims[0]))
    #             fig.colorbar(im, ax=ax)
        
    #     plt.savefig(filename, format="png", bbox_inches="tight")
    #     plt.close()

