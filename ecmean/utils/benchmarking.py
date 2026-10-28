#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" "
Provide benchmarking running on a set of cores and set of years EC-Earth3 (or any other model)
data on a specific machine for both performance indices and global mean
produce a figure to be automatically placed within the documentation and used as benchmark
in different phases of the ECmean4 project evolution.
This is made to run on CNR-ISAC machina 'Wilma' only

Each run measures the execution time and the peak memory of ecmean and its workers (sampled
with psutil). Runs can be labelled (--label, by default the git branch) and the csv of several
runs can be plotted side by side with --compare, e.g. to benchmark two branches.
"""

import argparse
import os
import shutil
import subprocess
import threading
import timeit
from datetime import date
import pandas as pd
import psutil
import seaborn as sns
import matplotlib.pyplot as plt
from ecmean.performance_indices import performance_indices
from ecmean.global_mean import global_mean
from ecmean import __version__ as version

expname = "historical"
refclim = "EC23"
# models = ['EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-g3', 'TaiESM1', 'CanESM5', 'CNRM-CM6-1',
#              'MIROC6', 'MPI-ESM1-2-HR', 'AWI-CM-1-1-MR']
models = ["EC-Earth3"]
ensemble = "r1i1p1f1"
scripts = ["Global Mean", "Performance Indices"]

# number of processors and years on which looop
nprocs = [1, 2, 4, 6, 8]
nyears = [1, 2, 5, 10, 30]
# nprocs = [2, 4]
# nyears = [1, 10]

# benchmark directory
benchdir = "/home/paolo/work/figures/ECmean4/benchmark"

# number of repetition of each run
nrepeat = 5

# number of years and processors used for the respective figure
nyears_fixed = 10
nprocs_fixed = 4

# what is measured: column of the dataframe, axis label, upper limit of the axis (None for automatic)
# and seaborn palette (the bars get one shade per script or label)
metrics = {"time": ("Execution time (sec)", 300, "Blues_d"), "memory": ("Peak memory (GB)", None, "Oranges_d")}


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark ecmean global mean and performance indices on Wilma")
    parser.add_argument("-c", "--config", default="config_benchmark.yml", help="configuration file (default: config_benchmark.yml)")
    parser.add_argument("-l", "--loglevel", default="info", help="ecmean log level (default: info)")
    parser.add_argument(
        "--label", help="label of this run, used in the csv/figure names and in the comparison (default: git branch)"
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="CSV", help="do not run: plot these benchmark csv files side by side"
    )
    parser.add_argument("--quick", action="store_true", help="smoke test: one repetition, fixed cores and years only")
    parser.add_argument(
        "--no-definitive",
        dest="definitive",
        action="store_false",
        help="do not copy the figure into the docs (to be done once in a while, or before big merge)",
    )
    return parser.parse_args()


def git_branch():
    """Name of the current git branch, or the ecmean version if not available"""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f"v{version}"
    return branch if branch != "HEAD" else f"v{version}"


class MemorySampler:
    """Context manager sampling in a thread the memory of this process and of its children
    (the ecmean workers). The proportional set size is used, so that the pages shared between
    the forked workers are counted once; the peak is the maximum of the sum over the samples,
    above the memory of the idle benchmark process, in GB."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak = 0.0
        self.baseline = 0.0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _memory(self):
        """Memory (GB) of this process and its children, ignoring the ones ending meanwhile"""
        total = 0
        for proc in [self._process] + self._process.children(recursive=True):
            try:
                total += proc.memory_full_info().pss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return total / 1e9

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._memory() - self.baseline)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self._memory()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def run_benchmark(args, label):
    """Time global mean and performance indices on the cores/years grid, sampling the memory,
    and return a dataframe"""

    repeat = 1 if args.quick else nrepeat
    howmuch = pd.DataFrame(columns=["script", "time", "memory", "nprocs", "nyears", "label"])

    # loop on models: so far not used
    for model in models:
        # loop on the scripts and on the procs
        for script in scripts:
            for nproc in [nprocs_fixed] if args.quick else nprocs:
                if nproc == nprocs_fixed:
                    nn = [nyears_fixed] if args.quick else nyears
                else:
                    nn = [nyears_fixed]

                for nyear in nn:
                    print("Running " + script + " with nprocs =", str(nproc) + " for nyears = " + str(nyear))
                    print(model)

                    year1 = 1980
                    year2 = year1 + nyear - 1
                    # separated calls for model (perhaps a specific config file should be added)
                    if script == "Performance Indices":
                        run = lambda: performance_indices(
                            expname,
                            year1,
                            year2,
                            config=args.config,
                            model=model,
                            numproc=nproc,
                            climatology=refclim,
                            ensemble=ensemble,
                            loglevel=args.loglevel,
                        )
                    elif script == "Global Mean":
                        run = lambda: global_mean(
                            expname,
                            year1,
                            year2,
                            config=args.config,
                            model=model,
                            numproc=nproc,
                            ensemble=ensemble,
                            loglevel=args.loglevel,
                        )
                    else:
                        raise ValueError("Script not recognized")

                    with MemorySampler() as memory:
                        single = timeit.timeit(run, number=repeat)

                    # concatenate
                    d = pd.DataFrame(
                        {
                            "script": [script],
                            "time": [round(single / repeat, 1)],
                            "memory": [round(memory.peak, 2)],
                            "nprocs": [nproc],
                            "nyears": [nyear],
                            "label": [label],
                        }
                    )
                    print(d)
                    howmuch = pd.concat([howmuch, d])

    return howmuch


def load_benchmarks(csvfiles):
    """Load benchmark csv files; runs saved without label are labelled with their date,
    runs saved without memory get NaN"""
    runs = []
    for csvfile in csvfiles:
        run = pd.read_csv(csvfile, sep="\t", index_col=0)
        if "label" not in run.columns:
            run["label"] = os.path.basename(csvfile).replace("csv-benchmark-", "").replace(".csv", "")
        runs.append(run)
    return pd.concat(runs)


def plot_benchmark(howmuch, figurepath):
    """One row per metric and scaling: time and memory against cores (fixed years) and against
    years (fixed cores). A single run is plotted with one panel per row (bars by script); several
    runs (labels) get one panel per script, with bars by label."""

    model = models[0]
    labels = list(dict.fromkeys(howmuch["label"]))
    compare = len(labels) > 1
    ncols = len(scripts) if compare else 1

    scalings = [
        ("nprocs", "nyears", nyears_fixed, "Number of cores", f"{nyears_fixed} years"),
        ("nyears", "nprocs", nprocs_fixed, "Number of years", f"{nprocs_fixed} nprocs"),
    ]
    # metrics not measured (e.g. old csv without memory) are left out
    measured = [metric for metric in metrics if metric in howmuch and howmuch[metric].notna().any()]
    rows = [(metric, scaling) for metric in measured for scaling in scalings]
    fig, axs = plt.subplots(len(rows), ncols, tight_layout=True, figsize=(8 * ncols, 5 * len(rows)), squeeze=False)

    for row, (metric, (xcol, fixcol, fixval, xlabel, fixdesc)) in enumerate(rows):
        ylabel, ymax, palette = metrics[metric]
        what = ylabel.split(" (")[0].lower()
        data = howmuch[howmuch[fixcol] == fixval]
        for col in range(ncols):
            ax = axs[row, col]
            if compare:
                sub = data[data["script"] == scripts[col]]
                chart = sns.barplot(data=sub, x=xcol, y=metric, hue="label", hue_order=labels, palette=palette, ax=ax)
                ax.set_title(f"{scripts[col]} {what} for CMIP6 {model} ({fixdesc})", fontsize=14)
            else:
                chart = sns.barplot(data=data, x=xcol, y=metric, hue="script", palette=palette, ax=ax)
                ax.set_title(f"ECmean {version} {what} for CMIP6 {model} ({fixdesc})", fontsize=15)
            for i in chart.containers:
                chart.bar_label(i)
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set(ylim=(0, ymax))
            hh, ll = ax.get_legend_handles_labels()
            ax.legend(hh, ll)

    fig.savefig(figurepath)
    print("Figure saved in", figurepath)


if __name__ == "__main__":
    args = parse_args()
    today = str(date.today())
    os.makedirs(benchdir, exist_ok=True)

    if args.compare:
        howmuch = load_benchmarks(args.compare)
        print(howmuch)
        labels = "-vs-".join(dict.fromkeys(howmuch["label"]))
        plot_benchmark(howmuch, os.path.join(benchdir, f"benchmark-{labels}-{today}.png"))
    else:
        label = args.label or git_branch()
        suffix = f"-{args.label}" if args.label else ""
        suffix += "-quick" if args.quick else ""
        howmuch = run_benchmark(args, label)
        howmuch.to_csv(os.path.join(benchdir, f"csv-benchmark{suffix}-{today}.csv"), sep="\t")
        print(howmuch)
        figurepath = os.path.join(benchdir, f"benchmark{suffix}-{today}.png")
        plot_benchmark(howmuch, figurepath)
        if args.definitive and not args.quick:
            localdir = os.path.dirname(os.path.abspath(__file__))
            shutil.copy(figurepath, os.path.join(localdir, "../../docs/sphinx/source/_static/benchmark.png"))
