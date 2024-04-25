"""
Plot raw and smooth signals on the same plot in separate folders.
The session_id can be found on the file name of the .png plot.

Author: {Sofiane Sifaoui}
"""

from doa_zero_eeg.viz import bis_and_signals
from doa_zero_eeg.utils import utils

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


dir = Path("~/data/NOEEG/plots/").expanduser()
dir.mkdir(parents=True, exist_ok=True)

unfiltered_raw_plots_dir = dir.joinpath("raw_signals/").expanduser()
unfiltered_raw_plots_dir.mkdir(parents=True, exist_ok=True)

raw_plots_dir = dir.joinpath("filtered_signals/").expanduser()
raw_plots_dir.mkdir(parents=True, exist_ok=True)

smooth_plots_dir = dir.joinpath("smooth_signals/").expanduser()
smooth_plots_dir.mkdir(parents=True, exist_ok=True)

indiv_signals_dir = dir.joinpath("indiv_signals/").expanduser()
indiv_signals_dir.mkdir(parents=True, exist_ok=True)


if any(raw_plots_dir.iterdir()) is False and any(smooth_plots_dir.iterdir()) is False:
    for parquet_file in utils.FILTERED_DATA_DIR.rglob('*.parquet'):
        d = pd.read_parquet(parquet_file)

        bis_and_signals.plot_raw_signals(d)
        outdir = dir.joinpath(raw_plots_dir)
        plt.savefig(outdir.joinpath(parquet_file.name).with_suffix('.png'))
        plt.close()

        bis_and_signals.plot_smooth_signals(d)
        outdir = dir.joinpath(smooth_plots_dir)
        plt.savefig(outdir.joinpath(parquet_file.name).with_suffix('.png'))
        plt.close()


if any(unfiltered_raw_plots_dir.iterdir()) is False:
    for parquet_file in utils.RAW_DATA_DIR.rglob('*.parquet'):
        d = pd.read_parquet(parquet_file)

        bis_and_signals.plot_raw_signals(d)
        outdir = dir.joinpath(unfiltered_raw_plots_dir)
        plt.savefig(outdir.joinpath(parquet_file.name).with_suffix('.png'))
        plt.close()


if any(indiv_signals_dir.iterdir()) is False:
    for parquet_file in utils.FILTERED_DATA_DIR.rglob('*.parquet'):
        d = pd.read_parquet(parquet_file)
        for column in d.columns:
            bis_and_signals.plot_individual_signals(d, column)
            outdir = dir.joinpath(indiv_signals_dir).joinpath(column)
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir.joinpath(parquet_file.name).with_suffix('.png'))
            plt.close()