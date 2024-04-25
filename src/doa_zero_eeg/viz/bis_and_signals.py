"""
Some visualization on the dataset

- Raw signals
- Filtered signals (based on the filtered dataset)
- Smoothed signals (using Savgol-Filter)
- Individual signals

Author: {Sofiane Sifaoui}
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def plot_raw_signals(data: pd.DataFrame) -> plt.figure:
    """
    Plot raw signals given a DataFrame of measurement.
    The DataFrame should contains the correct column names.
    """    
    plt.figure(figsize=(16,6))
    
    plt.plot(data["BIS"], label="BIS", c="r")
    plt.plot(data["FC"], label="FC", c="g")
    plt.plot(data["SpO₂"], label="SpO₂", c="b") 
    plt.plot(data["CO₂fe"], label="CO₂fe", c="orange")
    plt.plot(data["PNIm"].ffill(), label="PNIm", c="magenta")

    plt.xlabel("Time")
    plt.ylabel("Time series values")
    plt.title("All labels and BIS v.s time")

    plt.tight_layout()
    plt.legend(loc="upper right")


def savgol_smooth(signal: pd.DataFrame, window_length: int = 175, polyorder: int = 3) -> pd.DataFrame:
    # The window_length needs to be a positive odd number
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(signal, window_length, polyorder)


def plot_smooth_signals(data:pd.DataFrame) -> plt.figure:
    """
    Plot smoothed signals given a DataFrame of measurement.
    Smoothing with Savitzky-Golay (savgol) filter.
    """
    plt.figure(figsize=(16,6))

    d = data.interpolate() 

    # Apply the savgol_smooth function to each signal
    plt.plot(d.index, savgol_smooth(d["BIS"]), label="BIS", c="r")
    plt.plot(d.index, savgol_smooth(d["FC"]), label="FC", c="g")
    plt.plot(d.index, savgol_smooth(d["SpO₂"]), label="SpO₂", c="b")
    plt.plot(d.index, savgol_smooth(d["CO₂fe"]), label="CO₂fe", c="orange")
    plt.plot(d.index, data["PNIm"].ffill(), label="PNIm", c="magenta")

    plt.xlabel("Time")
    plt.ylabel("Time series values")
    plt.title("Smooth labels and BIS v.s time")

    plt.tight_layout()
    plt.legend(loc="upper right")


def plot_individual_signals(data: pd.DataFrame, column_name: str) -> plt.figure:
    """
    Plot individual signals given a DataFrame of measurement and 
    the variable to plot.
    """
    plt.figure(figsize=(16,6))
    if column_name == "PNIm" or column_name == "PNId" or column_name == "PNIs":
        plt.plot(data[column_name].ffill(), label=f"{column_name}")
    else:
        plt.plot(data[column_name], label=f"{column_name}")
    plt.xlabel("Time")
    plt.ylabel(f"{column_name} values")
    plt.title(f"{column_name} v.s time")

    plt.tight_layout()
    plt.legend(loc="upper right")