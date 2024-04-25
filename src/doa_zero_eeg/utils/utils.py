import random
from typing import Iterable
from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np
import torch

RAW_DATA_DIR = Path("~/data/NOEEG/doa-zero-eeg-sample/").expanduser() # raw data (original dataset)
FILTERED_DATA_DIR = Path("~/data/NOEEG/doa-zero-eeg-sample-filtered/").expanduser() # filtered data to one surgery per file


def set_seed(seed: int, cuda: bool, cudnn_benchmark=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        if isinstance(cudnn_benchmark, bool):
            torch.backends.cudnn.benchmark = cudnn_benchmark
        elif cudnn_benchmark is None:
            if torch.backends.cudnn.benchmark:
                logger.warning(
                    (
                        "torch.backends.cudnn.benchmark was set to True which may"
                        " results in lack of reproducibility. In some cases to ensure"
                        " reproducibility you may need to set"
                        " torch.backends.cudnn.benchmark to False."
                    ),
                    UserWarning,
                )
        else:
            raise ValueError(
                f"cudnn_benchmark expected to be bool or None, got '{cudnn_benchmark}'"
            )
        torch.cuda.manual_seed_all(seed)


def setup_pytorch(random_seed: int | None = None) -> str | int:
    torch.multiprocessing.set_sharing_strategy("file_system")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        torch.backends.cudnn.benchmark = True
        logger.info(f"using cuda device {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"using {device} device")
    else:
        device = "cpu"
        logger.warning("no GPU available")
    if random_seed is not None:
        set_seed(seed=random_seed, cuda=device == "cuda")
    # torch.set_default_device(f"{device}:0")
    return device


def compute_mae(df: pd.DataFrame)-> np.float64:
    """
    Given a pandas DataFrame with columns 'BIS' (true BIS values),
    'BIS_preds' (predicted BIS values) and 'rec_id', compute the MAE of 
    the regression model (average across number of BIS values for the specific
    recording and average for the total number of recording).
    """
    assert "BIS" in df.columns and "BIS_preds" in df.columns and "rec_id" in df.columns, "uncorrect column name"
    df['abs_error'] = (df['BIS'] - df['BIS_preds']).abs()
    grouped_mae = df.groupby('rec_id')['abs_error'].mean() # average across number of BIS values within the recording
    overall_mae = grouped_mae.mean() # average across all recordings
    return overall_mae


def concatenate_surgeries(path_list: Iterable[Path], data_dir: Path) -> pd.DataFrame:
    """
    Given an iterable of paths and the directory where data are stored, the function
    creates a DataFrame concatenating all surgeries from the path_list, ready
    to be used for training a regression model.

    data_dir: directory under where the data are stored
    """
    to_concat = list()
    for rec_id, path in enumerate(path_list):
        d = pd.read_parquet(data_dir.joinpath(path))
        d = d[["BIS", "CO₂fe", "CO₂mi", "FC", "PNIm", "PNIs", "PNId", "SpO₂"]]
        for c in ['PNIm', 'PNIs', 'PNId']:
            d[c] = d[c].ffill()
        d['rec_id'] = rec_id
        d['scope_session'] = path.name
        to_concat.append(d)
    df = pd.concat(to_concat, axis=0)
    df.dropna(axis=0, inplace=True)

    return df


def create_lagged_features(df: pd.DataFrame, feature_names: list, k: int) -> pd.DataFrame:
    """
    Create a DataFrame containing lagged features from k previous measurements.
    """
    to_concat = list()
    for feature in feature_names:
        for i in range(1, k + 1):
            # Create the lagged feature 
            lagged_feature_train = df.groupby('rec_id')[feature].shift(i).rename(f'{feature}_lag{i}')
            to_concat.append(lagged_feature_train)
    df_lagged = pd.concat(to_concat, axis=1)
    df_lagged = pd.concat([df, df_lagged], axis=1)
    df_lagged.dropna(inplace=True) # Dropping any rows with NaN values that were created due to lagging

    return df_lagged


def compute_time_delta(df: pd.DataFrame):
    """
    Given a pandas DataFrame of signals measurement indexing
    by timestamp, return the time_delta between two consecutive
    measurements.
    """
    df['TimeStamp'] = pd.to_datetime(df.index)
    df['time_delta'] = df['TimeStamp'].diff().dt.total_seconds().fillna(0)
    df.drop("TimeStamp", axis=1, inplace=True)
