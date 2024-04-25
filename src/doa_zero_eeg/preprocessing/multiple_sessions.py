"""
This script manages multiple sessions (i.e. multiple surgical operations) within the same
scope session (i.e. within the same .parquet file).

It creates a new dataset (under ~/data/doa-zero-eeg-sample-filtered/ directory), of the same
number of .parquet file in the original dataset (i.e. 100), but each file is restricted to 
a time window for which we have the longest streak of consecutive BIS values.  

Author: Sofiane Sifaoui
"""

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from doa_zero_eeg.utils import utils


def longest_streak_ix(mask: np.ndarray | pd.Series) -> np.ndarray:
    """
    Identify the indices of the longest streak of `True` (or 1) values in a binary sequence.
    It returns a boolean array of the same length as `mask`,
    where the indices of the longest streak of `True` values are set to `True`, and all
    other indices are set to `False`. If there are no `True` values in the input, it returns
    a boolean array with all `False` values.

    In our case we want to identify the longest streak of consecutive BIS values.

    Parameters
    ----------
    mask : np.ndarray | pd.Series
        A 1D binary sequence of numpy ndarray or pandas Series type, consisting of `True`/`False`
        or 1/0 values, where the longest streak of `True` (or 1) values is to be found.

    Returns
    -------
    np.ndarray
        A boolean numpy ndarray of the same shape as `mask`, with `True` at the indices of the
        longest streak of `True` values in the input `mask`, and `False` elsewhere.

    Examples
    --------
    >>> mask = pd.Series([True, False, True, True, False, True])
    >>> longest_streak_ix(mask)
    array([False, False,  True,  True, False, False])

    Notes
    -----
    - If there are multiple streaks of the same maximum length, this function will only
      return the first longest streak found.
    - This function is designed to work with binary sequences. Non-binary sequences may
      lead to unexpected results.
    """
    # get start, stop index pairs for sequences of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False], mask, [False]))))[0].reshape(-1, 2)
    if len(idx_pairs) == 0:  # no streak, return an array of False
        return np.full_like(mask, fill_value=False, dtype=bool)
    # get the sequence lengths, whose argmax would give us the id of longest streak
    start_ix, stop_ix = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), :]
    streak_ix = np.full(len(mask), False, dtype=bool)
    streak_ix[start_ix:stop_ix] = True
    return streak_ix


def keep_longest_streak_ix(data_path: Path, outdir=utils.FILTERED_DATA_DIR):
    """
    Given a datapath of a .parquet file, create a new .parquet file
    containing only the time window restricted to the longuest streak 
    of consecutive BIS values.
    """
    outdir.mkdir(parents=True, exist_ok=True) # create folder if not created yet
    data = pd.read_parquet(data_path)
    mask = data.resample('15min')['BIS'].nunique() > 1  # at least one BIS value within a 15min time window
    if mask.sum() == 0:
        logger.warning(f'could not find any streak in {data_path}')
        return
    # getting the indices of the longest streak 
    ix = longest_streak_ix(mask)
    ts_start = mask[ix].index[0] 
    ts_end = mask[ix].index[-1]
    data_filtered = data.loc[ts_start:ts_end]
    data_filtered.to_parquet(outdir.joinpath(f"{data_path.name}"))
