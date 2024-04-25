"""
Create a new dataset from the original dataset of 100 surgeries,
but adress the issue of multiple sessions within the same scope
session by keeping only the time window for which we have enough
consecutive BIS values.
"""

from doa_zero_eeg.preprocessing.multiple_sessions import keep_longest_streak_ix
from doa_zero_eeg.utils import utils

data_path = utils.RAW_DATA_DIR

for parquet_file in data_path.rglob('*.parquet'):
    keep_longest_streak_ix(parquet_file)
