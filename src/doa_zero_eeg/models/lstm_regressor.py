"""
LSTM regressor for BIS prediction based on the 4 mandatory signals
in the operating room
"""

from typing import Tuple
from loguru import logger
import numpy as np
import pandas as pd

import torch
import torch.nn

from doa_zero_eeg.utils import utils


def create_sequences(
    df: pd.DataFrame, sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a pandas DataFrame containing measurements of 4 mandatory
    signals in the operating room (CO₂fe, FC, PNIm, SpO₂), group by 'rec_id'
    and create sequences of data for each recording, using past values up to
    the current timestamp t to predict the BIS value at that timestamp t.

    Parameters:
    - df: Input DataFrame with signal measurements, BIS values, and rec_id.
    - sequence_length: The number of past measurements to include in each sequence.

    Returns:
    - Sequences (np.ndarray): Array of sequences, each with shape (seq_length, n_features).
    - Labels (np.ndarray): Array of BIS values, each corresponding to a sequence.
    """
    sequences = list()
    labels = list()

    # Ensure 'time_delta' is computed and present in the DataFrame
    utils.compute_time_delta(df)
    assert "time_delta" in df.columns, "time_delta is not a column of the DataFrame"

    for rec_id, group in df.groupby('rec_id'):

        data_array = group[['CO₂fe', 'FC', 'PNId', 'PNIm', 'PNIs', 'SpO₂', 'time_delta']].to_numpy()
        labels_array = group['BIS'].to_numpy()
        df_rec_id = df[df["rec_id"] == rec_id]

        for i in range(sequence_length, len(group)):
            start_index = i - sequence_length
            end_index = i

            sequence = data_array[start_index:end_index]
            label = labels_array[end_index - 1]

            # Check that BIS labels are correct for each recording
            assert (
                label == df_rec_id.iloc[end_index - 1]["BIS"]
            ), "'label' and true BIS value don't match"

            sequences.append(sequence)
            labels.append(label)

    return np.array(sequences), np.array(labels)


def resample_surgeries(df: pd.DataFrame, sfreq: str = "5s") -> pd.DataFrame:
    """
    Given a DataFrame containing the measurements of several surgeries,
    the function resamples the DataFrame every 1s by grouping each recording
    by rec_id (unique id of the surgery).
    """

    resampled_dfs = list()

    for _, group in df.groupby("rec_id"):
        # Resample each recording and forward fill missing values
        resampled_group = group.resample(sfreq).ffill()
        resampled_dfs.append(resampled_group)

    resampled_df = pd.concat(resampled_dfs)
    # Dropping NaN values corresponding to the first measurement of each surgery
    resampled_df.dropna(inplace=True)
    resampled_df["rec_id"] = resampled_df["rec_id"].astype("int")

    return resampled_df


def create_sequences_resampled(
    df_resampled: pd.DataFrame, sequence_length_seconds: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a resampled pandas DataFrame containing measurements of signals
    at a uniform 1-second frequency, create sequences of data with the shape
    (batch_size, seq_length, n_features), using measurements from the k previous seconds
    to predict the BIS value at each timestamp. Ensures sequences are within the same recording.

    Parameters:
    - df_resampled: Input DataFrame, resampled to 1-second intervals, with column 'rec_id'.
    - sequence_length_seconds: The number of previous seconds to include in each sequence.

    Returns:
    - Sequences (np.ndarray): Array of sequences, each with shape (seq_length, n_features).
    - Labels (np.ndarray): Array of BIS values, each corresponding to a sequence.
    """
    sequences = list()
    labels = list()

    for rec_id, group in df_resampled.groupby("rec_id"):
        # Convert relevant columns to numpy array for efficiency
        data_array = group[["CO₂fe", "CO₂mi", "FC", "PNIm", "PNId", "PNIs", "SpO₂"]].to_numpy()
        labels_array = group["BIS"].to_numpy()

        df_rec_id = df_resampled[df_resampled["rec_id"] == rec_id]

        # Create sequences for the current recording: shifted of 15s each time
        for i in range(sequence_length_seconds, len(group)):
            start_index = i - sequence_length_seconds
            end_index = i

            sequence = data_array[
                start_index:end_index
            ]  # Sequence of k previous seconds
            label = labels_array[end_index - 1]  # Corresponding label

            # Check that BIS labels are correct for each recording
            assert (
                label == df_rec_id.iloc[end_index - 1]["BIS"]
            ), "'label' and true BIS value don't match"

            sequences.append(sequence)
            labels.append(label)

    return np.array(sequences), np.array(labels)


class BISPredictor(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dense_dim: int,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super(BISPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.dense = torch.nn.Linear(hidden_dim, dense_dim)
        self.fc = torch.nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dense(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)  # output of the last timestamp
        return out

