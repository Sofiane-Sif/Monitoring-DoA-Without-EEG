"""
Training LSTM pipeline on bigpu
"""
from pathlib import Path
from loguru import logger
import joblib
from joblib import dump
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from doa_zero_eeg.utils import utils
from doa_zero_eeg.models import lstm_regressor
from doa_zero_eeg.preprocessing.multiple_sessions import keep_longest_streak_ix
from doa_zero_eeg.viz.plot_utils import plot_pred_hist, plot_sessions_preds, plot_learning_curves

from datetime import datetime

sns.set_style("whitegrid")

utils.set_seed(42, cuda=True)  # bigpu uses cuda
device = utils.setup_pytorch()

data_path = Path("/data/alphabrain/doa-zero-eeg")
clean_data_path = Path("/data/alphabrain/doa-zero-eeg-filtered")

datasets_path = Path("/data/alphabrain/datasets")
model_path = Path("/data/alphabrain/models")
fig_path = Path("/data/alphabrain/figs")

model_path.mkdir(parents=True, exist_ok=True)
fig_path.mkdir(parents=True, exist_ok=True)
datasets_path.mkdir(parents=True, exist_ok=True)

def load_datasets(datasets_path):
    logger.info("Loading Datasets")

    with open(Path.joinpath(datasets_path,'datasets.pkl'), 'rb') as f:
        result = pickle.load(f)

    def y_unscale(y_scaled):
        return y_scaled * (result["bis_scaler_max"] - result["bis_scaler_min"]) + result["bis_scaler_min"]

    result["y_unscale"] = y_unscale
    return result


def generate_datasets(seq_length_sec, datasets_path, save):
    logger.info("Generating Datasets")

    # Path under which the cleaned dataset is stored (i.e. the dataset containing only 1 surgery per file)
    data_path = clean_data_path.rglob("*.parquet")
    file_paths = [p for p in data_path]

    # 80 surgeries recording for training, 20 for testing
    path_train, path_tmp = train_test_split(
        file_paths, test_size=0.4, shuffle=True, random_state=42
    )

    path_val, path_test = train_test_split(
        path_tmp, test_size=0.5, shuffle=True, random_state=42
    )

    df_train = utils.concatenate_surgeries(path_train, clean_data_path)
    df_val = utils.concatenate_surgeries(path_val, clean_data_path)
    df_test = utils.concatenate_surgeries(path_test, clean_data_path)


    signals_to_scale = [
        "BIS",
        "CO₂fe",
        "CO₂mi",
        "FC",
        "PNIm",
        "PNIs",
        "PNId",
        "SpO₂",
    ]  # scaling all explanatory variables and target
    scaler = MinMaxScaler()

    scaler.fit(df_train[signals_to_scale])

    bis_scaler_min = scaler.data_min_[0]
    bis_scaler_max = scaler.data_max_[0]


    df_train_scaled = scaler.transform(df_train[signals_to_scale])
    df_val_scaled = scaler.transform(df_val[signals_to_scale])
    df_test_scaled = scaler.transform(df_test[signals_to_scale])
    # Replace original signal columns with scaled versions
    df_train[signals_to_scale] = df_train_scaled
    df_val[signals_to_scale] = df_val_scaled
    df_test[signals_to_scale] = df_test_scaled
    # Resampling everything with the same sampling freq (1 sec)
    df_train_resampled = lstm_regressor.resample_surgeries(df_train)
    df_val_resampled = lstm_regressor.resample_surgeries(df_val)
    df_test_resampled = lstm_regressor.resample_surgeries(df_test)

    logger.info("Creating resampled sequences of shape (n_samples, seq_length_sec, n_features)")
    # here n_features = 6, no time_delta here since everything has the same sampling freq
    X_train, y_train = lstm_regressor.create_sequences_resampled(
        df_train_resampled, seq_length_sec
    )
    X_val, y_val = lstm_regressor.create_sequences_resampled(
        df_val_resampled, seq_length_sec
    )
    X_test, y_test = lstm_regressor.create_sequences_resampled(
        df_test_resampled, seq_length_sec
    )


    ###############################################################################
    # Training the LSTM network
    ###############################################################################

    train_data = torch.tensor(X_train, dtype=torch.float, device=device)
    train_labels = torch.tensor(y_train, dtype=torch.float, device=device)
    val_data = torch.tensor(X_val, dtype=torch.float, device=device)
    val_labels = torch.tensor(y_val, dtype=torch.float, device=device)
    test_data = torch.tensor(X_test, dtype=torch.float, device=device)
    test_labels = torch.tensor(y_test, dtype=torch.float, device=device)

    batch_size = 8192  # can increase batch size if training on bigpu

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # num_workers=0,
        # shuffle=True,
        # pin_memory=True,
    )
    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        # num_workers=0,
        # shuffle=True,
        # pin_memory=True,
    )
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    result = {
        "train_loader":train_loader,
        "val_loader":val_loader,
        "test_loader":test_loader,
        "X_train":X_train,
        # "X_val":X_val,
        # "X_test":X_test,
        # "y_train":y_train,
        # "y_val":y_val,
        # "y_test":y_test,
        # "train_labels":train_labels,
        "test_labels":test_labels,
        # "val_labels":val_labels,
        "df_train":df_train,
        # "df_val":df_val,
        "df_test":df_test,
        "bis_scaler_min":bis_scaler_min,
        "bis_scaler_max":bis_scaler_max,
    }
    if (save):
        logger.info("Saving Datasets")
        with open(Path.joinpath(datasets_path,'datasets.pkl'), 'wb') as f:
            pickle.dump(result, f)

    def y_unscale(y_scaled):
        return y_scaled * (bis_scaler_max - bis_scaler_min) + bis_scaler_min

    result["y_unscale"] = y_unscale

    return result




def define_lstm_model(n_features, device):

    lstm_model = lstm_regressor.BISPredictor(
    input_dim=n_features,
    hidden_dim=1,  # need to experiment different hidden_dim
    dense_dim=32,
    output_dim=1,
    num_layers=1,
    dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), weight_decay=1e-5)


    # class WeightedL1Loss(torch.nn.Module):
    #     def forward(self, ypred, ytrue):
    #         weights = 0.5 - torch.abs(ypred - 0.5)
    #         loss = torch.abs(ytrue - ypred) * (1 + weights**2)
    #         return loss.mean()

    # loss = |(x - y)| * (1 + (0.5 - |(y - 0.5)|)**2)
    # loss = abs(x - y) * (1 + (0.5 - abs(y - 0.5))**2)


    class WeightedL1Loss(torch.nn.Module):
        def forward(self, ypred, ytrue):
            loss = torch.sqrt(torch.abs(ypred-ytrue))
            return loss.mean()

    criterion = torch.nn.L1Loss()  # MAE error
    # criterion = WeightedL1Loss()

    return lstm_model, optimizer, criterion


def train(lstm_model, train_loader, val_loader, y_unscale, optimizer, device, criterion, model_path, files_suffix):
    # Training loop
    logger.info(f"Start of the LSTM regressor training on device: {device}")
    train_losses = list()
    val_losses = list()

    num_epochs = 50
    for epoch in range(num_epochs):
        lstm_model.train()
        batch_losses = list()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            bis_preds = lstm_model(sequences)

            # loss = criterion(y_unscale(bis_preds.squeeze()), y_unscale(labels))
            loss = criterion(bis_preds, labels)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        print(f"[Epoch {epoch+1}], Loss: {np.mean(batch_losses)}")
        train_losses.append(np.mean(batch_losses))
        # Validation phase
        # TODO: add validation diagnostics, early stopping, checkpoints, etc.
        lstm_model.eval()
        batch_losses = list()
        with torch.no_grad():  # No need to track gradients for validation
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                bis_preds = lstm_model(sequences)

                # val_loss = criterion(y_unscale(bis_preds.squeeze()), y_unscale(labels))
                val_loss = criterion(bis_preds, labels)

                batch_losses.append(val_loss.item())
        print(f"[Epoch {epoch+1}], Validation Loss: {np.mean(batch_losses)}")
        val_losses.append(np.mean(batch_losses))

    logger.info("End of training")

    torch.save(lstm_model.state_dict(), model_path.joinpath(f"lstm_model_{files_suffix}.pth"))

    plot_learning_curves(y_unscale, train_losses, val_losses, fig_path, files_suffix)
    
    return lstm_model



def load_model(lstm_model, model_path, model_file_prefix):
    logger.info(f"Loading model")
    lstm_model.load_state_dict(torch.load(model_path.joinpath(f"lstm_model{model_file_prefix}.pth")))
    return lstm_model

def test(lstm_model, device, test_loader, criterion, test_labels):
    #TODO : do not use test_labels
    logger.info("Testing")

    lstm_model.eval()
    with torch.no_grad():

        test_losses = list()
    
        scaled_preds = torch.tensor([], device=device)

        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            batch_pred = lstm_model(sequences)

            scaled_preds = torch.cat((scaled_preds, batch_pred))

        test_loss = criterion(scaled_preds.flatten(), test_labels)

        print(f"Test Loss: {test_loss.item()}")
        

TRAIN = True
TEST = True
GENERATE_DATASETS = False
SEQ_LENGTH_SEC = 60 * 4  # take the last x minutes measurements to make the prediction


class SqrtLoss(torch.nn.Module):
    def forward(self, ypred, ytrue):
        loss = torch.sqrt(0.1+torch.abs(ypred-ytrue))
        return loss.mean()


class ValentinCorrectedLoss(torch.nn.Module):
    def forward(self, ypred, ytrue):
        weights = 0.5 - torch.abs(ypred - 0.5)
        loss = torch.abs(ytrue - ypred) * (1 + weights)
        return loss.mean()


# my_losses = [(SqrtLoss(), "sqrt_loss"), (torch.nn.L1Loss(), "MAE_loss"), (ValentinCorrectedLoss(), "Valentin_c_loss")]
my_losses = [(SqrtLoss(), "sqrt_loss")]

if (GENERATE_DATASETS):
    datasets_utils = generate_datasets(SEQ_LENGTH_SEC, datasets_path, True)
else :
    datasets_utils = load_datasets(datasets_path)

for curr_loss, loss_name in my_losses:

    lstm_model, optimizer, _  = define_lstm_model(datasets_utils["X_train"].shape[-1], device)
    criterion = curr_loss

    if TRAIN:
        lstm_model = train(lstm_model, datasets_utils["train_loader"], datasets_utils["val_loader"], datasets_utils["y_unscale"], optimizer, device, criterion, model_path, loss_name)
    else:
        lstm_model = load_model(lstm_model, model_path, "")

    if TEST:
        test(lstm_model, device, datasets_utils["test_loader"], criterion, datasets_utils["test_labels"])


    plot_pred_hist(lstm_model, device, datasets_utils["test_loader"], datasets_utils["test_labels"], datasets_utils["y_unscale"], criterion, SEQ_LENGTH_SEC, fig_path, loss_name)
    plot_sessions_preds(lstm_model, device, datasets_utils["df_test"], SEQ_LENGTH_SEC, fig_path, loss_name)
