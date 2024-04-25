"""
Train a LSTM BIS predictor.

Author: {Sofiane Sifaoui}
"""

from pathlib import Path
from loguru import logger
from joblib import dump

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

from doa_zero_eeg.utils import utils
from doa_zero_eeg.models import lstm_regressor

utils.set_seed(42, cuda=False)
device = utils.setup_pytorch()

data_path = utils.FILTERED_DATA_DIR.rglob('*.parquet')
file_paths = [p for p in data_path]

# 80 surgeries recording for training, 20 for testing
path_train, path_tmp = train_test_split(
    file_paths, 
    test_size=0.4, 
    shuffle=True,
    random_state=42
)

path_val, path_test = train_test_split(
    path_tmp, 
    test_size=0.5, 
    shuffle=True,
    random_state=42
)

df_train = utils.concatenate_surgeries(path_train, utils.FILTERED_DATA_DIR)
df_val = utils.concatenate_surgeries(path_val, utils.FILTERED_DATA_DIR)
df_test = utils.concatenate_surgeries(path_test, utils.FILTERED_DATA_DIR)

signals_to_scale = ['BIS', 'CO₂fe', 'FC', 'PNId', 'PNIm', 'PNIs', 'SpO₂'] # scaling all explanatory variables and target
scaler = StandardScaler()
scaler.fit(df_train[signals_to_scale])

# store scaler mean and std for BIS to inverse scaling for MAE calculation during training
bis_scaler_mean = scaler.mean_[0]
bis_scaler_std = scaler.scale_[0]
dump(scaler, "lstm_standard_scaler.joblib")

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

logger.info("Create resampled sequences of shape (n_samples, seq_length_seconds, n_features)")
# n_features = 4, no time_delta here since everything has the same sampling freq
seq_length_sec = 60 * 4
X_train, y_train = lstm_regressor.create_sequences_resampled(df_train_resampled, seq_length_sec)
X_val, y_val = lstm_regressor.create_sequences_resampled(df_val_resampled, seq_length_sec)
X_test, y_test = lstm_regressor.create_sequences_resampled(df_test_resampled, seq_length_sec)

# logger.info("Create sequences of shape (n_samples, seq_length, n_features)")
# seq_length = 30
# X_train, y_train = lstm_regressor.create_sequences(df_train, sequence_length=seq_length)
# X_val, y_val = lstm_regressor.create_sequences(df_val, sequence_length=seq_length)
# X_test, y_test = lstm_regressor.create_sequences(df_test, sequence_length=seq_length)

train_data = torch.tensor(X_train, dtype=torch.float, device=device)
train_labels = torch.tensor(y_train, dtype=torch.float, device=device)

val_data = torch.tensor(X_val, dtype=torch.float, device=device)
val_labels = torch.tensor(y_val, dtype=torch.float, device=device)

test_data = torch.tensor(X_test, dtype=torch.float, device=device)
test_labels = torch.tensor(y_test, dtype=torch.float, device=device)

# Create data loaders: shuffle everything but think to correct order for the batches
# in order to make the LSTM model prediction in the "order" of the surgery.
batch_size = 64 
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

val_dataset = TensorDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

n_features = X_train.shape[-1] # n_features = 4 if resampling strategy, 5 if time_delta strategy
lstm_model = lstm_regressor.BISPredictor(
    input_dim=n_features, 
    hidden_dim=16, # need to experiment different hidden_dim
    output_dim=1,
    dense_dim=32,
    num_layers=2,
    dropout=0,
).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = torch.nn.L1Loss() # MAE error

# Training loop
logger.info(f"Start of the LSTM regressor training on device: {device}")
num_epochs = 15
train_losses = list()
val_losses = list()

for epoch in range(num_epochs):
    lstm_model.train()
    batch_losses = list()
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        bis_preds = lstm_model(sequences)

        # inverse scaling for prediction and true BIS values 
        labels_unscaled = labels * bis_scaler_std + bis_scaler_mean
        bis_preds_unscaled = bis_preds.squeeze() * bis_scaler_std + bis_scaler_mean
        
        loss = criterion(bis_preds_unscaled, labels_unscaled)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    print(f'[Epoch {epoch+1}], Loss: {np.mean(batch_losses)}')
    train_losses.append(np.mean(batch_losses))

    # Validation phase
    # TODO: add validation diagnostics, early stopping, checkpoints, etc. 
    lstm_model.eval()
    batch_losses = list()
    with torch.no_grad():  # No need to track gradients for validation
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            bis_preds = lstm_model(sequences)
            
            # inverse scaling for prediction and true BIS values 
            labels_unscaled = labels * bis_scaler_std + bis_scaler_mean
            bis_preds_unscaled = bis_preds.squeeze() * bis_scaler_std + bis_scaler_mean

            val_loss = criterion(bis_preds_unscaled, labels_unscaled)
            batch_losses.append(val_loss.item())
    print(f'[Epoch {epoch+1}], Validation Loss: {np.mean(batch_losses)}')
    val_losses.append(np.mean(batch_losses))

logger.info("End of training")
torch.save(lstm_model.state_dict(), 'lstm_model.pth')

# Plotting train and val MAE to detect overfitting
dir = Path("~/data/NOEEG/plots/").expanduser()
dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 8))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (MAE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(dir.joinpath(Path(f"lstm_training_validation_loss_{num_epochs}_epochs_w.png")))
plt.close()

# TODO: compute correct MAE on test set (at this point all the test set in put into memory...)
lstm_model.eval()  
with torch.no_grad(): 
    test_predictions = lstm_model(test_data)
    test_predictions_unscaled = test_predictions.squeeze() * bis_scaler_std + bis_scaler_mean
    test_labels_unscaled = test_labels * bis_scaler_std + bis_scaler_mean
    test_loss = criterion(test_predictions_unscaled, test_labels_unscaled)
    print(f'Test Loss: {test_loss.item()}')
