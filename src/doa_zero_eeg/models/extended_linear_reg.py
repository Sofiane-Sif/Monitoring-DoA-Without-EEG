"""
Instead of taking into acount an instantaneous vector of measurement,
we will considerer, for each recording, a matrix of shape (k, 4), containing
the k previous measurements for this specific recording.
"""

from pathlib import Path
from joblib import dump, load
from loguru import logger
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from doa_zero_eeg.utils import utils

utils.set_seed(42)

# Must use the filtered dataset, i.e., the dataset restricted to only non NaN BIS values 
# (cf. preprocessing/multiple_sessions.py)
data_path = utils.FILTERED_DATA_DIR.rglob('*.parquet')

file_paths = [p for p in data_path]

# 80 surgeries recording for training, 20 for testing
path_train, path_test = train_test_split(file_paths, test_size=0.2, random_state=42)

df_train = utils.concatenate_surgeries(path_train, utils.FILTERED_DATA_DIR)
df_test = utils.concatenate_surgeries(path_test, utils.FILTERED_DATA_DIR)

t = 30  # For example, use the last 30 measurements
feature_names = ["CO₂fe", "FC", "PNIm", "SpO₂"]

# Create new lagged DataFrame for each feature
df_train_lagged = utils.create_lagged_features(df_train, feature_names, k=30)
df_test_lagged = utils.create_lagged_features(df_test, feature_names, k=30)

col = [col for col in df_train_lagged.columns if col != "BIS"]
X_train, y_train = df_train_lagged[col], df_train_lagged["BIS"]
X_test, y_test = df_test_lagged[col], df_test_lagged["BIS"]

logger.info("Start training the Extended Linear Regression baseline model")
t_start = time.time()

lr = LinearRegression()
lr.fit(X_train, y_train)
# dump(lr, "lr_baseline.joblib")

t_end = time.time()
logger.info("End of training")
logger.info(f"Training time: {t_end-t_start} seconds")

# Test phase and MAE calculation
bis_preds = pd.DataFrame(lr.predict(X_test), index=X_test.index, columns=["BIS_preds"])

bis_preds["BIS"] = y_test
bis_preds["rec_id"] = X_test["rec_id"]

mae = utils.compute_mae(bis_preds)

logger.info(f"Overall MAE for the Linear Regression model: {mae:.2f}") # MAE: 8.98

