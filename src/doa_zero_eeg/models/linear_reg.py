"""
Train a Linear Regression model to predict each value of the BIS based
on the last observed values for our 4 signals (HF, SPO2, etCO2, PNIm).

This is the simplest model that we can train on our data: based on a 4-dimensional
features vectors, we want to predict the BIS value at each timestamp t. 

This model can be further extended by considering a matrix of features representing
the last k second of measurements to predict the BIS value. 

Author: {Sofiane Sifaoui}
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

X_train, y_train = df_train[["CO₂fe", "FC", "PNIm", "SpO₂"]], df_train["BIS"]
X_test, y_test = df_test[["CO₂fe", "FC", "PNIm", "SpO₂"]], df_test["BIS"]

y_test_df = y_test.reset_index()
df_test_reset = df_test.reset_index()
y_test_df = pd.merge(y_test_df, df_test_reset[['TimeStamp', 'rec_id']], on='TimeStamp', how='left')
y_test_df.set_index("TimeStamp", inplace=True)

logger.info("Start training the Linear Regression baseline model")
t_start = time.time()

lr = LinearRegression()
lr.fit(X_train, y_train)
# dump(lr, "lr_baseline.joblib")

t_end = time.time()
logger.info("End of training")
logger.info(f"Training time: {t_end-t_start} seconds")

# Test phase:

bis_preds = pd.DataFrame(lr.predict(X_test), index=X_test.index, columns=["BIS_preds"])
# Create a DataFrame containing true BIS, predicted BIS, rec_id, indexing by timestamp
y_test_df["BIS_preds"] = bis_preds["BIS_preds"]

# MAE computation: 
# For each BIS value observed within a recording, we sum the absolute errors between the
# true and predicted BIS score and divide by the number of BIS values for that specific recording.
# Then, we sum over all the recordings (and divide by the number of recordings)
# to have the overall MAE of this Linear Regresstion model.

mae = utils.compute_mae(y_test_df)

logger.info(f"Overall MAE for the Linear Regression model: {mae:.2f}") # MAE: 9.O8

