"""
Plotting xgboost BIS predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import load

output_path = Path("~/data/NOEEG/plots/xgboost_preds/train").expanduser()
df_train_lagged = pd.read_parquet("./df_train_lagged.parquet")
df_test_lagged = pd.read_parquet("./df_test_lagged.parquet")
xgboost_model = load("/Users/Sofiane/Desktop/DoA-Zero-EEG/src/doa_zero_eeg/models/best_xgb_regressor.joblib")

col = [col for col in df_test_lagged.columns if col != "BIS" and col != "rec_id" and col != "scope_session"]

for i in range(df_test_lagged["rec_id"].unique()):
    plt.figure(figsize=(12,8))
    df_rec_id = df_test_lagged[df_test_lagged["rec_id"]==i]
    scope_session = df_rec_id["scope_session"].values[0].split(".")[0]
    preds = xgboost_model.predict(df_rec_id[col])
    plt.plot(df_rec_id["BIS"], label="True BIS")
    plt.plot(df_rec_id.index, preds, label="Predicted BIS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.joinpath(f"{scope_session}.png"))

# for i in range(df_train_lagged["rec_id"].unique()[-1]):
#     plt.figure(figsize=(12,8))
#     df_rec_id = df_train_lagged[df_train_lagged["rec_id"]==i]
#     if df_rec_id.shape[0] == 0:
#         continue
#     scope_session = df_rec_id["scope_session"].values[0].split(".")[0]
#     preds = xgboost_model.predict(df_rec_id[col])
#     plt.plot(df_rec_id["BIS"], label="True BIS")
#     plt.plot(df_rec_id.index, preds, label="Predicted BIS")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path.joinpath(f"{scope_session}.png"))
#     plt.close()