"""
Script for plotting BIS predictions for a trained regression model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from joblib import load

X_train = pd.read_parquet("/tmp/X_train.parquet")
X_test = pd.read_parquet("/tmp/X_test.parquet")
y_train = pd.read_parquet("/tmp/y_train.parquet")
y_test = pd.read_parquet("/tmp/y_test.parquet")

model = load("/Users/Sofiane/Desktop/DoA-Zero-EEG/src/doa_zero_eeg/models/best_xgb_regressor.joblib")

tmp_plots = Path("/tmp/plots").expanduser()
tmp_plots.mkdir(parents=True, exist_ok=True)

bis_preds = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=["BIS_preds"])
bis_preds["BIS"] = y_test
bis_preds["rec_id"] = X_test["rec_id"]

rec_ids = np.array(bis_preds["rec_id"])
for rec_id in np.unique(rec_ids):
    true_bis = bis_preds[bis_preds["rec_id"]==rec_id]["BIS"]
    predicted_bis = bis_preds[bis_preds["rec_id"]==rec_id]["BIS_preds"]

    plt.figure(figsize=(16,8))
    plt.plot(true_bis, c="r", label="True BIS")
    plt.plot(predicted_bis, c="g", label="Predicted BIS")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("BIS value")
    plt.title("Predicted vs True BIS values for XGBoost")
    plt.tight_layout()
    plt.savefig(tmp_plots.joinpath(Path(f"rec_id_{rec_id}")).with_suffix('.png'))
