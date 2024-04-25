"""
Train XGBoost regressor based on {Abhinav} script

No need to scale input features for XGBoost (https://github.com/dmlc/xgboost/issues/357)
from the XGBoost library main developer.
"""

from pathlib import Path
from joblib import dump, load
from loguru import logger
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from doa_zero_eeg.utils import utils

utils.set_seed(42, cuda=False)
# Must use the filtered dataset, i.e., the dataset restricted to only non NaN BIS values 
# (cf. preprocessing/multiple_sessions.py)
data_path = utils.FILTERED_DATA_DIR.rglob('*.parquet')

file_paths = [p for p in data_path]

# 80 surgeries recording for training/validation (Grid Search with cv on train set), 20 for testing
path_train, path_test = train_test_split(file_paths, test_size=0.2, random_state=42)

df_train = utils.concatenate_surgeries(path_train, utils.FILTERED_DATA_DIR)
df_test = utils.concatenate_surgeries(path_test, utils.FILTERED_DATA_DIR)

# include time difference between two meausurements into the input features vectors
utils.compute_time_delta(df_train)
utils.compute_time_delta(df_test)

n_prev_measurements = 50 # number of previous measurements to take into acocunt for the prediction
feature_names = ["CO₂fe", "FC", "PNIm", "SpO₂"]

# Create new lagged DataFrame for each feature
df_train_lagged = utils.create_lagged_features(df_train, feature_names, k=n_prev_measurements)
df_test_lagged = utils.create_lagged_features(df_test, feature_names, k=n_prev_measurements)

col = [col for col in df_train_lagged.columns if col != "BIS" and col != "rec_id" and col != "scope_session"]
X_train, y_train = np.array(df_train_lagged[col]), np.array(df_train_lagged["BIS"])
X_test, y_test = np.array(df_test_lagged[col]), np.array(df_test_lagged["BIS"])

param_grid = {
    'n_estimators': [100, 150],  # Number of trees
    'learning_rate': [0.01, 0.1],  # Step size shrinkage used to prevent overfitting
    'gamma': [0, 0.1, 0.2], # Minimum loss reduction required to make a further partition on a leaf node
    'max_depth': [3, 4, 5],  # Maximum depth of a tree
    'gamma': [0, 0.5, 1],  # Minimum loss reduction required to make a further partition on a leaf node
    'min_child_weight': [1, 3, 6],  # Minimum sum of instance weight (hessian) needed in a child
}

xgb_regressor = XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_regressor, 
    param_grid=param_grid, 
    cv=5, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1,
)

logger.info("Starting Grid Search for XGBoost Regressor")

t_start = time.time()
grid_search.fit(X_train, y_train)
t_end = time.time()

logger.info(f"Best parameters found: {grid_search.best_params_}")
logger.info(f"Best MAE score from Grid Search: {-grid_search.best_score_}")
logger.info(f"Grid Search time: {t_end - t_start} seconds")

best_model = grid_search.best_estimator_
dump(best_model, "best_xgb_regressor.joblib")

bis_preds = pd.DataFrame(best_model.predict(X_test), index=df_test_lagged.index, columns=["BIS_preds"])
bis_preds["BIS"] = y_test
bis_preds["rec_id"] = df_test_lagged["rec_id"] 

mae = utils.compute_mae(bis_preds) # MAE on unseen test set
r2 = r2_score(bis_preds["BIS"], bis_preds["BIS_preds"])

logger.info(f"Overall MAE for the XGBoost model: {mae:.2f}")
logger.info(f"R2 score for the XGBoost model: {r2:.2f}")
