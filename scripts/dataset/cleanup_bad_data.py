from loguru import logger
from pathlib import Path

import pandas as pd
import pyarrow.parquet
import joblib

DATA_DIR = Path("/data/alphabrain/doa-zero-eeg")

REQUIRED_LABELS = {"BIS", "CO₂fe", "CO₂mi", "FC", "PNId", "PNIm", "PNIs", "SpO₂"}


def worker(path: Path):
    found_labels = set(pyarrow.parquet.read_schema(path).names)
    # if not all labels included, then remove the data
    if not REQUIRED_LABELS < found_labels:
        logger.info(f'removing {path}')
        path.unlink()
        return False
    else:
        return True

paths = list(DATA_DIR.glob("*.parquet"))

print(len(paths), 'paths to process')

results = joblib.Parallel(n_jobs=12)(joblib.delayed(worker)(path) for path in paths)

print(sum(results), 'paths ok out of', len(results))
