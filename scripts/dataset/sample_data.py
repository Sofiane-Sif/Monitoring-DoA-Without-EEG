import os
import shutil
from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path("/data/alphabrain/doa-zero-eeg")


def compute_stats_worker(path: Path) -> dict:
    size_bytes = os.path.getsize(path)
    d = pd.read_parquet(path)
    n_doa_vals = len(d["BIS"].dropna())
    return dict(
        pid=path.stem,
        size_bytes=size_bytes,
        n_doa_vals=n_doa_vals,
    )


def compute_stats():
    paths = DATA_DIR.glob("*.parquet")
    stats = pd.DataFrame(
        joblib.Parallel(n_jobs=12)(
            joblib.delayed(compute_stats_worker)(path) for path in paths
        )
    )
    stats.to_parquet("_stats.parquet")


def sample_data():
    if not Path("_stats.parquet").is_file():
        compute_stats()
    stats = pd.read_parquet("_stats.parquet")
    stats = stats.loc[(stats.n_doa_vals > 3600) & (stats.n_doa_vals < (3600 * 3))]
    out_dir = DATA_DIR.parent / "doa-zero-eeg-sample"
    out_dir.mkdir()
    stats = stats.sample(100, random_state=42)
    for pid in stats.pid:
        src = DATA_DIR / (pid + ".parquet")
        dst = out_dir / (pid + ".parquet")
        shutil.copy(src, dst)


sample_data()
