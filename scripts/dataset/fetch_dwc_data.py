from pathlib import Path
import re

import dwclib
import pandas as pd
import joblib


def nbl_is_surgical_room(nbl: str | None) -> bool:
    if nbl is None:
        return False
    return nbl in {
        "bo10",
        "bo11",
        "bo12",
        "bo4",
        "bo5",
        "bo6",
        "bo7",
        "bo8",
        "bo9",
        "bot0",
        "mbo2",
        "mbo3",
        "obo0",
        "obo1",
        "obo2",
        "r5",
        "r6",
    }


def normalise_bed_label(x: str | None) -> str | None:
    if x is None:
        return None
    x = x.lower()
    x = x.replace(" ", "")
    x = x.replace("_", "")
    x = re.sub(r"0([1-9])", r"\1", x)
    return x


labels = ["BIS", "PNI", "FC", "SpO₂", "CO₂"]

p = dwclib.read_patients(numericlabels=labels)
p["duration"] = p.data_end - p.data_begin
p = p.loc[p.duration >= pd.Timedelta("1h")]
p["nbl"] = p.bedlabel.apply(normalise_bed_label)
p = p.loc[p.nbl.apply(nbl_is_surgical_room)]

# p = p.sample(100, random_state=42)

out_dir = Path("/data/alphabrain/doa_zero_eeg")
out_dir.mkdir(exist_ok=True)


def worker(pid: str, dtb: str, dte: str):
    data = dwclib.read_numerics(
        patientids=[pid],
        dtbegin=dtb,
        dtend=dte,
        labels=labels,
    )
    if len(data) == 0:
        print("ERROR")
    else:
        data.to_parquet(out_dir / (pid + ".parquet"))
        print(f"OK: {data.columns}")


p["data_begin"] = pd.to_datetime(p.data_begin, utc=True)
p["data_end"] = pd.to_datetime(p.data_end, utc=True)
p["dtb"] = p.data_begin.dt.floor("1d") - pd.Timedelta(days=1)
p["dte"] = p.data_begin.dt.ceil("1d") + pd.Timedelta(days=1)

to_run_params = list()
for pid, r in p.iterrows():
    dtb = r.dtb.strftime("%Y-%m-%d")
    dte = r.dte.strftime("%Y-%m-%d")
    to_run_params.append((pid, dtb, dte))

joblib.Parallel(n_jobs=12)(joblib.delayed(worker)(*params) for params in to_run_params)
