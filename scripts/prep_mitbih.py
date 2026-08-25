"""Preprocess uploaded_datasets/mitbih/*.csv + *annotations.txt -> data/mitbih/.

Each record provides:
  * <rec>.csv         — columns: sample #, MLII (lead II), V5
  * <rec>annotations.txt — PhysioNet rdann text dump:
        Time     Sample #  Type  Sub  Chan  Num  Aux

Standard AAMI 5-class mapping (Kachuee et al. 2018):
  N (0): Normal      — N, L, R, e, j
  S (1): SVEB        — A, a, J, S
  V (2): VEB         — V, E
  F (3): Fusion      — F
  Q (4): Unknown     — /, f, Q

Each beat window: 90 samples before R-peak + 97 samples after = 187 samples on the
MLII (lead II) channel, normalised per-record to zero mean / unit std.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "uploaded_datasets", "mitbih")
OUT_DIR = os.path.join(ROOT, "data", "mitbih")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_BEFORE = 90
WINDOW_AFTER  = 97
WIN = WINDOW_BEFORE + WINDOW_AFTER     # 187

AAMI = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}


def parse_annotations(path: str) -> pd.DataFrame:
    """Whitespace-delimited, fixed-ish columns. Skip header. Use first 3 cols
    we care about (Time, Sample#, Type)."""
    rows = []
    with open(path, "r") as f:
        next(f)        # header
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                samp = int(parts[1])
            except ValueError:
                continue
            typ = parts[2]
            rows.append((samp, typ))
    return pd.DataFrame(rows, columns=["sample", "type"])


def beats_for_record(rec_path: str, ann_path: str, rec_id: int):
    sig = pd.read_csv(rec_path)
    # column names like 'MLII' or "'MLII'" depending on how saved
    cols = [c.strip("'") for c in sig.columns]
    sig.columns = cols
    if "MLII" in sig.columns:
        x = sig["MLII"].astype(np.float32).values
    else:
        # take second column as primary lead if MLII missing
        x = sig.iloc[:, 1].astype(np.float32).values
    # per-record z-score
    x = (x - x.mean()) / (x.std() + 1e-6)

    ann = parse_annotations(ann_path)
    Xs, ys, rs = [], [], []
    for _, row in ann.iterrows():
        if row["type"] not in AAMI:
            continue
        c = int(AAMI[row["type"]])
        s = int(row["sample"])
        a = s - WINDOW_BEFORE
        b = s + WINDOW_AFTER
        if a < 0 or b > len(x):
            continue
        Xs.append(x[a:b].astype(np.float32))
        ys.append(c)
        rs.append(rec_id)
    if not Xs:
        return None
    X = np.stack(Xs).reshape(-1, 1, WIN).astype(np.float32)
    return X, np.asarray(ys, dtype=np.int64), np.asarray(rs, dtype=np.int64)


def main() -> None:
    csvs = sorted(glob.glob(os.path.join(SRC_DIR, "*.csv")))
    print(f"[mitbih] found {len(csvs)} record csvs")
    Xs, ys, rs = [], [], []
    for csv_path in csvs:
        rec_str = os.path.splitext(os.path.basename(csv_path))[0]
        m = re.match(r"^(\d+)$", rec_str)
        if not m:
            continue
        rec_id = int(m.group(1))
        ann_path = os.path.join(SRC_DIR, f"{rec_id}annotations.txt")
        if not os.path.exists(ann_path):
            continue
        try:
            out = beats_for_record(csv_path, ann_path, rec_id)
        except Exception as e:
            print(f"  skipped {rec_id}: {e}")
            continue
        if out is None:
            continue
        X, y, r = out
        Xs.append(X); ys.append(y); rs.append(r)
        print(f"  rec {rec_id}: {len(y):>5d} beats")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    record_id = np.concatenate(rs, axis=0)

    # collapse rare classes (anything <500 instances) into class 4 to keep model output usable
    uniq, cnts = np.unique(y, return_counts=True)
    print("[mitbih] raw class hist:", dict(zip(uniq.tolist(), cnts.tolist())))

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "record_id.npy"), record_id)
    print(f"[mitbih] wrote {OUT_DIR}/X.npy shape={X.shape}")


if __name__ == "__main__":
    main()
