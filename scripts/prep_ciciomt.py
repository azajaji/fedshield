"""Preprocess uploaded_datasets/ciciomt_2024 CSVs -> data/ciciomt/{X,y,protocol}.npy.

* Folds 51 raw labels into 6 high-level classes consistent with the IoMTMLP head.
* Stratified subsampling to N_PER_CLASS rows per class to keep training tractable.
* Standardises features (zero mean / unit std) on the train split and applies the
  same transform to test.
* Derives ``protocol`` from the column ``Protocol Type`` rounded to int — this is
  what we partition clients on in the federated split.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "uploaded_datasets", "ciciomt_2024")
OUT_DIR = os.path.join(ROOT, "data", "ciciomt")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(SRC_DIR, "CIC_IoMT_2024_WiFi_MQTT_train.csv")
TEST_CSV  = os.path.join(SRC_DIR, "CIC_IoMT_2024_WiFi_MQTT_test.csv")

N_PER_CLASS = 25000        # 25k × 6 classes ≈ 150k rows; tractable + balanced
CHUNKSIZE   = 400_000
SEED        = 42


# six high-level classes -> matches num_classes=6 in the existing IoMTMLP
def fold_label(raw: str) -> int:
    s = raw.replace("_train", "").replace("_test", "")
    if s == "Benign":
        return 0
    if s.startswith("TCP_IP-DDoS"):
        return 1
    if s.startswith("TCP_IP-DoS"):
        return 2
    if s.startswith("MQTT"):
        return 3
    if "ICMP" in s and "DDoS" in s:    # already covered above; keep for safety
        return 1
    if s.startswith("Recon") or s.startswith("ARP"):
        return 4
    return 5                            # other


CLASS_NAMES = ["Benign", "TCP-DDoS", "TCP-DoS", "MQTT", "Recon/ARP", "Other"]


def stratified_load(csv_path: str, n_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stream the CSV, take up to ``n_per_class`` rows per high-level class."""
    bins = {c: [] for c in range(len(CLASS_NAMES))}
    cap_total = n_per_class * len(CLASS_NAMES)
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
        chunk = chunk.dropna()
        chunk["__y"] = chunk["label"].astype(str).map(fold_label)
        for c, sub in chunk.groupby("__y"):
            need = n_per_class - len(bins[c])
            if need <= 0:
                continue
            take = sub.sample(min(need, len(sub)), random_state=rng.integers(0, 2**31 - 1))
            bins[c].append(take)
        seen = sum(sum(len(d) for d in bins[c]) for c in bins)
        if all(sum(len(d) for d in bins[c]) >= n_per_class for c in bins):
            break
        if seen >= cap_total * 1.2:
            break
    parts = []
    for c, lst in bins.items():
        if lst:
            parts.append(pd.concat(lst, ignore_index=True))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("[prep] loading train ...")
    df_tr = stratified_load(TRAIN_CSV, N_PER_CLASS, rng)
    print("[prep] loading test ...")
    df_te = stratified_load(TEST_CSV, max(2000, N_PER_CLASS // 5), rng)

    # feature columns: numeric, drop label + intermediate folded label
    feat_cols = [c for c in df_tr.columns if c not in ("label", "__y")]
    Xtr = df_tr[feat_cols].astype(np.float32).values
    ytr = df_tr["__y"].astype(np.int64).values
    Xte = df_te[feat_cols].astype(np.float32).values
    yte = df_te["__y"].astype(np.int64).values

    # protocol (used for non-IID partition): integer-cast Protocol Type
    pt_idx = feat_cols.index("Protocol Type")
    proto_tr = np.clip(Xtr[:, pt_idx].astype(np.int32), 0, 50)
    proto_te = np.clip(Xte[:, pt_idx].astype(np.int32), 0, 50)

    # standardise features on train, apply to test
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    # truncate / pad to model input dim (32) — keep first 32 features
    Xtr = Xtr[:, :32]
    Xte = Xte[:, :32]

    # combine train+test (the FL splitter does its own holdout)
    X = np.concatenate([Xtr, Xte], axis=0).astype(np.float32)
    y = np.concatenate([ytr, yte], axis=0).astype(np.int64)
    protocol = np.concatenate([proto_tr, proto_te], axis=0).astype(np.int64)

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "protocol.npy"), protocol)
    print(f"[prep] wrote {OUT_DIR}/X.npy  shape={X.shape}")
    print(f"[prep] class hist:")
    uniq, cnts = np.unique(y, return_counts=True)
    for u, c in zip(uniq, cnts):
        print(f"   {u} {CLASS_NAMES[u]:<10} {c}")
    print(f"[prep] protocols: unique={len(np.unique(protocol))}")


if __name__ == "__main__":
    main()
