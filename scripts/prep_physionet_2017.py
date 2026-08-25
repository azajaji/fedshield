"""Preprocess PhysioNet 2017 (CinC AF Challenge) into FedShield NPY format.

Output: data/physionet2017/X.npy (N, 1, T), y.npy (N,), record_id.npy (N,)
T is fixed at 9000 (30 s * 300 Hz). Shorter records are zero-padded; longer
records are truncated.

Class scheme (4 classes, mapped from REFERENCE-v3.csv):
    N -> 0 (Normal)
    A -> 1 (Atrial Fibrillation)
    O -> 2 (Other rhythm)
    ~ -> 3 (Noisy / unclassifiable)

Partition signal: record_id (every record is a unique recording session;
clients formed by random or sequential record-id grouping).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

LABEL_MAP = {"N": 0, "A": 1, "O": 2, "~": 3}
TARGET_LEN = 9000  # 30 s at 300 Hz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="./data/physionet_2017.zip")
    ap.add_argument("--out_root", default="./data/physionet2017")
    ap.add_argument("--max_records", type=int, default=None)
    args = ap.parse_args()

    try:
        import wfdb
    except ImportError:
        print("Need wfdb: pip install wfdb")
        return

    out = Path(args.out_root)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Step 1a: extract outer zip if not already done
    inner_zip_candidates = list(raw.rglob("training2017.zip"))
    if not inner_zip_candidates:
        print(f"Extracting outer {args.zip}...")
        with zipfile.ZipFile(args.zip, "r") as z:
            for name in z.namelist():
                z.extract(name, raw)
        inner_zip_candidates = list(raw.rglob("training2017.zip"))
        if not inner_zip_candidates:
            raise SystemExit(f"No training2017.zip found after extraction of {args.zip}")
    inner_zip = inner_zip_candidates[0]
    print(f"Inner zip: {inner_zip}")

    # Step 1b: extract inner training2017.zip into raw/
    if not list(raw.rglob("REFERENCE-v3.csv")) and not list(raw.rglob("REFERENCE.csv")):
        print(f"Extracting inner {inner_zip.name}...")
        with zipfile.ZipFile(inner_zip, "r") as z:
            names = z.namelist()
            for i, name in enumerate(names):
                z.extract(name, raw)
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1}/{len(names)}", flush=True)

    # Find the actual training2017 directory (now should exist after inner extract)
    train_dirs = [p for p in raw.rglob("training2017") if p.is_dir()]
    if not train_dirs:
        raise SystemExit(f"No training2017 directory found under {raw}")
    train_dir = train_dirs[0]
    print(f"Found training data at: {train_dir}")

    # Step 2: read REFERENCE-v3.csv (label table)
    ref_path = train_dir / "REFERENCE-v3.csv"
    if not ref_path.exists():
        ref_path = train_dir / "REFERENCE.csv"  # fallback
    print(f"Reading labels from: {ref_path.name}")
    labels = {}
    with open(ref_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[1] in LABEL_MAP:
                labels[parts[0]] = LABEL_MAP[parts[1]]
    print(f"  {len(labels)} labelled records")

    # Step 3: read each record via wfdb
    record_ids = sorted(labels.keys())
    if args.max_records:
        record_ids = record_ids[:args.max_records]

    Xs, ys, rids = [], [], []
    for i, rid in enumerate(record_ids):
        try:
            rec = wfdb.rdrecord(str(train_dir / rid))
            sig = rec.p_signal[:, 0].astype(np.float32)
        except Exception as e:
            continue
        # zero-pad / truncate to TARGET_LEN
        if len(sig) < TARGET_LEN:
            sig = np.concatenate([sig, np.zeros(TARGET_LEN - len(sig), dtype=np.float32)])
        else:
            sig = sig[:TARGET_LEN]
        Xs.append(sig.reshape(1, TARGET_LEN))
        ys.append(labels[rid])
        # encode record_id as integer (sortable, partitions cleanly)
        rids.append(int(rid[1:]))  # 'A00001' -> 1
        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{len(record_ids)}", flush=True)

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    rid_arr = np.asarray(rids, dtype=np.int64)

    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "record_id.npy", rid_arr)
    print(f"Wrote: {out / 'X.npy'} {X.shape}  y unique={np.unique(y).tolist()}")
    cls_counts = {LABEL_MAP_INV[c]: int((y == c).sum()) for c in np.unique(y)}
    print(f"  class distribution: {cls_counts}")


LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


if __name__ == "__main__":
    main()
