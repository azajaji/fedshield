"""Download + preprocess PTB-XL into the FEDShield data layout.

Output (matches data/mitbih/ format):
    data/ptbxl/X.npy            (N, 12, 1000)  float32  — 100 Hz, 12-lead, 10 s windows
    data/ptbxl/y.npy            (N,)           int64    — 5-class superdiagnostic label
    data/ptbxl/patient_id.npy   (N,)           int64    — for non-IID partitioning

Label scheme: 5-class superdiagnostic (NORM, MI, STTC, CD, HYP). Each PTB-XL
record carries multiple SCP codes; we take the dominant class by likelihood
(ties broken by the order NORM, MI, STTC, CD, HYP). Records with no
diagnostic-class code are discarded.

Usage:
    python -m scripts.prep_ptbxl
    python -m scripts.prep_ptbxl --max_records 1000   # quick smoke test
    python -m scripts.prep_ptbxl --sampling_rate 500  # high-res variant (7.6 GB)
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PHYSIONET_BASE = "https://physionet.org/files/ptb-xl/1.0.3"
SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_INT = {c: i for i, c in enumerate(SUPERCLASS_ORDER)}


def _fetch(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Resumable HTTP download via urllib (stdlib only)."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return  # cached
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:
            done = 0
            t0 = time.time()
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total and done % (chunk * 16) == 0:
                    pct = 100.0 * done / total
                    rate = done / max(1.0, time.time() - t0) / 1024.0 / 1024.0
                    print(f"\r  {dest.name}: {pct:5.1f}%  {rate:.1f} MB/s", end="", flush=True)
    tmp.rename(dest)
    print(f"\r  {dest.name}: done", flush=True)


def download_metadata(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[1/3] Downloading metadata...")
    db_path = out_root / "raw" / "ptbxl_database.csv"
    scp_path = out_root / "raw" / "scp_statements.csv"
    _fetch(f"{PHYSIONET_BASE}/ptbxl_database.csv", db_path)
    _fetch(f"{PHYSIONET_BASE}/scp_statements.csv", scp_path)
    db = pd.read_csv(db_path, index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(scp_path, index_col=0)
    return db, scp


def aggregate_diagnostic_class(scp_codes: dict, scp_table: pd.DataFrame) -> str | None:
    """Map an SCP-code dict to a single 5-class superdiagnostic label.

    Strategy: among codes that have a diagnostic_class entry in the catalog,
    pick the one with highest likelihood. Ties broken by SUPERCLASS_ORDER.
    """
    diag_rows = scp_table[scp_table.diagnostic_class.notna()]
    candidates: list[tuple[float, str]] = []
    for code, likelihood in scp_codes.items():
        if code in diag_rows.index:
            klass = diag_rows.loc[code, "diagnostic_class"]
            if klass in SUPERCLASS_TO_INT:
                candidates.append((float(likelihood or 0.0), klass))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], SUPERCLASS_TO_INT[x[1]]))
    return candidates[0][1]


def download_records(out_root: Path, db: pd.DataFrame, sampling_rate: int = 100,
                     max_records: int | None = None) -> list[Path]:
    print(f"[2/3] Downloading records at {sampling_rate} Hz...")
    suffix = "lr" if sampling_rate == 100 else "hr"
    rate_dir = f"records{sampling_rate}"
    raw_root = out_root / "raw" / rate_dir
    raw_root.mkdir(parents=True, exist_ok=True)

    rows = db.iterrows()
    if max_records:
        rows = list(rows)[:max_records]
        total = len(rows)
    else:
        total = len(db)

    paths: list[Path] = []
    for idx, (ecg_id, row) in enumerate(rows):
        # filename pattern from ptbxl_database.csv: filename_lr like 'records100/00000/00001_lr'
        rel = row[f"filename_{suffix}"]  # e.g. "records100/00000/00001_lr"
        for ext in (".hea", ".dat"):
            url = f"{PHYSIONET_BASE}/{rel}{ext}"
            local = out_root / "raw" / f"{rel}{ext}"
            try:
                _fetch(url, local)
            except Exception as e:
                print(f"\n  fetch failed for {rel}{ext}: {e}", flush=True)
                break
        paths.append(out_root / "raw" / rel)
        if (idx + 1) % 100 == 0:
            print(f"\r  records: {idx + 1}/{total}", end="", flush=True)
    print(f"\r  records: {total} done", flush=True)
    return paths


def load_signals(record_paths: list[Path]) -> np.ndarray:
    """Read 12-lead signals via wfdb. Returns (N, 12, T) array."""
    try:
        import wfdb
    except ImportError:
        print("\nERROR: wfdb not installed. Run:")
        print('  & "<venv>\\Scripts\\python.exe" -m pip install "wfdb>=4.1"')
        sys.exit(1)

    sigs = []
    for i, p in enumerate(record_paths):
        rec = wfdb.rdrecord(str(p))
        sigs.append(rec.p_signal.T.astype(np.float32))  # (12, T)
        if (i + 1) % 500 == 0:
            print(f"\r  loaded: {i + 1}/{len(record_paths)}", end="", flush=True)
    print(f"\r  loaded: {len(record_paths)} signals", flush=True)
    return np.stack(sigs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="./data/ptbxl")
    ap.add_argument("--sampling_rate", type=int, choices=[100, 500], default=100)
    ap.add_argument("--max_records", type=int, default=None,
                    help="Cap for smoke-testing (default: full 21,837)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    db, scp = download_metadata(out_root)

    # Map each record to a single superdiagnostic label
    db["label_str"] = db.scp_codes.apply(lambda c: aggregate_diagnostic_class(c, scp))
    db = db[db.label_str.notna()].copy()
    db["label_int"] = db.label_str.map(SUPERCLASS_TO_INT)

    if args.max_records:
        db = db.head(args.max_records)
    print(f"  retained {len(db)} records with valid superdiagnostic labels")
    print(f"  class distribution: {dict(db.label_str.value_counts())}")

    record_paths = download_records(out_root, db, args.sampling_rate, args.max_records)

    print("[3/3] Loading signals + writing NPY arrays...")
    X = load_signals(record_paths)
    y = db.label_int.values.astype(np.int64)
    pid = db.patient_id.values.astype(np.int64)

    assert X.shape[0] == len(y) == len(pid), \
        f"mismatch: X={X.shape[0]}, y={len(y)}, pid={len(pid)}"

    np.save(out_root / "X.npy", X)
    np.save(out_root / "y.npy", y)
    np.save(out_root / "patient_id.npy", pid)

    print(f"\nDone. Wrote:")
    print(f"  {out_root / 'X.npy'}            shape={X.shape}  dtype={X.dtype}")
    print(f"  {out_root / 'y.npy'}            shape={y.shape}  classes={np.unique(y).tolist()}")
    print(f"  {out_root / 'patient_id.npy'}   unique_patients={len(np.unique(pid))}")


if __name__ == "__main__":
    main()
