"""Preprocess PhysioNet 2020 (12-lead Challenge aggregate) into FedShield NPY.

The 2020 challenge merged 6 source databases (CPSC, CPSC-Extra, PTB,
PTB-XL, INCART, Georgia, Chapman-Shaoxing). PTB-XL is excluded here to
avoid overlap with our PTB-XL evaluation; remaining 5 sources form a
natural multi-site partition.

Output: data/physionet2020/X.npy (N, 12, T), y.npy (N,), site_id.npy (N,)
T is fixed at 5000 (10 s at 500 Hz; resampled if needed).

Label scheme: 5-class superdiagnostic (NORM/MI/STTC/CD/HYP), same as PTB-XL,
mapped from SNOMED-CT codes via a reduced subset.

Site IDs (used for non-IID per-site partitioning):
    0 = CPSC, 1 = CPSC-Extra, 2 = PTB, 3 = INCART, 4 = Georgia,
    5 = Chapman-Shaoxing  (PTB-XL excluded)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

# SNOMED-CT -> 5-class superdiagnostic mapping (subset; covers ~80% of records)
# Drawn from PhysioNet 2020 mapping table cross-referenced with PTB-XL SCP codes.
SNOMED_TO_SUPER = {
    # Normal
    "426783006": 0,    # Normal sinus rhythm (NSR)
    # Myocardial Infarction (MI)
    "164865005": 1,    # Q-wave abnormalities (associated with MI)
    "164867002": 1,    # T-wave inversion (associated with MI)
    "164873001": 1,    # Left axis deviation (often MI)
    "164931005": 1,    # ST-T abnormality (MI)
    "428750005": 1,    # ST-T change due to MI
    "164861001": 1,    # ST elevation
    # ST/T changes (STTC) — non-MI
    "164930006": 2,    # ST depression
    "164934002": 2,    # T-wave abnormal
    "59118001": 2,     # Right bundle branch block (sometimes)
    # Conduction Disorders (CD)
    "270492004": 3,    # First-degree AV block
    "164909002": 3,    # Left bundle branch block
    "713427006": 3,    # Complete right bundle branch block
    "445118002": 3,    # Left anterior fascicular block
    "39732003": 3,     # Left ventricular hypertrophy → reclass to HYP below
    # Hypertrophy (HYP)
    # (LVH 39732003 reassigned)
}
# Override: LVH should be HYP not CD
SNOMED_TO_SUPER["39732003"] = 4

SITE_IDS = {
    "cpsc_2018":            0,    # China  (~3,900 mappable records)
    "cpsc_2018_extra":      1,    # China extra  (~3,300)
    "georgia":              2,    # USA  (~8,090)
    "ptb":                  3,    # Germany original PTB  (~448)
    "st_petersburg_incart": None, # SKIPPED — only 27 mappable records,
                                  # too few for a per-site federation client
                                  # (causes num_samples=0 after non-IID partition)
    "ptb-xl":               None, # SKIPPED — overlap with our PTB-XL eval
}

TARGET_LEN = 5000  # 10 s at 500 Hz
TARGET_SR = 500


def parse_header(hea_text: str) -> tuple[int, int, list[str]]:
    """Returns (sampling_rate, num_samples, list_of_dx_codes)."""
    lines = hea_text.strip().split("\n")
    first = lines[0].split()
    sr = int(first[2])
    n_samples = int(first[3])
    dx_codes: list[str] = []
    for ln in lines:
        if ln.startswith("# Dx:"):
            dx_codes = [c.strip() for c in ln[5:].strip().split(",")]
            break
    return sr, n_samples, dx_codes


def aggregate_label(dx_codes: list[str]) -> int | None:
    """Map a list of SNOMED Dx codes to a single 5-class superdiagnostic.
    Strategy: vote across mappable codes; ties broken by class order
    (NORM > MI > STTC > CD > HYP)."""
    votes = [SNOMED_TO_SUPER[c] for c in dx_codes if c in SNOMED_TO_SUPER]
    if not votes:
        return None
    # most common, ties broken by class index ascending
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="./data/physionet_2020.zip")
    ap.add_argument("--out_root", default="./data/physionet2020")
    ap.add_argument("--max_per_site", type=int, default=None,
                    help="Cap per source DB for fast smoke testing")
    args = ap.parse_args()

    try:
        import wfdb
    except ImportError:
        print("Need wfdb: pip install wfdb")
        return
    try:
        from scipy.signal import resample
    except ImportError:
        print("Need scipy")
        return

    out = Path(args.out_root)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Step 1: extract zip (skip if already done)
    print(f"Inspecting {args.zip}...")
    with zipfile.ZipFile(args.zip, "r") as z:
        all_names = z.namelist()
        # Files we need: .hea + .mat per record (PhysioNet 2020 uses .mat instead of .dat)
        # Skip ptb-xl/ records (excluded)
        wanted = [n for n in all_names
                  if (n.endswith(".hea") or n.endswith(".mat"))
                  and "/ptb-xl/" not in n.lower()
                  and "/ptbxl/" not in n.lower()]
        # Group by site
        present = set()
        for n in wanted:
            for site_key, site_id in SITE_IDS.items():
                if site_id is None:
                    continue
                if f"/{site_key}/" in n.lower() or f"/{site_key}_" in n.lower():
                    present.add(site_key)
                    break
        print(f"  {len(wanted)} files across {len(present)} sites: {sorted(present)}")
        # Extract only files not already present
        for i, name in enumerate(wanted):
            dest = raw / Path(name).relative_to(Path(name).parts[0])
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if (i + 1) % 2000 == 0:
                print(f"  extracted {i + 1}/{len(wanted)}", flush=True)
    print("Extraction complete.")

    # Step 2: walk extracted files, read each record, label, resample
    Xs, ys, sites = [], [], []
    site_counts = {sid: 0 for sid in set(SITE_IDS.values()) if sid is not None}
    hea_files = sorted(raw.rglob("*.hea"))
    print(f"Loading {len(hea_files)} records...")

    # Sort site keys by length descending so "cpsc_2018_extra" matches before "cpsc_2018".
    site_keys_sorted = sorted(SITE_IDS.keys(), key=len, reverse=True)

    for i, hea in enumerate(hea_files):
        # Determine site by path-segment match (anchored on directory separators
        # to avoid prefix collisions like cpsc_2018 vs cpsc_2018_extra).
        path_str = str(hea).replace("\\", "/").lower()
        site_id = None
        for site_key in site_keys_sorted:
            sid = SITE_IDS[site_key]
            if sid is None:
                continue
            if f"/{site_key}/" in path_str:
                site_id = sid
                break
        if site_id is None:
            continue  # skip ptb-xl or unknown
        if args.max_per_site and site_counts[site_id] >= args.max_per_site:
            continue

        # Read header for label + signal info
        try:
            with open(hea, "r") as f:
                hea_text = f.read()
            sr, n_samples, dx_codes = parse_header(hea_text)
            label = aggregate_label(dx_codes)
            if label is None:
                continue
            # Read signal
            rec = wfdb.rdrecord(str(hea.with_suffix("")))
            sig = rec.p_signal.T.astype(np.float32)  # (12, T)
            if sig.shape[0] != 12:
                continue
            # Resample to TARGET_SR if needed
            if sr != TARGET_SR:
                new_len = int(sig.shape[1] * TARGET_SR / sr)
                sig = resample(sig, new_len, axis=1).astype(np.float32)
            # Truncate / pad to TARGET_LEN
            if sig.shape[1] >= TARGET_LEN:
                sig = sig[:, :TARGET_LEN]
            else:
                pad = np.zeros((12, TARGET_LEN - sig.shape[1]), dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=1)
            Xs.append(sig)
            ys.append(label)
            sites.append(site_id)
            site_counts[site_id] += 1
        except Exception:
            continue
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(hea_files)}  (site_counts: {site_counts})", flush=True)

    print(f"\nLoaded {len(Xs)} records total")
    print(f"  per-site counts: {site_counts}")

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    s = np.asarray(sites, dtype=np.int64)

    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "site_id.npy", s)
    print(f"\nWrote {out}/X.npy {X.shape}  y={np.unique(y).tolist()}  sites={np.unique(s).tolist()}")


if __name__ == "__main__":
    main()
