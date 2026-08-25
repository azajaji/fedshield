"""Compute Tier-1 alternative metrics for FedShield: pairwise win-rate,
CVaR-25 Score_F1, and cross-dataset stability index.

Loads the same proto_* CSVs used by make_figures.py and computes:
  (A) Pairwise win-rate matrix (aggregated + per-dataset)
  (B) CVaR-25 Score_F1 per method (mean of worst quartile of cells)
  (C) Stability index = CV (std/mean) of per-dataset mean Score_F1
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

KNOWN_ATTACKS = [
    "label_flip+backdoor+sign_flip+scaling",
    "sign_flip+scaling", "backdoor+scaling",
    "sign_flip", "label_flip", "backdoor", "scaling", "noise_update", "clean",
]
_ATTACK_ALT = "|".join(re.escape(a) for a in KNOWN_ATTACKS)
RE = re.compile(
    rf"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_"
    rf"(?P<attack>{_ATTACK_ALT})_r(?P<mr>\d{{2}})_s(?P<seed>\d{{2}})"
    rf"(?:_n(?P<n>\d{{2}}))?_metrics\.csv$"
)
VARIANTS = {
    "fedavg": "FedAvg", "krum": "Krum", "multi_krum": "Multi-Krum",
    "trimmed_mean": "Trim-Mean",
    "fltrust": "FLTrust",
    "fedshield_v10_a025": "FedShield",
    "fedshield_v10": "FedShield-Compact",
}
DATASETS = {
    "mitbih": "MIT-BIH", "ciciomt": "CIC-IoMT-2024",
    "ptbxl": "PTB-XL", "physionet2017": "PhysioNet/CinC 2017",
}
METHOD_ORDER = list(VARIANTS.keys())


def load() -> pd.DataFrame:
    rows = []
    for csv in Path("results/proto").rglob("proto_*_metrics.csv"):
        m = RE.match(csv.name)
        if not m or m.group("n"):
            continue
        if m.group("variant") not in VARIANTS or m.group("dataset") not in DATASETS:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if len(df) == 0:
            continue
        last = df.iloc[-1]
        f1  = float(last.get("f1",  float("nan")))
        asr = float(last.get("asr", float("nan")))
        if np.isnan(f1) or np.isnan(asr):
            continue
        rows.append({
            "dataset": m.group("dataset"),
            "variant": m.group("variant"),
            "attack":  m.group("attack"),
            "mr":      round(float(m.group("mr")) / 100, 2),
            "seed":    int(m.group("seed")),
            "score":   f1 * (1.0 - asr),
        })
    df = pd.DataFrame(rows)
    df = df[df.mr > 0].reset_index(drop=True)
    return df


def pairwise_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """For each (row, col) method pair, fraction of matched cells where row > col."""
    methods = METHOD_ORDER
    mat = pd.DataFrame(np.nan, index=methods, columns=methods)
    for a in methods:
        for b in methods:
            if a == b:
                mat.loc[a, b] = 0.5
                continue
            sub_a = df[df.variant == a][["dataset", "attack", "mr", "seed", "score"]] \
                       .rename(columns={"score": "score_a"})
            sub_b = df[df.variant == b][["dataset", "attack", "mr", "seed", "score"]] \
                       .rename(columns={"score": "score_b"})
            m = sub_a.merge(sub_b, on=["dataset", "attack", "mr", "seed"], how="inner")
            if len(m) == 0:
                continue
            wins = (m["score_a"] > m["score_b"]).sum()
            ties = (m["score_a"] == m["score_b"]).sum()
            rate = (wins + 0.5 * ties) / len(m)
            mat.loc[a, b] = rate
    return mat


def per_dataset_winrate(df: pd.DataFrame) -> dict:
    out = {}
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        if len(sub) == 0: continue
        out[ds] = pairwise_winrate(sub)
    return out


def cvar25(df: pd.DataFrame) -> pd.Series:
    """Mean of worst 25% of cells per method, on Score_F1."""
    out = {}
    for v in METHOD_ORDER:
        sub = df[df.variant == v]["score"].dropna()
        if len(sub) == 0:
            out[v] = float("nan"); continue
        q25 = sub.quantile(0.25)
        worst_quartile = sub[sub <= q25]
        out[v] = worst_quartile.mean()
    return pd.Series(out, name="CVaR25_ScoreF1")


def stability_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation (CV = std/mean) of per-dataset mean Score_F1."""
    per_ds = df.groupby(["dataset", "variant"])["score"].mean().unstack("variant")
    per_ds = per_ds.reindex(columns=METHOD_ORDER)
    rows = []
    for v in METHOD_ORDER:
        vals = per_ds[v].values
        m = vals.mean()
        s = vals.std(ddof=1)
        cv = s / m if m > 0 else float("nan")
        rows.append({"method": VARIANTS[v], "mean": m, "std": s, "CV": cv})
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} attacked cells across {df.dataset.nunique()} datasets.")
    print(f"Cells per method:")
    print(df.variant.value_counts().to_string())
    print()

    # (A) Pairwise win-rate, aggregated
    print("\n=== (A) Pairwise win-rate, AGGREGATED across all 4 datasets ===")
    print("(rows beat columns; values are win-rate of row vs col across matched cells)")
    mat = pairwise_winrate(df)
    mat_disp = mat.rename(index=VARIANTS, columns=VARIANTS)
    print(mat_disp.round(3).to_string())

    print("\n--- Headline summary: FedShield's win-rate vs each baseline (pooled) ---")
    fs = "fedshield_v10_a025"
    for b in METHOD_ORDER:
        if b == fs: continue
        print(f"  FedShield vs {VARIANTS[b]:20s} = {mat.loc[fs, b]:.3f}")

    # (A) Per-dataset win-rate
    print("\n=== (A) Pairwise win-rate, PER-DATASET (FedShield row only) ===")
    ds_mats = per_dataset_winrate(df)
    head = "{:20s} | " + " | ".join([f"{DATASETS[d]:>22s}" for d in DATASETS])
    print(head.format(""))
    for b in METHOD_ORDER:
        if b == fs: continue
        cells = []
        for ds in DATASETS:
            v = ds_mats.get(ds, pd.DataFrame()).loc[fs, b] if ds in ds_mats else float("nan")
            cells.append(f"{v:>22.3f}")
        print(f"FedShield vs {VARIANTS[b]:8s}| " + " | ".join(cells))

    # (B) CVaR-25
    print("\n=== (B) CVaR-25 Score_F1 (mean of worst quartile of cells per method) ===")
    cvar = cvar25(df)
    cvar_disp = cvar.rename(VARIANTS)
    print(cvar_disp.sort_values(ascending=False).round(3).to_string())

    # (C) Stability index
    print("\n=== (C) Stability index (CV = std/mean of per-dataset mean Score_F1) ===")
    stab = stability_cv(df)
    print(stab.assign(
        mean=lambda d: d["mean"].round(3),
        std=lambda d: d["std"].round(3),
        CV=lambda d: d["CV"].round(3),
    ).to_string(index=False))


if __name__ == "__main__":
    main()
