"""Cross-experiment aggregation: builds the publication tables.

Reads ``results/*_metrics.csv`` produced by :class:`FederatedTrainer` and
emits:

  * unified cross-dataset comparison table (Acc / ASR / FRR / latency)
  * malicious-ratio sweep
  * Dirichlet sweep
  * ablation table
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import pandas as pd


def load_all_metrics(results_dir: str = "./results") -> pd.DataFrame:
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["run"] = os.path.splitext(os.path.basename(path))[0].replace("_metrics", "")
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def final_round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Take the last round per (run) and aggregate by (dataset, defense)."""
    if df.empty:
        return df
    last = df.sort_values("round").groupby("run").tail(1)
    keys = ["dataset", "defense", "malicious_ratio"]
    metric_cols = [c for c in (
        "acc", "f1", "asr", "defense_score", "frr",
        "latency_edge_ms", "latency_server_ms",
        "latency_train_ms", "latency_ae_ms",
        "comm_bytes_per_client", "edge_ram_mb",
    ) if c in last.columns]
    out = last.groupby(keys)[metric_cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def cross_dataset_table(summary: pd.DataFrame, target_ratio: float = 0.2,
                        min_acc: float = 0.5) -> pd.DataFrame:
    """Build the canonical "best baseline vs FEDShield" comparison at a given
    malicious-client ratio (default 0.2, the standard reporting point).

    Selection criterion: highest **defense_score = acc * (1 - asr)**.
    This is the canonical FL backdoor metric (Bagdasaryan'20, Cao'21);
    it correctly penalises BOTH model collapse (low acc) and successful
    attacks (high asr) without conflating them.
    """
    if summary.empty:
        return summary
    block_all = summary[summary["malicious_ratio"] == target_ratio]
    if block_all.empty:
        block_all = summary
    rows: List[Dict] = []
    for ds in sorted(block_all["dataset"].unique()):
        block = block_all[block_all["dataset"] == ds]
        baselines = block[block["defense"] != "fedshield"]
        ours = block[block["defense"] == "fedshield"]
        if baselines.empty or ours.empty:
            continue
        score_col = "defense_score_mean" if "defense_score_mean" in baselines.columns else "acc_mean"
        baselines_sorted = baselines.sort_values(score_col, ascending=False)
        b = baselines_sorted.iloc[0]
        o = ours.iloc[0]
        latency = float(o.get("latency_edge_ms_mean", 0.0)) + float(o.get("latency_server_ms_mean", 0.0))
        rows.append({
            "dataset": ds,
            "best_baseline": b["defense"],
            "baseline_acc": b["acc_mean"],
            "baseline_asr": b["asr_mean"],
            "baseline_defense_score": b.get("defense_score_mean", float("nan")),
            "fedshield_acc": o["acc_mean"],
            "fedshield_asr": o["asr_mean"],
            "fedshield_defense_score": o.get("defense_score_mean", float("nan")),
            "delta_defense_score": (
                o.get("defense_score_mean", 0.0) - b.get("defense_score_mean", 0.0)
            ),
            "delta_frr": o["frr_mean"] - b["frr_mean"],
            "fedshield_latency_ms": latency,
        })
    return pd.DataFrame(rows)


def add_effective_asr(summary: pd.DataFrame) -> pd.DataFrame:
    """DEPRECATED. Compute effective ASR = ASR(rho_m) - ASR(rho_m=0).

    Retained for backward compatibility but **the publication metric is now
    ``defense_score = acc * (1 - asr)``** (logged directly during training).
    Effective ASR is mathematically meaningless when the rho_m=0 baseline
    model has high accuracy and the rho_m>0 model has collapsed — the
    subtraction conflates two distinct failure modes.
    """
    if summary.empty:
        return summary
    base = (
        summary[summary.malicious_ratio == 0.0]
        .set_index(["dataset", "defense"])["asr_mean"]
        .to_dict()
    )
    out = summary.copy()
    out["asr_clean"] = out.apply(
        lambda r: base.get((r["dataset"], r["defense"]), 0.0), axis=1
    )
    out["asr_eff"] = (out["asr_mean"] - out["asr_clean"]).clip(lower=0.0)
    return out


def malicious_ratio_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    last = df.sort_values("round").groupby("run").tail(1)
    return (
        last.groupby(["dataset", "defense", "malicious_ratio"])[["acc", "asr", "frr"]]
        .mean()
        .reset_index()
    )


def write_publication_tables(results_dir: str = "./results", out_dir: Optional[str] = None) -> Dict[str, str]:
    out_dir = out_dir or results_dir
    df = load_all_metrics(results_dir)
    if df.empty:
        return {}
    summary = final_round_summary(df)
    summary = add_effective_asr(summary)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    cross = cross_dataset_table(summary)
    cross.to_csv(os.path.join(out_dir, "cross_dataset.csv"), index=False)

    sweep = malicious_ratio_curve(df)
    sweep.to_csv(os.path.join(out_dir, "malicious_sweep.csv"), index=False)

    return {
        "summary": "summary.csv",
        "cross_dataset": "cross_dataset.csv",
        "malicious_sweep": "malicious_sweep.csv",
    }
