"""Publication-quality figures (300 DPI, IEEE/MDPI-ready).

All plotting functions accept the dataframes produced by :mod:`evaluation`
and write PNG + PDF outputs to ``paper/figs/``.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# IEEE/MDPI typesetting defaults
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
})

DEFENSE_ORDER = ["fedavg", "krum", "trimmed_mean", "foolsgold", "fltrust", "fedshield"]
DEFENSE_LABEL = {
    "fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "Trim. Mean",
    "foolsgold": "FoolsGold", "fltrust": "FLTrust", "fedshield": "FEDShield",
}
DEFENSE_COLOR = {
    "fedavg": "#7f7f7f", "krum": "#1f77b4", "trimmed_mean": "#2ca02c",
    "foolsgold": "#9467bd", "fltrust": "#ff7f0e", "fedshield": "#d62728",
}


def _save(fig: plt.Figure, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_accuracy_vs_attack(df: pd.DataFrame, dataset: str, out_path: str) -> None:
    """Accuracy vs malicious-client ratio, one line per defense."""
    sub = df[df.dataset == dataset]
    if sub.empty:
        return
    last = sub.sort_values("round").groupby(["defense", "malicious_ratio", "run"]).tail(1)
    agg = last.groupby(["defense", "malicious_ratio"])["acc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for defn in DEFENSE_ORDER:
        d = agg[agg.defense == defn]
        if d.empty:
            continue
        ax.errorbar(d["malicious_ratio"], d["mean"], yerr=d["std"],
                    label=DEFENSE_LABEL[defn], color=DEFENSE_COLOR[defn],
                    marker="o", lw=1.2, ms=4, capsize=2)
    ax.set_xlabel(r"Malicious client ratio $\rho_m$")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs attack strength — {dataset}")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    _save(fig, out_path)


def fig_asr_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar: ASR per (dataset, defense) at default rho_m=0.2."""
    last = df.sort_values("round").groupby("run").tail(1)
    last = last[last.malicious_ratio == 0.2]
    if last.empty:
        return
    pivot = last.groupby(["dataset", "defense"])["asr"].mean().unstack("defense")
    pivot = pivot[[c for c in DEFENSE_ORDER if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(pivot.index))
    n = len(pivot.columns)
    width = 0.8 / max(n, 1)
    for i, defn in enumerate(pivot.columns):
        ax.bar(x + (i - n / 2 + 0.5) * width, pivot[defn].values,
               width=width, label=DEFENSE_LABEL[defn], color=DEFENSE_COLOR[defn])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Attack Success Rate")
    ax.set_title(r"ASR across defenses ($\rho_m = 0.2$)")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    _save(fig, out_path)


def fig_roc_like(df_alarms: pd.DataFrame, out_path: str) -> None:
    """Sweep AE threshold k and plot detection rate vs FRR."""
    if df_alarms.empty:
        return
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for ds, sub in df_alarms.groupby("dataset"):
        sub = sub.sort_values("frr")
        ax.plot(sub.frr, sub.detection, marker="o", lw=1.3, ms=4, label=ds)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False Rejection Rate (FRR)")
    ax.set_ylabel("Malicious Detection Rate")
    ax.set_title("Edge AE detection — ROC-like curve")
    ax.legend(frameon=False)
    _save(fig, out_path)


def fig_confusion(matrix: np.ndarray, class_names: List[str], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:d}", ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()/2 else "black",
                    fontsize=7)
    _save(fig, out_path)


def fig_latency_vs_perf(df: pd.DataFrame, out_path: str) -> None:
    last = df.sort_values("round").groupby("run").tail(1)
    last = last[last.malicious_ratio == 0.2]
    if last.empty:
        return
    last["latency_total_ms"] = last.latency_edge_ms + last.latency_server_ms
    agg = last.groupby(["dataset", "defense"]).agg(
        acc=("acc", "mean"), latency=("latency_total_ms", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for defn in DEFENSE_ORDER:
        d = agg[agg.defense == defn]
        if d.empty:
            continue
        ax.scatter(d.latency, d.acc, label=DEFENSE_LABEL[defn],
                   color=DEFENSE_COLOR[defn], s=40, alpha=0.85)
        for _, r in d.iterrows():
            ax.annotate(r.dataset, (r.latency, r.acc), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Round latency (ms, edge + server)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Latency vs performance")
    ax.legend(loc="lower right", ncol=2, frameon=False)
    _save(fig, out_path)


def fig_ablation(df_ablation: pd.DataFrame, out_path: str) -> None:
    """Bar plot of accuracy / ASR for each ablation variant."""
    if df_ablation.empty:
        return
    fig, ax1 = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(df_ablation))
    w = 0.4
    ax1.bar(x - w/2, df_ablation.acc, w, label="Accuracy", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_ablation.variant, rotation=20, ha="right")
    ax1.set_ylabel("Accuracy", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, df_ablation.asr, w, label="ASR", color="#d62728")
    ax2.set_ylabel("ASR", color="#d62728")
    ax2.grid(False)
    ax1.set_title("FEDShield ablation")
    _save(fig, out_path)


def fig_effective_asr(summary: pd.DataFrame, out_path: str) -> None:
    """Bar chart of *effective* ASR (= ASR(rho) - ASR(0)) per dataset, defense.
    The metric subtracts the trigger's innate efficacy from ASR to isolate the
    additional attack power obtained by the adversary; this is the canonical
    backdoor-defense metric in the FL literature.
    """
    if summary.empty or "asr_eff" not in summary.columns:
        return
    sub = summary[summary.malicious_ratio == 0.2]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="dataset", columns="defense", values="asr_eff", aggfunc="mean")
    pivot = pivot[[c for c in DEFENSE_ORDER if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(pivot.index))
    n = len(pivot.columns)
    width = 0.8 / max(n, 1)
    for i, defn in enumerate(pivot.columns):
        ax.bar(x + (i - n / 2 + 0.5) * width, pivot[defn].values,
               width=width, label=DEFENSE_LABEL[defn], color=DEFENSE_COLOR[defn])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel(r"Effective ASR  ($\mathrm{ASR}(\rho_m{=}0.2)-\mathrm{ASR}(\rho_m{=}0)$)")
    ax.set_title("Backdoor effectiveness across defenses (lower is better)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    _save(fig, out_path)


def fig_robustness_pareto(summary: pd.DataFrame, out_path: str) -> None:
    """Pareto frontier: accuracy vs effective ASR at rho=0.2, all defenses,
    one marker per (dataset, defense). FEDShield should sit on the upper-left."""
    if summary.empty:
        return
    sub = summary[summary.malicious_ratio == 0.2]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for defn in DEFENSE_ORDER:
        d = sub[sub.defense == defn]
        if d.empty:
            continue
        x = d["asr_eff"] if "asr_eff" in d.columns else d["asr_mean"]
        ax.scatter(x, d["acc_mean"], label=DEFENSE_LABEL[defn],
                   color=DEFENSE_COLOR[defn], s=46, alpha=0.85)
        for _, r in d.iterrows():
            ax.annotate(r["dataset"], (r["asr_eff"], r["acc_mean"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Effective ASR (lower = better)")
    ax.set_ylabel("Accuracy (higher = better)")
    ax.set_title(r"Robustness Pareto at $\rho_m=0.2$")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    _save(fig, out_path)


def fig_ablation_heatmap(df_abl: pd.DataFrame, metric: str, out_path: str) -> None:
    """Heatmap of (variant x dataset) for the chosen metric (acc / asr / frr)."""
    if df_abl.empty:
        return
    sub = df_abl[df_abl.malicious_ratio == 0.2]
    if sub.empty:
        sub = df_abl
    pivot = sub.pivot_table(index="variant", columns="dataset", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r" if metric == "asr" else "viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_title(f"FEDShield ablation — {metric.upper()}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if metric == "asr" and v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, out_path)


def fig_attack_matrix_heatmap(matrix_csv: str, out_path: str, rho: float = 0.2) -> None:
    """Per-attack × per-defense defense_score heatmap from `attack_matrix.csv`.
    Color-coded so darker green = better defense, darker red = collapse.
    """
    if not os.path.exists(matrix_csv):
        return
    df = pd.read_csv(matrix_csv)
    sub = df[df.rho == rho]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="defense", columns="attack",
                            values="ds", aggfunc="mean")
    defense_order = ["fedavg", "krum", "trimmed_mean", "foolsgold", "fltrust", "fedshield"]
    attack_order = ["label_flip", "backdoor", "data_only", "noise_update",
                    "sign_flip", "scaling", "model_only", "all_composite"]
    pivot = pivot.reindex(index=[d for d in defense_order if d in pivot.index],
                          columns=[a for a in attack_order if a in pivot.columns])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.0, vmax=0.55, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([a.replace("_", "\n") for a in pivot.columns], fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([DEFENSE_LABEL.get(d, d) for d in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="black")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.15 or v > 0.45 else "black")
    ax.set_title(rf"Defense Score per (defense, attack) at $\rho_m={rho}$  —  higher is better")
    fig.colorbar(im, fraction=0.025, pad=0.02, label=r"$\mathrm{Acc}\cdot(1-\mathrm{ASR})$")
    _save(fig, out_path)


def fig_round_curves(df: pd.DataFrame, dataset: str, metric: str, out_path: str) -> None:
    sub = df[df.dataset == dataset]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for defn in DEFENSE_ORDER:
        d = sub[sub.defense == defn]
        if d.empty:
            continue
        agg = d.groupby("round")[metric].mean()
        ax.plot(agg.index, agg.values, label=DEFENSE_LABEL[defn],
                color=DEFENSE_COLOR[defn], lw=1.4)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(metric.upper() if metric in {"acc","f1","asr","frr"} else metric)
    ax.set_title(f"{metric} over rounds — {dataset}")
    ax.legend(loc="best", ncol=2, frameon=False)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
def assert_chart_completeness(df: pd.DataFrame) -> List[str]:
    """Return a list of human-readable warnings about (dataset, defense,
    malicious_ratio) cells that are missing from the metrics. Empty list = OK.
    """
    warnings: List[str] = []
    if df.empty:
        return ["[viz] no metrics rows"]
    expected_defenses = set(DEFENSE_ORDER)
    for ds, dsub in df.groupby("dataset"):
        present = set(dsub.defense.unique())
        missing = expected_defenses - present
        if missing:
            warnings.append(f"[viz] dataset={ds} missing defenses: {sorted(missing)}")
        ratios = sorted(dsub.malicious_ratio.unique())
        for d in present:
            d_ratios = set(dsub[dsub.defense == d].malicious_ratio.unique())
            gap = set(ratios) - d_ratios
            if gap:
                warnings.append(
                    f"[viz] dataset={ds} defense={d} missing ratios: {sorted(gap)}"
                )
    return warnings


def render_all(results_dir: str = "./results", figs_dir: str = "./paper/figs") -> Dict[str, str]:
    """Convenience: load metrics CSVs and render every standard figure."""
    from .evaluation import load_all_metrics
    df = load_all_metrics(results_dir)
    if df.empty:
        print("[viz] no metrics found in", results_dir)
        return {}
    out = {}
    for ds in sorted(df.dataset.unique()):
        fig_accuracy_vs_attack(df, ds, os.path.join(figs_dir, f"acc_vs_rho_{ds}"))
        fig_round_curves(df, ds, "acc", os.path.join(figs_dir, f"acc_curve_{ds}"))
        fig_round_curves(df, ds, "asr", os.path.join(figs_dir, f"asr_curve_{ds}"))
        out[ds] = "ok"
    fig_asr_comparison(df, os.path.join(figs_dir, "asr_grouped"))
    fig_latency_vs_perf(df, os.path.join(figs_dir, "latency_vs_acc"))
    # publication-grade additions: effective-ASR + robustness Pareto
    from .evaluation import final_round_summary, add_effective_asr
    summary = add_effective_asr(final_round_summary(df))
    fig_effective_asr(summary, os.path.join(figs_dir, "asr_effective"))
    fig_robustness_pareto(summary, os.path.join(figs_dir, "robustness_pareto"))
    # ablation heatmaps if ablation.csv exists
    abl_path = os.path.join(results_dir, "ablation.csv")
    if os.path.exists(abl_path):
        df_abl = pd.read_csv(abl_path)
        for metric in ("acc", "asr", "frr"):
            fig_ablation_heatmap(df_abl, metric,
                                 os.path.join(figs_dir, f"ablation_{metric}"))
    return out


if __name__ == "__main__":
    render_all()
