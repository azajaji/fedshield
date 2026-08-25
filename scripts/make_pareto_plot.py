"""Generate the defense-score vs latency Pareto plot from completed sweep results.

Reads results/proto/<dataset>/proto_*_metrics.csv, aggregates by defense
(headline variants only), and plots:
    x-axis = mean per-round server latency (ms)
    y-axis = mean defense_score across the headline cells
Each defense is one point; FedShield (the proposed method) is highlighted.

Run AFTER Sweep 1 (headline matrix) finishes:
    python -m scripts.make_pareto_plot --dataset mitbih
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(
    r"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_"
    r"(?P<attack>[a-z_+]+)_r(?P<mr>\d{2})_s(?P<seed>\d{2})(?:_n\d{2})?_metrics\.csv$"
)

# Variants that belong in the headline matrix Pareto.
HEADLINE_VARIANTS = {
    "fedavg", "krum", "multi_krum", "trimmed_mean",
    "foolsgold", "fltrust", "median",
    "fedshield_v10", "fedshield_v10_a025",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mitbih",
                    choices=["mitbih", "wesad", "ciciomt", "ptbxl",
                             "physionet2017", "physionet2020"])
    ap.add_argument("--proto_root", default="./results/proto")
    ap.add_argument("--out", default="./paper/figs/pareto_score_vs_latency.pdf")
    args = ap.parse_args()

    rows = []
    for csv in Path(args.proto_root, args.dataset).rglob("proto_*_metrics.csv"):
        m = FILENAME_RE.match(csv.name)
        if not m or m.group("variant") not in HEADLINE_VARIANTS:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if len(df) == 0:
            continue
        last = df.iloc[-1]
        rows.append({
            "variant": m.group("variant"),
            "attack": m.group("attack"),
            "mr": float(m.group("mr")) / 100,
            "score": float(last.get("defense_score", float("nan"))),
            "latency_server_ms": float(last.get("latency_server_ms", float("nan"))),
            "latency_edge_ms":   float(last.get("latency_edge_ms", float("nan"))),
        })

    if not rows:
        print(f"No headline runs found under {args.proto_root}/{args.dataset}")
        return

    df = pd.DataFrame(rows)
    agg = df.groupby("variant").agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        mean_lat_server=("latency_server_ms", "mean"),
        mean_lat_edge=("latency_edge_ms", "mean"),
    ).reset_index()
    print("=== headline aggregates ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Lazy plotting import so the script works on systems without matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for _, row in agg.iterrows():
        is_ours = row["variant"] == "fedshield_v10"
        marker = "*" if is_ours else "o"
        size = 200 if is_ours else 80
        color = "tab:red" if is_ours else "tab:blue"
        label = "FedShield (ours)" if is_ours else row["variant"]
        ax.scatter(row["mean_lat_server"], row["mean_score"],
                   s=size, marker=marker, color=color, alpha=0.85,
                   edgecolors="black", zorder=3, label=label)
        ax.annotate(label, (row["mean_lat_server"], row["mean_score"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Server latency per round (ms)")
    ax.set_ylabel("Mean defense score across headline cells")
    ax.set_title(f"Cost vs robustness on {args.dataset.upper()}")
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
