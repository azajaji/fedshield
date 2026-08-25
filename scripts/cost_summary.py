"""Per-defense cost summary across results/proto/<dataset>/*.csv.

Aggregates latency_edge_ms, latency_server_ms, comm_bytes_per_client,
edge_ram_mb across all completed proto runs and prints one row per
(dataset, defense). Run after the headline sweep to populate Table 4.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(
    r"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_"
    r"(?P<attack>[a-z_+]+)_r(?P<mr>\d{2})_s(?P<seed>\d{2})_metrics\.csv$"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto_root", default="./results/proto")
    args = ap.parse_args()

    rows = []
    for csv in Path(args.proto_root).rglob("proto_*_metrics.csv"):
        m = FILENAME_RE.match(csv.name)
        if not m:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if len(df) == 0:
            continue
        last = df.iloc[-1]
        rows.append({
            "dataset": m.group("dataset"),
            "variant": m.group("variant"),
            "latency_edge_ms":      float(last.get("latency_edge_ms", float("nan"))),
            "latency_server_ms":    float(last.get("latency_server_ms", float("nan"))),
            "latency_train_ms":     float(last.get("latency_train_ms", float("nan"))),
            "comm_bytes_per_client":float(last.get("comm_bytes_per_client", float("nan"))),
            "edge_ram_mb":          float(last.get("edge_ram_mb", float("nan"))),
            "n_params":             float(last.get("n_params", float("nan"))),
        })

    if not rows:
        print(f"No proto result CSVs under {args.proto_root}")
        return

    full = pd.DataFrame(rows)
    full["comm_kb_per_client"] = full["comm_bytes_per_client"] / 1024.0
    cols = ["latency_edge_ms", "latency_server_ms", "latency_train_ms",
            "comm_kb_per_client", "edge_ram_mb"]
    summary = (
        full.groupby(["dataset", "variant"])[cols]
            .mean()
            .reset_index()
            .sort_values(["dataset", "variant"])
    )

    print(f"=== Cost summary across {len(full)} runs ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
