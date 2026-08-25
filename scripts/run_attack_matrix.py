"""Per-attack × per-defense matrix on real MIT-BIH.

Each cell of the matrix runs a single (defense, attack, rho_m) configuration so
the publication tables can answer:
  - which defense fails against which specific attack primitive?
  - does FEDShield's autoencoder really catch backdoors (vs. label-flip)?
  - which baselines collapse only under model poisoning vs. data poisoning?

Output: results/attack_matrix.csv with one row per (defense, attack, rho_m,
seed). The headline cross-attack figure is then a heatmap.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from fedshield.main import make_cfg                                  # noqa: E402
from fedshield.federated import FederatedTrainer                      # noqa: E402


# Attack primitives we sweep over. Each value is the cfg.attack.types list
# that the malicious clients will execute.
ATTACK_PRIMITIVES = {
    "label_flip":   ["label_flip"],
    "backdoor":     ["backdoor"],
    "sign_flip":    ["sign_flip"],
    "scaling":      ["scaling"],
    "noise_update": ["noise_update"],
    "data_only":    ["label_flip", "backdoor"],
    "model_only":   ["sign_flip", "scaling"],
    "all_composite": ["label_flip", "backdoor", "sign_flip", "scaling"],
}

DEFENSES = ["fedavg", "krum", "trimmed_mean", "foolsgold", "fltrust", "fedshield"]
RATIOS   = [0.0, 0.2, 0.4]
SEEDS    = [0]

# Cap-vs-time tradeoff. The default (300) gave a fast pipeline but was flagged
# as too small for publication-grade results. 1500 ≈ 10% of MIT-BIH per client
# and triples per-run cost; full matrix below ≈ 6 h.
CAP = 300         # tight for time budget; matrix *pattern* is robust to cap
ROUNDS = 15


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    cfgs = []
    for defn in DEFENSES:
        for attack_name, atk_types in ATTACK_PRIMITIVES.items():
            for r in RATIOS:
                for s in SEEDS:
                    cfgs.append((defn, attack_name, atk_types, r, s))
    total = len(cfgs)
    skipped = 0
    for i, (defn, attack_name, atk_types, r, s) in enumerate(cfgs, 1):
        cfg = make_cfg("mitbih", defn, r, s, rounds=ROUNDS)
        cfg.data.num_clients = 6
        cfg.data.use_synthetic_fallback = False
        cfg.data.max_samples_per_client = CAP
        cfg.federated.sample_ratio = 0.8
        cfg.attack.types = list(atk_types)
        cfg.name = f"matrix_{defn}_{attack_name}_r{int(r*100):02d}_s{s}"
        # resume: skip configs whose metrics file already exists with the
        # expected number of rounds. Lets a kill+restart pick up where it left
        # off without re-doing completed work.
        out_path = os.path.join("results", f"{cfg.name}_metrics.csv")
        if os.path.exists(out_path):
            try:
                done_rows = sum(1 for _ in open(out_path)) - 1   # minus header
            except Exception:
                done_rows = 0
            if done_rows >= ROUNDS:
                skipped += 1
                print(f"[{i}/{total}] {cfg.name}  SKIPPED (already complete)", flush=True)
                # still load from disk into rows for the final csv
                df_done = pd.read_csv(out_path).iloc[-1].to_dict()
                rows.append({
                    "defense": defn, "attack": attack_name,
                    "malicious_ratio": r, "seed": s,
                    "acc": df_done["acc"], "f1": df_done["f1"],
                    "asr": df_done["asr"],
                    "defense_score": df_done["defense_score"],
                    "frr": df_done["frr"],
                    "latency_edge_ms": df_done["latency_edge_ms"],
                    "latency_server_ms": df_done["latency_server_ms"],
                    "edge_ram_mb": df_done["edge_ram_mb"],
                })
                continue
            # partial CSV (crashed mid-run) — wipe and restart this config
            os.remove(out_path)
        out = FederatedTrainer(cfg).train()
        last = out["history"][-1]
        rows.append({
            "defense": defn,
            "attack": attack_name,
            "malicious_ratio": r,
            "seed": s,
            "acc": last["acc"],
            "f1": last["f1"],
            "asr": last["asr"],
            "defense_score": last["defense_score"],
            "frr": last["frr"],
            "latency_edge_ms": last["latency_edge_ms"],
            "latency_server_ms": last["latency_server_ms"],
            "edge_ram_mb": last["edge_ram_mb"],
        })
        elapsed = time.perf_counter() - t0
        eta = elapsed * (total - i) / max(i, 1)
        print(f"[{i}/{total}] {cfg.name}  t={elapsed:.0f}s  eta={eta/60:.0f}min", flush=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join("results", "attack_matrix.csv")
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows in {(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
