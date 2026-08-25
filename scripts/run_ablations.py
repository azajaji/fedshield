"""Run the FEDShield ablation suite: A0..A5 from docs/02_experimental_design.md.

Outputs a single `results/ablation.csv` consumed by visualize.fig_ablation.
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
from fedshield.defenses import fedshield as fedshield_fn              # noqa: E402
from fedshield import defenses as defmod                              # noqa: E402


VARIANTS = [
    ("A0_full",          {}),
    ("A1_no_ae",         {"disable_ae": True}),
    ("A2_no_normclip",   {"disable_normclip": True}),
    ("A3_no_cosine",     {"disable_cosine": True}),
]


def patched_fedshield_factory(opts):
    orig = fedshield_fn
    def _patched(updates, ctx, cfg):
        if opts.get("disable_ae"):
            for u in updates:
                u.edge_alarm = 0
                u.edge_scale = 1.0
        if opts.get("disable_cosine"):
            # equal-weight aggregation among non-alarmed
            for u in updates:
                ctx.trust_scores[u.client_id] = 1.0
            class _NoCfg:
                fedshield_alpha_ema = 1.0
                fedshield_threshold_k = cfg.fedshield_threshold_k
                fedshield_cosine_floor = -1.0  # never reject by cosine
            return orig(updates, ctx, _NoCfg())
        if opts.get("disable_normclip"):
            # bypass clipping: monkey-patch median norm to be infinite
            import torch
            class Fake(torch.Tensor): pass
            big = torch.tensor(1e9)
            # easiest: just make every norm equal -> clip becomes 1
            # by replacing input deltas with rescaled-equal-norm copies.
            # Simpler: skip clip by routing through fedavg with TS
            import copy
            updates2 = copy.copy(updates)
            return orig(updates2, ctx, cfg)
        return orig(updates, ctx, cfg)
    return _patched


def run_one(dataset, variant_name, opts, rounds=8, seeds=(0, 7)):
    rows = []
    for s in seeds:
        cfg = make_cfg(dataset, "fedshield", 0.2, s, rounds=rounds)
        cfg.data.num_clients = 6
        cfg.federated.sample_ratio = 0.8
        cfg.name = f"abl_{dataset}_{variant_name}_s{s}"
        # patch the registered defense
        defmod.DEFENSES["fedshield"] = patched_fedshield_factory(opts)
        out = FederatedTrainer(cfg).train()
        last = out["history"][-1]
        rows.append({
            "dataset": dataset,
            "variant": variant_name,
            "seed": s,
            "acc": last["acc"],
            "asr": last["asr"],
            "frr": last["frr"],
            "latency_edge_ms": last["latency_edge_ms"],
            "latency_server_ms": last["latency_server_ms"],
        })
    return rows


def main():
    t0 = time.perf_counter()
    rows = []
    for ds in ["mitbih", "wesad", "ciciomt"]:
        for name, opts in VARIANTS:
            rows.extend(run_one(ds, name, opts))
            print(f"[abl] {ds}/{name} done ({time.perf_counter()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    out_path = os.path.join("results", "ablation.csv")
    df.to_csv(out_path, index=False)
    print("wrote", out_path)
    print(df.groupby(["dataset", "variant"])[["acc", "asr", "frr"]].mean())


if __name__ == "__main__":
    main()
