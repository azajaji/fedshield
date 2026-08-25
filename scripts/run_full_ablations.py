"""Expanded FEDShield ablation suite (publication-grade).

Variants:
  A0  Full FEDShield (v4)
  A1  - Edge AE         (alarm forced 0)
  A2  - Norm test       (no MAD outlier rejection; trust soft weight only)
  A3  - Cosine weight   (uniform weight over non-rejected)
  A4  - Warmup           (no FedAvg warmup, robust from round 0)
  A5  - Trust EMA        (alpha=0; instant scoring)
  A6  AE bottleneck=4    (smallest AE)
  A7  AE bottleneck=32   (largest AE)
  A8  k_mad=3            (more aggressive norm test)
  A9  k_mad=8            (laxer norm test)

Each variant runs across 3 datasets x 3 ratios (0.0, 0.2, 0.4) x 1 seed.
Outputs results/ablation.csv with one row per (dataset, variant) pair.
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


VARIANTS = [
    # Defense ablations — isolate each FEDShield component
    ("A0_full",            {}),
    ("A1_no_ae",           {"force_no_ae": True}),
    ("A2_no_normtest",     {"k_mad": 1e9}),
    ("A3_no_cosine",       {"flat_cosine": True}),
    ("A4_no_warmup",       {"warmup": 0}),
    ("A5_no_trust_ema",    {"alpha": 0.0}),
    ("A6_ae_b4",           {"ae_b": 4}),
    ("A7_ae_b32",          {"ae_b": 32}),
    ("A8_kmad_3",          {"k_mad": 3.0}),
    ("A9_kmad_8",          {"k_mad": 8.0}),
    # Attack ablations — isolate each adversarial primitive
    # (override attack.types to a single primitive instead of the composite)
    ("ATT_label_only",     {"attack_types": ["label_flip"]}),
    ("ATT_backdoor_only",  {"attack_types": ["backdoor"]}),
    ("ATT_signflip_only",  {"attack_types": ["sign_flip"]}),
    ("ATT_scaling_only",   {"attack_types": ["scaling"]}),
    ("ATT_data_only",      {"attack_types": ["label_flip", "backdoor"]}),
    ("ATT_model_only",     {"attack_types": ["sign_flip", "scaling"]}),
    ("ATT_all_composite",  {}),    # same as A0 — sanity baseline
]


def configure(cfg, opts):
    if "k_mad" in opts:
        cfg.defense.fedshield_k_mad = opts["k_mad"]
    if "warmup" in opts:
        cfg.defense.fedshield_warmup_rounds = opts["warmup"]
    if "alpha" in opts:
        cfg.defense.fedshield_alpha_ema = opts["alpha"]
    if "ae_b" in opts:
        cfg.model.ae_bottleneck = opts["ae_b"]
    if "attack_types" in opts:
        cfg.attack.types = list(opts["attack_types"])
    return cfg


def patch_runtime(opts):
    """Apply runtime patches to FEDShield's behaviour."""
    from fedshield import defenses as defmod
    orig = defmod.fedshield

    def wrapped(updates, ctx, cfg):
        if opts.get("force_no_ae"):
            for u in updates:
                u.edge_alarm = 0
                u.edge_scale = 1.0
        out, info = orig(updates, ctx, cfg)
        if opts.get("flat_cosine"):
            # uniform weight over the non-rejected set
            rejected = set(info.get("rejected", []))
            kept_ids = [u.client_id for u in updates if u.client_id not in rejected]
            if kept_ids:
                w = 1.0 / len(kept_ids)
                # rebuild uniform aggregation
                from fedshield.utils import flatten_state_dict, unflatten_to_state_dict
                import torch
                acc = None
                for u in updates:
                    if u.client_id in rejected:
                        continue
                    f = flatten_state_dict(u.delta) * w
                    acc = f if acc is None else acc + f
                if acc is not None:
                    out = unflatten_to_state_dict(acc, updates[0].delta)
                info["weights"] = {cid: w for cid in kept_ids}
        return out, info
    defmod.DEFENSES["fedshield"] = wrapped


def run_one(dataset, variant_name, opts):
    rows = []
    for r in (0.0, 0.2, 0.4):
        cfg = make_cfg(dataset, "fedshield", r, 0, rounds=20)
        cfg.data.num_clients = 6
        cfg.data.use_synthetic_fallback = False
        cfg.data.max_samples_per_client = 500
        cfg.federated.sample_ratio = 0.8
        if dataset == "ciciomt":
            cfg.model.num_classes = 5
        if dataset == "wesad":
            cfg.model.num_classes = 4
        cfg.name = f"abl_{dataset}_{variant_name}_r{int(r*100):02d}"
        cfg = configure(cfg, opts)
        patch_runtime(opts)
        out = FederatedTrainer(cfg).train()
        last = out["history"][-1]
        rows.append({
            "dataset": dataset,
            "variant": variant_name,
            "malicious_ratio": r,
            "acc": last["acc"],
            "asr": last["asr"],
            "defense_score": last.get("defense_score", float("nan")),
            "frr": last["frr"],
            "latency_edge_ms": last["latency_edge_ms"],
            "latency_server_ms": last["latency_server_ms"],
        })
    return rows


def main(datasets=None) -> None:
    datasets = datasets or ["mitbih"]      # MIT-BIH first; ciciomt/wesad after
    t0 = time.perf_counter()
    rows = []
    for ds in datasets:
        for name, opts in VARIANTS:
            rows.extend(run_one(ds, name, opts))
            print(f"[abl] {ds}/{name} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join("results", "ablation.csv")
    df.to_csv(out, index=False)
    print("wrote", out)
    print(df.groupby(["dataset", "variant", "malicious_ratio"])[["acc", "asr", "defense_score", "frr"]].mean())


if __name__ == "__main__":
    main()
