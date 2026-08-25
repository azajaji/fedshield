"""Krum-aware mimicry stress-test sweep.

Replaces every malicious client's submitted delta with `mu_honest + eps * v_attack`
(server-side, before defense aggregation). See fedshield/attacks.py
`apply_post_collection_mimicry`. The attack is activated by adding "mimicry"
to AttackConfig.types.

This script runs the sweep across (dataset, eps, rho_m, seed, defense),
caches each cell's per-round metrics under
    results/proto/<dataset>/proto_<dataset>_<defense>_mimicry_eps<E>_r<RR>_s<SS>_metrics.csv
and is fully resumable: cached cells are skipped on rerun.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Allow the Anaconda + torch OpenMP conflict (harmless single-thread workaround).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch

from fedshield.config import (
    AttackConfig, DataConfig, DefenseConfig, ExperimentConfig,
    FederatedConfig, ModelConfig,
)
from fedshield.federated import FederatedTrainer
from scripts.proto_bench import (
    BASELINES_VARIANTS, DATASET_PRESETS,
)


def _ds_preset(ds: str) -> dict:
    # Translate the human dataset name to proto_bench's DATASET_PRESETS keys.
    aliases = {"physionet2017": "physionet2017", "physionet_2017": "physionet2017"}
    return DATASET_PRESETS[aliases.get(ds, ds)]


def make_cfg(dataset, defense_label, eps, rho_m, seed, rounds, out_dir):
    preset = _ds_preset(dataset)
    defense_kwargs = BASELINES_VARIANTS[defense_label]
    name = (f"proto_{dataset}_{defense_label}_mimicry_eps{int(round(eps*100)):03d}"
            f"_r{int(round(rho_m*100)):02d}_s{seed:02d}")
    return ExperimentConfig(
        name=name,
        out_dir=str(out_dir),
        data=DataConfig(
            name=dataset, num_clients=preset["num_clients"],
            use_synthetic_fallback=True,
        ),
        model=ModelConfig(arch=preset["arch"], num_classes=preset["num_classes"]),
        federated=FederatedConfig(
            rounds=rounds, lr=1e-2, momentum=0.9,
            weight_decay=1e-4, sample_ratio=0.8,
            local_epochs_choices=[1, 2, 3],
        ),
        attack=AttackConfig(
            types=["mimicry"],
            malicious_ratio=float(rho_m),
            mimicry_epsilon=float(eps),
            mimicry_seed=int(seed),
            backdoor_target=preset.get("backdoor_target", 0),
        ),
        defense=DefenseConfig(**defense_kwargs),
        seed=int(seed),
    )


def main():
    ap = argparse.ArgumentParser(description="Krum-aware mimicry stress-test sweep")
    ap.add_argument("--datasets", nargs="+", default=["mitbih", "ciciomt", "ptbxl"])
    ap.add_argument("--defenses", nargs="+",
                    default=["fedavg", "median", "fedshield_v10", "fedshield_v10_a025"],
                    help="Keys in BASELINES_VARIANTS to evaluate.")
    ap.add_argument("--epsilons", nargs="+", type=float,
                    default=[0.10, 0.25, 0.50])
    ap.add_argument("--rho_m", nargs="+", type=float, default=[0.2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out_root", default="results/proto")
    args = ap.parse_args()

    print(f"[mimicry-sweep] device={'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"datasets={args.datasets} defenses={args.defenses} "
          f"epsilons={args.epsilons} rho_m={args.rho_m} seeds={args.seeds} "
          f"rounds={args.rounds}", flush=True)

    total = (len(args.datasets) * len(args.defenses) * len(args.epsilons)
             * len(args.rho_m) * len(args.seeds))
    print(f"[mimicry-sweep] total cells: {total}", flush=True)

    completed = 0
    skipped = 0
    failed = 0
    t_start = time.time()
    for ds in args.datasets:
        out_dir = Path(args.out_root) / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        for defense in args.defenses:
            for eps in args.epsilons:
                for rho in args.rho_m:
                    for seed in args.seeds:
                        cfg = make_cfg(ds, defense, eps, rho, seed,
                                       args.rounds, out_dir)
                        csv_path = out_dir / f"{cfg.name}_metrics.csv"
                        if csv_path.exists() and csv_path.stat().st_size > 100:
                            try:
                                df = pd.read_csv(csv_path)
                                if len(df) >= 1:
                                    skipped += 1
                                    last = df.iloc[-1]
                                    print(f"  CACHE {cfg.name}: "
                                          f"acc={last['acc']:.3f} asr={last['asr']:.3f} "
                                          f"score={last['defense_score']:.3f}",
                                          flush=True)
                                    continue
                            except Exception:
                                pass
                        t0 = time.time()
                        try:
                            FederatedTrainer(cfg).train()
                            df = pd.read_csv(csv_path)
                            last = df.iloc[-1]
                            dt = time.time() - t0
                            completed += 1
                            print(f"  RAN   {cfg.name}: "
                                  f"acc={last['acc']:.3f} asr={last['asr']:.3f} "
                                  f"score={last['defense_score']:.3f} ({dt:.1f}s)",
                                  flush=True)
                        except Exception as e:
                            failed += 1
                            import traceback
                            print(f"  FAIL  {cfg.name}: {type(e).__name__}: {e}",
                                  flush=True)
                            traceback.print_exc()
    total_dt = time.time() - t_start
    print(f"[mimicry-sweep] DONE: ran={completed} cached={skipped} failed={failed} "
          f"wall={total_dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
