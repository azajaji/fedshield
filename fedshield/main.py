"""Entry point.

Usage::

    python -m fedshield.main --config configs/mitbih_fedshield.yaml
    python -m fedshield.main --grid                     # run full grid
    python -m fedshield.main --aggregate                # build tables only
"""
from __future__ import annotations

import argparse
import itertools
import os
from typing import List

from .config import (
    AttackConfig,
    DataConfig,
    DefenseConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
)
from .evaluation import write_publication_tables
from .federated import FederatedTrainer


# --------------------------------------------------------------------------- #
#                              Built-in experiment grid
# --------------------------------------------------------------------------- #
DATASET_PRESETS = {
    # ``backdoor_target`` is the attacker's chosen target class. We pick the
    # class an attacker actually wants in real healthcare deployments:
    #   MIT-BIH:  flip abnormal beats -> "Normal" (hide arrhythmia)            -> 0
    #   WESAD:    flip stress windows -> "Baseline" (hide stress detection)   -> 0
    #   CIC-IoMT: flip attack flows   -> "Benign"  (hide intrusions)          -> 0
    # Explicitly set to avoid the silent argmax-of-majority-class fallback,
    # which would conflate a destroyed model with a defended one.
    # Composite IoMT threat model:
    #   Data poisoning      : label_flip, backdoor, traffic_injection
    #   Model poisoning     : sign_flip (Byzantine direction inversion)
    #                         scaling   (model-replacement, amplifies Δw_m)
    #                         noise_update (Gaussian on the update)
    #                         sybil     (colluding near-duplicate updates)
    "mitbih":  dict(num_clients=10, arch="ecg_cnn",       attacks=["label_flip", "backdoor", "sign_flip", "scaling"], backdoor_target=0),
    "wesad":   dict(num_clients=15, arch="wesad_cnnlstm", attacks=["label_flip", "noise_update", "scaling"],          backdoor_target=0),
    "ciciomt": dict(num_clients=12, arch="iomt_mlp",      attacks=["label_flip", "sign_flip", "scaling", "backdoor", "traffic_injection"], backdoor_target=0),
}
DEFENSES = ["fedavg", "krum", "trimmed_mean", "foolsgold", "fltrust", "fedshield"]
MAL_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 7, 13, 21, 42]


def make_cfg(dataset: str, defense: str, mal_ratio: float, seed: int,
             rounds: int = 30, alpha: float = 0.5,
             out_dir: str = "./results") -> ExperimentConfig:
    preset = DATASET_PRESETS[dataset]
    name = f"{dataset}_{defense}_r{int(mal_ratio*100):02d}_s{seed}"
    return ExperimentConfig(
        name=name,
        seed=seed,
        out_dir=out_dir,
        data=DataConfig(name=dataset, num_clients=preset["num_clients"], dirichlet_alpha=alpha),
        model=ModelConfig(arch=preset["arch"]),
        attack=AttackConfig(
            malicious_ratio=mal_ratio,
            types=preset["attacks"],
            backdoor_target=preset["backdoor_target"],
        ),
        defense=DefenseConfig(name=defense),
        federated=FederatedConfig(rounds=rounds),
        log_every=1,
    )


def run_grid(rounds: int = 30, datasets: List[str] | None = None,
             defenses: List[str] | None = None,
             mal_ratios: List[float] | None = None,
             seeds: List[int] | None = None,
             out_dir: str = "./results") -> None:
    datasets = datasets or list(DATASET_PRESETS)
    defenses = defenses or DEFENSES
    mal_ratios = mal_ratios or MAL_RATIOS
    seeds = seeds or SEEDS
    for ds, defn, mr, sd in itertools.product(datasets, defenses, mal_ratios, seeds):
        cfg = make_cfg(ds, defn, mr, sd, rounds=rounds, out_dir=out_dir)
        print(f"[run] {cfg.name}")
        FederatedTrainer(cfg).train()
    write_publication_tables(out_dir)


# --------------------------------------------------------------------------- #
#                              CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="Path to YAML/JSON config file")
    p.add_argument("--grid", action="store_true", help="Run the full grid")
    p.add_argument("--aggregate", action="store_true", help="Only build summary tables")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--out_dir", default="./results")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--defenses", nargs="*", default=None)
    p.add_argument("--mal_ratios", nargs="*", type=float, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    args = p.parse_args()

    if args.aggregate:
        out = write_publication_tables(args.out_dir)
        print("Wrote:", out)
        return

    if args.grid:
        run_grid(
            rounds=args.rounds,
            datasets=args.datasets,
            defenses=args.defenses,
            mal_ratios=args.mal_ratios,
            seeds=args.seeds,
            out_dir=args.out_dir,
        )
        return

    if args.config:
        cfg = ExperimentConfig.load(args.config)
        FederatedTrainer(cfg).train()
        return

    p.error("Provide --config or --grid or --aggregate")


if __name__ == "__main__":
    main()
