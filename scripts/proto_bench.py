"""Fast micro-bench for FedShield prototyping.

Runs a tiny attack-matrix slice (the diagnostic cells where FedShield currently
loses to Krum) and prints a head-to-head delta table. Intended to complete in
under a minute on a GPU so prototype tweaks can be iterated quickly.

Usage:
    python -m scripts.proto_bench
    python -m scripts.proto_bench --defenses krum fedshield --rounds 8

Add a new FedShield variant by listing its name (e.g. fedshield_v8) under
``--defenses`` AFTER wiring it into ``defenses.py``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def _keep_system_awake() -> None:
    """Prevent Windows from sleeping/hibernating during the bench.

    Uses kernel32.SetThreadExecutionState — works without admin.
        ES_CONTINUOUS         keep the request active until process exit
        ES_SYSTEM_REQUIRED    block system sleep
        ES_AWAYMODE_REQUIRED  keep running in "away mode" even if user closes
                              the lid (laptop) or hits the power button
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ES_CONTINUOUS        = 0x80000000
        ES_SYSTEM_REQUIRED   = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ctypes.c_uint(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        )
    except Exception as e:
        print(f"[proto_bench] keep-awake setup failed (ok to ignore): {e}", flush=True)

from fedshield.config import (
    AttackConfig, DataConfig, DefenseConfig, ExperimentConfig,
    FederatedConfig, ModelConfig,
)
from fedshield.federated import FederatedTrainer


DIAGNOSTIC_CELLS = [
    # (attack_types, ratio, label)
    # 4-cell "diagnostic" subset used for fast iteration + ablation table.
    (["sign_flip"], 0.0, "clean"),
    (["sign_flip"], 0.4, "sign_flip@0.4"),
    (["backdoor", "scaling"], 0.2, "bd+scale@0.2"),
    (["label_flip", "backdoor", "sign_flip", "scaling"], 0.4, "composite@0.4"),
]

FULL_CELLS = [
    # Paper-style matrix: 8 attack types × 4 ratios + 1 clean cell = 33 cells.
    # Expanded from 17 to 33 cells by adding ratios 0.1 and 0.3.
    (["sign_flip"], 0.0, "clean"),
    (["sign_flip"], 0.1, "sign_flip@0.1"),
    (["sign_flip"], 0.2, "sign_flip@0.2"),
    (["sign_flip"], 0.3, "sign_flip@0.3"),
    (["sign_flip"], 0.4, "sign_flip@0.4"),
    (["label_flip"], 0.1, "label_flip@0.1"),
    (["label_flip"], 0.2, "label_flip@0.2"),
    (["label_flip"], 0.3, "label_flip@0.3"),
    (["label_flip"], 0.4, "label_flip@0.4"),
    (["backdoor"], 0.1, "backdoor@0.1"),
    (["backdoor"], 0.2, "backdoor@0.2"),
    (["backdoor"], 0.3, "backdoor@0.3"),
    (["backdoor"], 0.4, "backdoor@0.4"),
    (["scaling"], 0.1, "scaling@0.1"),
    (["scaling"], 0.2, "scaling@0.2"),
    (["scaling"], 0.3, "scaling@0.3"),
    (["scaling"], 0.4, "scaling@0.4"),
    (["noise_update"], 0.1, "noise@0.1"),
    (["noise_update"], 0.2, "noise@0.2"),
    (["noise_update"], 0.3, "noise@0.3"),
    (["noise_update"], 0.4, "noise@0.4"),
    (["sign_flip", "scaling"], 0.1, "sign+scale@0.1"),
    (["sign_flip", "scaling"], 0.2, "sign+scale@0.2"),
    (["sign_flip", "scaling"], 0.3, "sign+scale@0.3"),
    (["sign_flip", "scaling"], 0.4, "sign+scale@0.4"),
    (["backdoor", "scaling"], 0.1, "bd+scale@0.1"),
    (["backdoor", "scaling"], 0.2, "bd+scale@0.2"),
    (["backdoor", "scaling"], 0.3, "bd+scale@0.3"),
    (["backdoor", "scaling"], 0.4, "bd+scale@0.4"),
    (["label_flip", "backdoor", "sign_flip", "scaling"], 0.1, "composite@0.1"),
    (["label_flip", "backdoor", "sign_flip", "scaling"], 0.2, "composite@0.2"),
    (["label_flip", "backdoor", "sign_flip", "scaling"], 0.3, "composite@0.3"),
    (["label_flip", "backdoor", "sign_flip", "scaling"], 0.4, "composite@0.4"),
]

# Focused ρ-sweep panel for the "robustness vs malicious-ratio" figure.
# 1 attack type × 6 ratios = 6 cells. Used with FedShield + Krum only.
RHO_SWEEP_CELLS = [
    (["sign_flip"], 0.0, "sign_flip@0.0"),
    (["sign_flip"], 0.1, "sign_flip@0.1"),
    (["sign_flip"], 0.2, "sign_flip@0.2"),
    (["sign_flip"], 0.3, "sign_flip@0.3"),
    (["sign_flip"], 0.4, "sign_flip@0.4"),
    (["sign_flip"], 0.5, "sign_flip@0.5"),
]


# Variants under test. Display name → DefenseConfig kwargs.
# Always include 'krum' (baseline) and 'fedshield' (current main).
# Add new prototypes here; they appear automatically in the head-to-head table.
def _ens(alpha, m2=2, mm=7, **extra):
    base = dict(name="fedshield", fedshield_aggreg_mode="ensemble",
                fedshield_ensemble_alpha=alpha,
                fedshield_ensemble_mk2_m=m2,
                fedshield_ensemble_med_m=mm)
    base.update(extra)
    return base


# === Sweep 5: Design rationale — empirical justification for each ===
# === architectural choice in Stage E (mean+median ensemble).      ===
DESIGN_RATIONALE_VARIANTS = {
    # Single aggregators (single-component baselines)
    "single_mean_top2":    _ens(1.00, m2=2, mm=7),                                      # = mean(top-2) only
    "single_median_top7":  _ens(0.00, m2=2, mm=7),                                      # = median(top-7) only
    "single_trim_top7":    _ens(0.00, m2=2, mm=7,
                                 fedshield_ens_b_kind="trim",
                                 fedshield_aggreg_trim_beta=0.3),                       # = trim-mean(top-7) only
    # Pairwise alternative components at α = 0.50 (equal mix to isolate kind effect)
    "ens_mean+trim":       _ens(0.50, m2=2, mm=7,
                                 fedshield_ens_a_kind="mean",
                                 fedshield_ens_b_kind="trim",
                                 fedshield_aggreg_trim_beta=0.3),
    "ens_median+trim":     _ens(0.50, m2=7, mm=7,
                                 fedshield_ens_a_kind="median",
                                 fedshield_ens_b_kind="trim",
                                 fedshield_aggreg_trim_beta=0.3),
    "ens_mean+geomed":     _ens(0.50, m2=2, mm=7,
                                 fedshield_ens_a_kind="mean",
                                 fedshield_ens_b_kind="geomed"),
    # Cascade vs parallel — Bulyan is the canonical cascaded baseline
    "bulyan_classic":      dict(name="bulyan", krum_f=2),
    # Selection-rule ablation — drop the Krum pre-filter
    "ens_no_krum_filter":  _ens(0.90, m2=2, mm=7,
                                 fedshield_disable_krum_filter=True),
    # Combination-form ablation
    "ens_max":             _ens(0.50, m2=2, mm=7, fedshield_combine_form="max"),
    "ens_min":             _ens(0.50, m2=2, mm=7, fedshield_combine_form="min"),
    "ens_alarm_gated":     _ens(0.50, m2=2, mm=7, fedshield_combine_form="alarm_gated"),
    # The chosen design (control)
    "fedshield_v10":       _ens(0.90, m2=2, mm=7),
}

# Dataset presets — match fedshield/main.py DATASET_PRESETS so that the
# bench can sweep across MIT-BIH / WESAD / CICIoMT with one CLI flag.
DATASET_PRESETS = {
    "mitbih":  dict(num_clients=10, arch="ecg_cnn",       num_classes=5,
                    backdoor_target=0),
    "wesad":   dict(num_clients=15, arch="wesad_cnnlstm", num_classes=4,
                    backdoor_target=0),
    "ciciomt": dict(num_clients=12, arch="iomt_mlp",      num_classes=6,
                    backdoor_target=0),
    "ptbxl":   dict(num_clients=10, arch="ecg_cnn",       num_classes=5,
                    backdoor_target=0),
    "physionet2017": dict(num_clients=10, arch="ecg_cnn", num_classes=4,
                          backdoor_target=1),  # AF (minority class) — meaningful ASR
                                               # (target=0 Normal would be ~59% of test
                                               # so ASR conflates with class collapse)
    "physionet2020": dict(num_clients=4,  arch="ecg_cnn", num_classes=5,
                          backdoor_target=1),  # 4 native sites (INCART dropped:
                                               # too few records); backdoor MI
                                               # (minority) since NORM is majority.
}

# === Sweep 1: Headline matrix — FedShield vs all baselines (Table 1) ===
BASELINES_VARIANTS = {
    "fedavg":              dict(name="fedavg"),
    "krum":                dict(name="krum"),
    "multi_krum":          dict(name="multi_krum"),
    "trimmed_mean":        dict(name="trimmed_mean"),
    "foolsgold":           dict(name="foolsgold"),
    "fltrust":             dict(name="fltrust"),
    "median":              dict(name="median"),
    "rfa":                 dict(name="rfa"),
    "dnc":                 dict(name="dnc"),
    "fedshield_v10":       _ens(0.90),
    # Dataset-tuned variants per the per-dataset alpha sensitivity sweep:
    # MIT-BIH and CIC-IoMT favour magnitude-dominated mix (alpha=0.90);
    # PTB-XL's per-patient 12-lead partition favours median-dominated (alpha=0.25).
    "fedshield_v10_a025":  _ens(0.25),
}

# === Sweep 4: Tuned baselines — addresses fairness concern that we tuned ===
# === FedShield's knobs but ran baselines at default. Sweeps each baseline's ===
# === main hyperparameter to find its best setting per dataset.            ===
TUNED_BASELINES_VARIANTS = {
    # Krum: f sweep
    "krum_f1":          dict(name="krum", krum_f=1),
    "krum_f2":          dict(name="krum", krum_f=2),
    "krum_f3":          dict(name="krum", krum_f=3),
    # Multi-Krum: m sweep at f=2
    "multi_krum_m3":    dict(name="multi_krum", krum_f=2, multi_krum_m=3),
    "multi_krum_m5":    dict(name="multi_krum", krum_f=2, multi_krum_m=5),
    "multi_krum_m7":    dict(name="multi_krum", krum_f=2, multi_krum_m=7),
    # Trimmed-Mean: trim_beta sweep
    "trimmed_b10":      dict(name="trimmed_mean", trim_beta=0.10),
    "trimmed_b20":      dict(name="trimmed_mean", trim_beta=0.20),
    "trimmed_b30":      dict(name="trimmed_mean", trim_beta=0.30),
    # FLTrust: root_size sweep
    "fltrust_r100":     dict(name="fltrust", fltrust_root_size=100),
    "fltrust_r200":     dict(name="fltrust", fltrust_root_size=200),
    "fltrust_r400":     dict(name="fltrust", fltrust_root_size=400),
    # FedShield (control)
    "fedshield_v10":    _ens(0.90),
}

# === Sweep 2: Ablation matrix (Table 2) ===
ABLATION_VARIANTS = {
    "fedshield_v10":   _ens(0.90),
    "A1_no_edge_ae":   _ens(0.90, fedshield_ablate_edge_ae=True),
    "A2_no_norm_clip": _ens(0.90, fedshield_ablate_norm_clip=True),
    "A3_mean_ref":     _ens(0.90, fedshield_ablate_trim_ref=True),
    "A4a_mean_only":   _ens(1.00),                    # α=1.0 → pure mean(top-2)
    "A4b_median_only": _ens(0.00),                    # α=0.0 → pure median(top-7)
    "A5_buggy_f":      _ens(0.90, fedshield_ablate_buggy_f=True),
}

# === Sweep 3: Sensitivity (Table 3) ===
SENSITIVITY_VARIANTS = {
    # alpha sweep
    "S_a000":  _ens(0.00), "S_a025": _ens(0.25), "S_a050": _ens(0.50),
    "S_a065":  _ens(0.65), "S_a075": _ens(0.75), "S_a085": _ens(0.85),
    "S_a090":  _ens(0.90), "S_a095": _ens(0.95), "S_a100": _ens(1.00),
    # (m2, mm) sweep at alpha=0.90
    "S_m2_3":  _ens(0.90, mm=3), "S_m2_5": _ens(0.90, mm=5),
    "S_m2_9":  _ens(0.90, mm=9), "S_m3_7": _ens(0.90, m2=3),
}

# Default = headline matrix (the paper's Table 1)
VARIANTS = BASELINES_VARIANTS


def make_cfg(variant: str, attack_types, mr: float, rounds: int, seed: int,
             dataset: str = "mitbih", num_clients: int = None) -> ExperimentConfig:
    defense_kwargs = VARIANTS[variant]
    preset = DATASET_PRESETS[dataset]
    n_clients = num_clients if num_clients is not None else preset["num_clients"]
    # Filename includes dataset and client count so cross-dataset / cohort
    # runs don't clobber each other.
    cohort_tag = "" if num_clients is None else f"_n{n_clients:02d}"
    name = (f"proto_{dataset}_{variant}_{'+'.join(attack_types) or 'clean'}"
            f"_r{int(mr*100):02d}_s{seed:02d}{cohort_tag}")
    return ExperimentConfig(
        name=name, seed=seed, device="auto", out_dir=f"./results/proto/{dataset}",
        data=DataConfig(
            name=dataset, num_clients=n_clients,
            use_synthetic_fallback=False, batch_size_choices=[32, 64],
        ),
        model=ModelConfig(arch=preset["arch"], num_classes=preset["num_classes"],
                          ae_bottleneck=16),
        attack=AttackConfig(
            malicious_ratio=mr, types=list(attack_types),
            backdoor_target=preset["backdoor_target"], sign_flip_lambda=5.0,
        ),
        defense=DefenseConfig(**defense_kwargs),
        federated=FederatedConfig(
            rounds=rounds, sample_ratio=0.8,
            local_epochs_choices=[1, 2, 3], lr=0.01, momentum=0.9, weight_decay=1e-4,
        ),
        log_every=max(1, rounds - 1),  # log only at the end for speed
    )


def run_one(variant: str, attack_types, mr: float, rounds: int, seed: int,
            dataset: str = "mitbih", num_clients: int = None) -> dict:
    cfg = make_cfg(variant, attack_types, mr, rounds, seed, dataset=dataset,
                   num_clients=num_clients)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg.out_dir) / f"{cfg.name}_metrics.csv"
    # Resume: if a complete CSV exists for this exact (variant, cell, seed),
    # skip the training and just load the saved metrics.
    cached = False
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            if len(existing) >= 1:
                cached = True
        except Exception:
            pass
    if cached:
        dt = 0.0
    else:
        t0 = time.time()
        try:
            FederatedTrainer(cfg).train()
        except Exception as e:
            # Per-cell failure (CUDA OOM, transient, etc.) MUST NOT kill the
            # whole sweep. Log, return a NaN-scored sentinel row, and move on.
            # Re-running the bench will retry this cell from cache miss.
            import traceback
            print(f"  FAIL  {cfg.name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return dict(
                variant=variant, attack="+".join(attack_types) or "clean", mr=mr,
                dataset=dataset, acc=float("nan"), asr=float("nan"),
                score=float("nan"), seconds=float("nan"),
                latency_edge_ms=float("nan"), latency_server_ms=float("nan"),
                comm_bytes_per_client=float("nan"), edge_ram_mb=float("nan"),
            )
        dt = time.time() - t0
    last = pd.read_csv(csv_path).iloc[-1]
    print(f"  {'CACHE' if cached else 'RAN  '} {cfg.name}: score={float(last['defense_score']):.3f} ({dt:.1f}s)", flush=True)
    return dict(
        variant=variant, attack="+".join(attack_types) or "clean", mr=mr,
        dataset=dataset,
        acc=float(last["acc"]), asr=float(last["asr"]),
        score=float(last["defense_score"]), seconds=dt,
        # Cost-table fields (logged by FederatedTrainer; passed through so the
        # report can produce a per-defense cost summary alongside score).
        latency_edge_ms=float(last.get("latency_edge_ms", float("nan"))),
        latency_server_ms=float(last.get("latency_server_ms", float("nan"))),
        comm_bytes_per_client=float(last.get("comm_bytes_per_client", float("nan"))),
        edge_ram_mb=float(last.get("edge_ram_mb", float("nan"))),
    )


def main() -> None:
    _keep_system_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",
                    choices=["headline", "ablation", "sensitivity", "tuned_baselines",
                             "rho_sweep", "cohort", "design_rationale"],
                    default=None,
                    help="Predefined sweep set. Overrides --variants if given.")
    ap.add_argument("--variants", nargs="+",
                    default=list(VARIANTS.keys()),
                    help="Variants to compare. First is the baseline.")
    ap.add_argument("--cell_set", choices=["diagnostic", "full", "rho_sweep"], default="diagnostic",
                    help="diagnostic = 4 cells (fast); full = 17-cell paper matrix; rho_sweep = sign_flip x 6 ratios.")
    ap.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()) + ["physionet2017", "physionet2020"], default="mitbih")
    ap.add_argument("--num_clients", type=int, default=None,
                    help="Override the dataset preset's num_clients (for cohort-size sensitivity).")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="Seeds to average over. More seeds = lower variance, longer runtime.")
    args = ap.parse_args()
    cells = {"full": FULL_CELLS, "rho_sweep": RHO_SWEEP_CELLS}.get(args.cell_set, DIAGNOSTIC_CELLS)

    # Apply sweep selection by mutating the module-level VARIANTS via globals().
    # Using `global VARIANTS` here would conflict with the argparse default that
    # reads VARIANTS.keys() above (Python treats both as the same scope).
    if args.sweep == "headline":
        globals()["VARIANTS"] = BASELINES_VARIANTS
        args.variants = list(BASELINES_VARIANTS.keys())
    elif args.sweep == "ablation":
        globals()["VARIANTS"] = ABLATION_VARIANTS
        args.variants = list(ABLATION_VARIANTS.keys())
    elif args.sweep == "sensitivity":
        sens = {**SENSITIVITY_VARIANTS, "krum": dict(name="krum")}
        globals()["VARIANTS"] = sens
        args.variants = list(sens.keys())
    elif args.sweep == "tuned_baselines":
        globals()["VARIANTS"] = TUNED_BASELINES_VARIANTS
        args.variants = list(TUNED_BASELINES_VARIANTS.keys())
    elif args.sweep == "rho_sweep":
        # Focused breakdown-curve panel: just FedShield + Krum on sign_flip × 6 ratios.
        rho_set = {"krum": dict(name="krum"), "fedshield_v10": _ens(0.90)}
        globals()["VARIANTS"] = rho_set
        args.variants = list(rho_set.keys())
        args.cell_set = "rho_sweep"
    elif args.sweep == "cohort":
        # Cohort-size sensitivity: FedShield + Krum only, diagnostic cells.
        # Caller must pass --num_clients separately (we run one bench per N).
        cohort_set = {"krum": dict(name="krum"), "fedshield_v10": _ens(0.90)}
        globals()["VARIANTS"] = cohort_set
        args.variants = list(cohort_set.keys())
    elif args.sweep == "design_rationale":
        globals()["VARIANTS"] = DESIGN_RATIONALE_VARIANTS
        args.variants = list(DESIGN_RATIONALE_VARIANTS.keys())

    rows = []
    for v in args.variants:
        for attack_types, mr, _label in cells:
            for seed in args.seeds:
                rows.append({
                    **run_one(v, attack_types, mr, args.rounds, seed,
                              dataset=args.dataset, num_clients=args.num_clients),
                    "seed": seed,
                })

    df = pd.DataFrame(rows)

    if len(args.seeds) > 1:
        print("\n=== mean ± std over seeds ===")
        agg = (df.groupby(["variant", "attack", "mr"])["score"]
                 .agg(["mean", "std", "count"])
                 .reset_index())
        print(agg.to_string(index=False,
              float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x))
    else:
        print("\n=== raw scores ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== head-to-head mean score (rows = cells, cols = variants) ===")
    base = args.variants[0]
    pivot = df.pivot_table(index=["attack", "mr"], columns="variant", values="score", aggfunc="mean")
    if base in pivot.columns:
        for v in args.variants[1:]:
            if v in pivot.columns:
                pivot[f"{v}-{base}"] = pivot[v] - pivot[base]
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\n=== aggregate leaderboard (sum of per-cell mean scores; max = #cells) ===")
    cell_means = df.groupby(["variant", "attack", "mr"])["score"].mean().reset_index()
    sums = cell_means.groupby("variant")["score"].sum().reset_index().rename(columns={"score": "sum_score"})
    sums = sums.sort_values("sum_score", ascending=False)
    print(sums.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Paired t-test: FedShield (or first variant containing 'fedshield') vs each
    # other variant, paired across (attack, mr, seed) tuples. Reviewers ask for
    # this to be sure the headline gain is statistically significant.
    if len(args.seeds) >= 3:
        try:
            from scipy import stats as _stats
        except ImportError:
            _stats = None
        if _stats is not None:
            target = next((v for v in args.variants if "fedshield" in v.lower()), None)
            if target is not None and len(args.variants) > 1:
                print(f"\n=== paired t-test ({target} vs each baseline; per-(cell, seed) pairs) ===")
                t_rows = []
                for v in args.variants:
                    if v == target:
                        continue
                    pivot = df.pivot_table(
                        index=["attack", "mr", "seed"], columns="variant",
                        values="score", aggfunc="first"
                    )
                    if target in pivot.columns and v in pivot.columns:
                        pair = pivot[[target, v]].dropna()
                        if len(pair) >= 3:
                            t, p = _stats.ttest_rel(pair[target], pair[v])
                            mean_diff = float((pair[target] - pair[v]).mean())
                            t_rows.append((v, mean_diff, float(t), float(p), len(pair)))
                if t_rows:
                    # ASCII-only formatting (cp1252 stdout chokes on Greek delta).
                    print(f"{'baseline':<22} {'mean_diff':>10} {'t':>8} {'p':>10} {'n_pairs':>8}")
                    for v, d, t, p, n in t_rows:
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"{v:<22} {d:+10.3f} {t:+8.2f} {p:10.4f} {n:>8}  {sig}")

    # Cost summary — single row per variant, averaged across all runs.
    cost_cols = ["latency_edge_ms", "latency_server_ms", "comm_bytes_per_client", "edge_ram_mb"]
    if any(c in df.columns for c in cost_cols):
        print("\n=== cost summary (per-defense average across all runs) ===")
        cost = df.groupby("variant")[cost_cols].mean().reset_index()
        # comm_bytes → KB for readability
        if "comm_bytes_per_client" in cost.columns:
            cost["comm_kb_per_client"] = cost["comm_bytes_per_client"] / 1024.0
            cost = cost.drop(columns=["comm_bytes_per_client"])
        print(cost.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    total_s = df["seconds"].sum()
    print(f"\ntotal wall time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
