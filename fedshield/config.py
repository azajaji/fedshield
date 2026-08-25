"""Configuration objects.

Every experiment is fully specified by an :class:`ExperimentConfig`. YAML files in
``configs/`` are loaded into this dataclass tree so runs are 100% reproducible
from a single artefact. No hyper-parameter is hard-coded inside model or training
code; everything that varies must live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import os

try:
    import yaml  # optional
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    name: str = "mitbih"           # mitbih | wesad | ciciomt
    root: str = "./data"
    num_clients: int = 10
    dirichlet_alpha: float = 0.5
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    batch_size_choices: List[int] = field(default_factory=lambda: [32, 64])
    use_synthetic_fallback: bool = True   # generate synthetic data when real not present
    max_samples_per_client: int = 0       # 0 = no cap; subsamples each client's training set


@dataclass
class ModelConfig:
    arch: str = "ecg_cnn"           # ecg_cnn | wesad_cnnlstm | iomt_mlp
    num_classes: int = 5
    input_dim: Optional[int] = None  # filled by data loader
    seq_len: Optional[int] = None
    ae_bottleneck: int = 16


@dataclass
class AttackConfig:
    malicious_ratio: float = 0.2
    types: List[str] = field(default_factory=lambda: ["label_flip"])
    flip_map: Optional[Dict[int, int]] = None   # source -> target class
    sign_flip_lambda: float = 5.0
    # Scaling-based model poisoning (a.k.a. model replacement attack):
    # the malicious client submits a scaled-up local update
    # Δw'_m = scale_lambda * Δw_m to compensate for averaging in FedAvg.
    # Distinct from sign_flip (which inverts direction). Selected by
    # adding "scaling" to attack.types.
    scale_lambda: float = 10.0
    noise_sigma_factor: float = 0.5
    backdoor_target: Optional[int] = None
    backdoor_amplitude: float = 0.02
    backdoor_freq: float = 15.0
    # Krum-aware mimicry (semi-adaptive stress test). Active when
    # "mimicry" is in `types`. Each malicious update is replaced (server-side,
    # before defense aggregation) with mu_honest + eps * v_attack, where
    # v_attack is a per-(client, param) reproducible unit-norm vector and the
    # magnitude is scaled by the honest-centroid norm so that `eps` is the
    # fractional deviation from the honest centroid. See attacks.py
    # `apply_post_collection_mimicry`.
    mimicry_epsilon: float = 0.10
    mimicry_seed: int = 0


@dataclass
class DefenseConfig:
    name: str = "fedshield"          # fedavg | krum | multi_krum | trimmed_mean | foolsgold | fltrust | fedshield
    krum_f: int = 2
    multi_krum_m: int = 5
    trim_beta: float = 0.2
    fltrust_root_size: int = 200
    fedshield_alpha_ema: float = 0.6
    fedshield_threshold_k: float = 5.0       # AE alarm at mu + k*sigma; 5σ avoids spurious alarms
    fedshield_cosine_floor: float = 0.0
    # v2 efficiency/robustness improvements (defaults: ON)
    fedshield_use_geo_median: bool = False      # superseded by trim-mean reference
    fedshield_geo_median_iters: int = 3
    fedshield_softmax_temp: float = 0.3         # 0 disables (= ReLU cosine)
    fedshield_ae_lazy_phi: float = 0.5          # only run AE when cos < this threshold
    fedshield_use_trimmed_ref: bool = True      # legacy v2 flag (no-op in v4)
    fedshield_trim_beta: float = 0.2            # legacy v2 flag (no-op in v4)
    # FEDShield-v4 knobs
    fedshield_k_mad: float = 5.0                # MAD multiplier (legacy; norm test no longer hard-rejects)
    # Warmup default = 0. The 2-round FedAvg warmup was historically used so the
    # AE could establish a clean baseline, but ``warmup_autoencoders`` already
    # bootstraps the AE *before* federated training starts. Running FedAvg for
    # any rounds means malicious clients get full weight during those rounds —
    # at high rho_m this poisons the model beyond recovery before robust mode
    # activates.
    fedshield_warmup_rounds: int = 0
    fedshield_use_krum_aug: bool = True   # v6: Krum-with-AE selection (set False for v4 soft-cosine)
    # v7 server-side delta-pattern detector: tracks norm + cos-to-running-mean
    # of past accepted updates; flags new updates that fall outside the
    # historical distribution. Catches data-poisoning that passes Krum.
    # v7 server-AE: opt-in only; default off because it over-flagged honest
    # clients at clean rho=0 (z-score on naturally heterogeneous norms).
    fedshield_use_server_ae: bool = False
    fedshield_server_ae_warmup: int = 3
    fedshield_server_ae_z: float = 3.0
    # v8: Multi-Krum selection. 1 = top-1 vanilla (backwards-compatible);
    # m>1 averages the m clients with smallest Krum score. Robust to high
    # malicious ratios where the single-best pick coincidentally lands on
    # a flipped client. Active only when fedshield_use_krum_aug is True.
    fedshield_multikrum_m: int = 1
    # v8: hard-reject clients with cos(Δw_i, ref) < cos_floor BEFORE Krum
    # scoring. Default cos_floor=0 (any negative cosine → reject). Targets
    # high-ratio sign_flip where AE-alarm gating is too lenient.
    fedshield_hard_neg_cos: bool = False
    # v8: hard-reject clients whose ‖Δw_i‖ exceeds median + k_mad·MAD
    # (the "norm_outlier" set computed in Stage B). Currently those clients
    # are only norm-clipped; flipping this on adds them to the AE-penalty
    # set so Krum-aug ignores them entirely.
    fedshield_hard_norm_outlier: bool = False
    # v8: use a positive cosine threshold (e.g. 0.2) for hard rejection
    # when fedshield_hard_neg_cos is True. Defaults to 0.0 (reject only
    # strictly-negative cosines).
    fedshield_hard_cos_thresh: float = 0.0
    # v8: aggregation mode for the Krum-aug path.
    #   "topm"     — average of top-m clients by Krum score (current behaviour;
    #                m = fedshield_multikrum_m)
    #   "softmax"  — softmax-weighted average of ALL clients with weights
    #                ∝ exp(-krum_score / T), T = fedshield_softscore_temp
    #   "trimavg"  — coordinate-wise trimmed mean over the top-m clients with
    #                trim fraction = fedshield_aggreg_trim_beta
    #   "median"   — coordinate-wise median over the top-m clients
    fedshield_aggreg_mode: str = "topm"
    fedshield_softscore_temp: float = 0.3
    fedshield_aggreg_trim_beta: float = 0.3
    # v8: weight top-m members of the average by max(cos(Δw_i, ref), 0)
    # instead of equal/trust-EMA weights. Demotes survivors with weak
    # alignment to the consensus direction without hard-rejecting them.
    fedshield_topm_cos_weight: bool = False
    # v9: adaptive aggregation mode — switch between mean-of-top-mk2 and
    # median-of-top-mk_med based on the number of adversaries detected
    # this round. Detection signal: edge_alarm OR norm_outlier count.
    # If detected_adversaries < adaptive_threshold: use mk2 (clean mode)
    # Else: use median over mk_med (adversarial mode)
    fedshield_adaptive: bool = False
    fedshield_adaptive_threshold: int = 1
    fedshield_adaptive_clean_m: int = 2
    fedshield_adaptive_adv_m: int = 7
    # v10: ensemble of two aggregation strategies. Active when
    # fedshield_aggreg_mode == "ensemble". Computes
    #   alpha * mean(top-mk2) + (1-alpha) * median(top-mk_med)
    # alpha=0.5 averages equally; alpha=0.25 favours median;
    # alpha=0.75 favours mean.
    fedshield_ensemble_alpha: float = 0.5
    fedshield_ensemble_mk2_m: int = 2
    fedshield_ensemble_med_m: int = 7
    # Design-rationale knobs — used to validate each architectural choice
    # of Stage E with empirical comparisons (see §6 design ablation).
    #
    # combine_form: how the two aggregator outputs are fused.
    #   "convex"        — α·a + (1−α)·b (Stage E v10 default)
    #   "max"           — element-wise max(a, b)
    #   "min"           — element-wise min(a, b)
    #   "alarm_gated"   — a if no AE alarms in cohort else b
    fedshield_combine_form: str = "convex"
    # Aggregator choice for the two ensemble terms.
    #   "mean"   — uniform mean of the top-m clients (Stage E "a" term)
    #   "median" — per-coord median of the top-m clients
    #   "trim"   — per-coord trim-mean of the top-m clients (β = aggreg_trim_beta)
    #   "geomed" — geometric median over the top-m clients (Weiszfeld iter)
    fedshield_ens_a_kind: str = "mean"
    fedshield_ens_b_kind: str = "median"
    # Disable the Krum-score filter and use all m clients in both terms.
    # Used to validate that the Krum pre-filter is load-bearing.
    fedshield_disable_krum_filter: bool = False
    # Ablation toggles for the paper. Each disables one stage of FedShield
    # while leaving everything else at the v10 defaults.
    fedshield_ablate_edge_ae: bool = False     # A1: force u.edge_alarm = False
    fedshield_ablate_norm_clip: bool = False   # A2: skip Stage B clipping (raw flat to ref+cos)
    fedshield_ablate_trim_ref: bool = False    # A3: use mean-of-all as reference (not trim-median)
    # A4 = use fedshield_ensemble_alpha=1.0 (mean only) or =0.0 (median only)
    fedshield_ablate_buggy_f: bool = False     # A5: re-enable f=int(0.4*m_count) bug


@dataclass
class FederatedConfig:
    rounds: int = 100
    sample_ratio: float = 0.6
    local_epochs_choices: List[int] = field(default_factory=lambda: [1, 2, 3])
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class ExperimentConfig:
    name: str = "exp"
    seed: int = 42
    device: str = "auto"
    out_dir: str = "./results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    log_every: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "exp"),
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
            out_dir=d.get("out_dir", "./results"),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            attack=AttackConfig(**d.get("attack", {})),
            defense=DefenseConfig(**d.get("defense", {})),
            federated=FederatedConfig(**d.get("federated", {})),
            log_every=d.get("log_every", 1),
        )

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            text = f.read()
        if path.endswith((".yml", ".yaml")):
            if yaml is None:
                raise RuntimeError("PyYAML required to load YAML configs.")
            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
        return cls.from_dict(d)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            if path.endswith((".yml", ".yaml")) and yaml is not None:
                yaml.safe_dump(self.to_dict(), f)
            else:
                json.dump(self.to_dict(), f, indent=2)
