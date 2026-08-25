"""Attack implementations.

Two layers of attacks:
  * **Data-layer**   transform a client's local dataset *before* training.
  * **Update-layer** transform a client's outgoing pseudo-gradient.

Each attack is dataset-aware to keep semantics valid (e.g., signal-level
backdoors are restricted to time-series datasets).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .config import AttackConfig
from .data_loader import ArrayDataset


# --------------------------------------------------------------------------- #
#                              Selecting malicious clients
# --------------------------------------------------------------------------- #
def select_malicious_clients(
    num_clients: int, ratio: float, rng: np.random.Generator
) -> List[int]:
    n_mal = int(round(ratio * num_clients))
    return sorted(rng.choice(num_clients, size=n_mal, replace=False).tolist())


# --------------------------------------------------------------------------- #
#                              Data-layer attacks
# --------------------------------------------------------------------------- #
def label_flip(ds: ArrayDataset, flip_map: Dict[int, int]) -> ArrayDataset:
    new_y = ds.y.numpy().copy()
    for src, tgt in flip_map.items():
        new_y[ds.y.numpy() == src] = tgt
    return ArrayDataset(ds.X.numpy(), new_y)


def gaussian_noise_relabel(
    ds: ArrayDataset, sigma: float, num_classes: int, rng: np.random.Generator
) -> ArrayDataset:
    X = ds.X.numpy().copy()
    X += rng.normal(scale=sigma, size=X.shape).astype(np.float32)
    y = rng.integers(0, num_classes, size=ds.y.shape[0])
    return ArrayDataset(X, y)


def signal_backdoor(
    ds: ArrayDataset,
    target_class: int,
    amplitude: float,
    freq_hz: float,
    fs: float = 360.0,
) -> ArrayDataset:
    """Inject a low-amplitude sinusoidal pulse into the trailing portion of the
    signal and re-label to the attacker target class. Default parameters match
    the MIT-BIH context (fs=360 Hz)."""
    X = ds.X.numpy().copy()
    if X.ndim < 2:
        return ds       # no temporal axis → skip
    T = X.shape[-1]
    n_inject = max(8, T // 6)
    t = np.arange(n_inject) / fs
    pulse = amplitude * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    # broadcast across channels
    X[..., -n_inject:] += pulse
    y = np.full_like(ds.y.numpy(), fill_value=target_class)
    return ArrayDataset(X, y)


def network_feature_stamp(
    ds: ArrayDataset, target_class: int, feat_indices: Iterable[int],
    stamp_value: float = 0.05, mode: str = "additive"
) -> ArrayDataset:
    """Stealthy backdoor: small additive perturbation on a few features.

    ``mode='additive'`` adds ``stamp_value`` to each indexed feature (works on
    standardised data because perturbations stay within natural variance).
    ``mode='overwrite'`` replaces feature values; only useful for unit-tests.
    """
    X = ds.X.numpy().copy()
    for fi in feat_indices:
        if mode == "overwrite":
            X[..., fi] = stamp_value
        else:
            X[..., fi] = X[..., fi] + stamp_value
    y = np.full_like(ds.y.numpy(), fill_value=target_class)
    return ArrayDataset(X, y)


def malicious_traffic_injection(
    clean: ArrayDataset, n_inject: int, num_classes: int, rng: np.random.Generator
) -> ArrayDataset:
    """Append crafted attack-like rows mislabeled as benign (class 0)."""
    if n_inject <= 0:
        return clean
    feat_dim = clean.X.shape[1:]
    extra_X = rng.normal(loc=0.0, scale=2.0, size=(n_inject, *feat_dim)).astype(np.float32)
    extra_y = np.zeros(n_inject, dtype=np.int64)         # benign target
    X = np.concatenate([clean.X.numpy(), extra_X], axis=0)
    y = np.concatenate([clean.y.numpy(), extra_y], axis=0)
    return ArrayDataset(X, y)


# --------------------------------------------------------------------------- #
#                              Update-layer attacks
# --------------------------------------------------------------------------- #
def _is_float(t: torch.Tensor) -> bool:
    """Update-layer attacks must skip non-float tensors (e.g., BatchNorm
    integer running-stat counters), otherwise torch.randn_like and float
    multiplication raise NotImplementedError on int dtypes."""
    return t.is_floating_point()


def sign_flip_update(delta: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    return {k: (-scale * v if _is_float(v) else v) for k, v in delta.items()}


def scaling_update(delta: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    """Scaling-based model poisoning (a.k.a. model replacement attack).

    The malicious client submits ``Δw'_m = scale * Δw_m``, where ``Δw_m`` is
    the locally-computed update. The scale factor is chosen to compensate
    for averaging in FedAvg: with ``m`` clients in the cohort and one
    malicious participant, choosing ``scale = m`` causes that single
    update to dominate aggregation (Bagdasaryan et al., 2020).

    Distinct from ``sign_flip`` (which inverts direction) and from the
    data-layer ``label_flip`` (which corrupts labels): here only the
    magnitude of the submitted update is changed; the direction is the
    direction the local data dictates.

    Taxonomy: Poisoning Attacks → Model Poisoning Attacks →
              Scaling-Based / Model Replacement Attacks.
    """
    return {k: (scale * v if _is_float(v) else v) for k, v in delta.items()}


def gaussian_noise_update(
    delta: Dict[str, torch.Tensor], sigma: float
) -> Dict[str, torch.Tensor]:
    return {
        k: (v + torch.randn_like(v) * sigma if _is_float(v) else v)
        for k, v in delta.items()
    }


def sybil_collude(
    delta: Dict[str, torch.Tensor], jitter: float = 1e-3
) -> Dict[str, torch.Tensor]:
    """Used by colluding Sybils — return *near-identical* updates with tiny
    Gaussian jitter to defeat exact-duplicate filters."""
    return {
        k: (v + torch.randn_like(v) * jitter if _is_float(v) else v)
        for k, v in delta.items()
    }


# --------------------------------------------------------------------------- #
#                              Dispatcher
# --------------------------------------------------------------------------- #
@dataclass
class AttackPlan:
    malicious_clients: List[int]
    cfg: AttackConfig

    def is_malicious(self, client_id: int) -> bool:
        return client_id in self.malicious_clients


def apply_data_attack(
    ds: ArrayDataset,
    plan: AttackPlan,
    num_classes: int,
    rng: np.random.Generator,
    dataset_name: str,
) -> ArrayDataset:
    cfg = plan.cfg
    out = ds
    name = dataset_name.lower()
    if "label_flip" in cfg.types:
        if cfg.flip_map:
            out = label_flip(out, {int(k): int(v) for k, v in cfg.flip_map.items()})
        else:
            # default per-dataset flip
            default_map = {
                "mitbih": {0: 1, 1: 0},          # normal <-> ventricular
                "wesad": {1: 2, 2: 1},           # stress <-> baseline
                "ciciomt": {0: 1, 1: 0},         # benign <-> attack
            }
            out = label_flip(out, default_map.get(name, {0: 1}))
    if "noise" in cfg.types or "gaussian_noise" in cfg.types:
        out = gaussian_noise_relabel(out, sigma=0.5, num_classes=num_classes, rng=rng)
    if "backdoor" in cfg.types:
        if name == "ciciomt":
            out = network_feature_stamp(
                out,
                target_class=cfg.backdoor_target if cfg.backdoor_target is not None else 0,
                feat_indices=[0, 1, 2],
                stamp_value=cfg.backdoor_amplitude,   # small additive perturbation
                mode="additive",
            )
        else:
            out = signal_backdoor(
                out,
                target_class=cfg.backdoor_target if cfg.backdoor_target is not None else 0,
                amplitude=cfg.backdoor_amplitude,
                freq_hz=cfg.backdoor_freq,
            )
    if "traffic_injection" in cfg.types and name == "ciciomt":
        out = malicious_traffic_injection(
            out, n_inject=max(50, len(out) // 4), num_classes=num_classes, rng=rng
        )
    return out


def apply_update_attack(
    delta: Dict[str, torch.Tensor],
    plan: AttackPlan,
) -> Dict[str, torch.Tensor]:
    cfg = plan.cfg
    out = delta
    if "sign_flip" in cfg.types:
        out = sign_flip_update(out, scale=cfg.sign_flip_lambda)
    if "scaling" in cfg.types or "model_replacement" in cfg.types:
        # Scaling-based / model replacement attack: amplify the submitted
        # update by ``scale_lambda`` to compensate for FedAvg averaging.
        out = scaling_update(out, scale=cfg.scale_lambda)
    if "noise_update" in cfg.types:
        # match scale of typical update std (over float tensors only)
        flats = [v.reshape(-1).float() for v in delta.values() if _is_float(v)]
        if flats:
            flat = torch.cat(flats)
            sigma = cfg.noise_sigma_factor * flat.std().item()
            out = gaussian_noise_update(out, sigma=sigma)
    if "sybil" in cfg.types:
        out = sybil_collude(out, jitter=1e-3)
    # NOTE: "mimicry" is handled server-side after all client updates are
    # collected (it needs mu_honest). See `apply_post_collection_mimicry`
    # invoked from FederatedTrainer; the per-client pass is a no-op for it.
    return out


# --------------------------------------------------------------------------- #
#                   Server-side semi-adaptive mimicry attack
# --------------------------------------------------------------------------- #
def apply_post_collection_mimicry(updates, plan: AttackPlan, round_idx: int):
    """Krum-aware mimicry stress test.

    Replaces every malicious client's submitted delta with

        Delta_mal = mu_honest + eps * ||mu_honest|| * v_attack

    where mu_honest is the per-parameter mean of honest clients in this
    round and v_attack is a per-(client, parameter, round) reproducible
    unit-norm random direction. This is a *semi-adaptive* test: it assumes
    the attacker can see the geometry of honest updates this round, which
    is stronger than the non-adaptive headline threat model but weaker than
    full optimisation against the FedShield decision boundary.

    Activated when ``"mimicry"`` is in ``plan.cfg.types``. Otherwise the
    function is a no-op. Returns the (possibly modified) ``updates`` list
    and a diagnostics dict with the achieved per-client centroid distance
    so the caller can log survival statistics.

    Parameters
    ----------
    updates : List[ClientUpdate]
        Per-client updates already produced by ``_client_round``. Must be
        mutated in place (or replaced) before aggregation.
    plan : AttackPlan
        Attack plan with ``cfg.mimicry_epsilon`` and ``cfg.mimicry_seed``.
    round_idx : int
        Current federated round, used to seed the adversarial direction.

    Returns
    -------
    (updates, diag) where diag carries ``mu_honest_norm`` and the per
    malicious-client centroid-distance metric.
    """
    cfg = plan.cfg
    if "mimicry" not in cfg.types:
        return updates, {}

    honest = [u for u in updates if not u.is_malicious]
    malicious = [u for u in updates if u.is_malicious]
    if not honest or not malicious:
        return updates, {"mu_honest_norm": 0.0}

    # Per-parameter honest centroid (float-only; non-float tensors copied
    # from the first honest client to keep BN counters etc. valid).
    mu_honest: Dict[str, torch.Tensor] = {}
    for k, v in honest[0].delta.items():
        if v.is_floating_point():
            mu_honest[k] = torch.stack([h.delta[k] for h in honest]).mean(dim=0)
        else:
            mu_honest[k] = v.clone()

    mu_norm_sq = sum(
        float(v.norm().item()) ** 2 for v in mu_honest.values() if v.is_floating_point()
    )
    mu_norm = max(mu_norm_sq ** 0.5, 1e-9)
    eps = float(cfg.mimicry_epsilon)
    base_seed = int(cfg.mimicry_seed)

    # Replace each malicious update.
    diag_distances = []
    for u in malicious:
        new_delta: Dict[str, torch.Tensor] = {}
        for k, v in u.delta.items():
            if not v.is_floating_point():
                new_delta[k] = v
                continue
            seed = (base_seed * 100003 + round_idx * 1009 + u.client_id * 17 + hash(k)) & 0x7FFFFFFF
            g = torch.Generator(device=v.device).manual_seed(seed)
            v_attack = torch.randn(v.shape, generator=g, device=v.device, dtype=v.dtype)
            v_attack = v_attack / (v_attack.norm() + 1e-9)
            new_delta[k] = mu_honest[k] + (eps * mu_norm) * v_attack
        u.delta = new_delta
        # diagnostic: L2 distance between the new malicious delta and the centroid
        d_sq = sum(
            float((new_delta[k] - mu_honest[k]).norm().item()) ** 2
            for k in new_delta
            if new_delta[k].is_floating_point()
        )
        diag_distances.append(d_sq ** 0.5)

    diag = {
        "mu_honest_norm": mu_norm,
        "mal_centroid_distances": diag_distances,
        "mal_centroid_distance_mean": (
            sum(diag_distances) / len(diag_distances) if diag_distances else 0.0
        ),
    }
    return updates, diag


def apply_post_collection_adaptive_krum(updates, plan: AttackPlan, round_idx: int,
                                        krum_f: int = 2, top_med: int = 7):
    """Adaptive Krum-survival poisoning attack (reviewer experiment E2).

    Stronger than the bounded mimicry stress test: instead of a fixed
    perturbation scale, the attacker performs a per-round line search to find
    the *largest* damaging perturbation whose malicious updates still survive
    FedShield's top-``top_med`` Krum-ranked survivor list. Concretely, each
    malicious update is

        Delta_mal = mu_H + lambda * ||mu_H|| * v_attack + xi,

    where ``mu_H`` is the attacker's estimate of the honest-update centroid
    (mean of honest updates this round), ``v_attack`` is a unit-norm
    coordinate sign-flip direction ``-sign(mu_H)`` (a recognized damaging
    direction), ``xi`` is a tiny per-client noise to avoid exact duplication,
    and ``lambda`` is chosen as the largest value on a fixed grid for which
    *all* malicious copies remain inside the top-``top_med`` lowest-Krum-score
    set. The line search uses FedShield's filter parameters (``krum_f=2``,
    ``top_med=7``) regardless of the aggregator actually being evaluated, so
    the *same* malicious updates are fed to every defense for a fair
    comparison.

    Activated when ``"adaptive_krum"`` is in ``plan.cfg.types``; otherwise a
    no-op. Returns ``(updates, diag)`` with ``diag["top7_survival_rate"]`` (the
    fraction of malicious updates landing in the top-``top_med`` survivors of
    the finally applied updates) and ``diag["adaptive_lambda"]``.
    """
    cfg = plan.cfg
    if "adaptive_krum" not in cfg.types:
        return updates, {}

    honest = [u for u in updates if not u.is_malicious]
    malicious = [u for u in updates if u.is_malicious]
    if not honest or not malicious:
        return updates, {}

    float_keys = [k for k, v in honest[0].delta.items() if v.is_floating_point()]
    # Attacker's server-blind estimate of the honest centroid.
    mu_honest: Dict[str, torch.Tensor] = {}
    for k, v in honest[0].delta.items():
        if v.is_floating_point():
            mu_honest[k] = torch.stack([h.delta[k] for h in honest]).mean(dim=0)
        else:
            mu_honest[k] = v.clone()
    mu_norm = max(sum(float(mu_honest[k].norm().item()) ** 2 for k in float_keys) ** 0.5, 1e-9)
    # Unit-norm coordinate sign-flip direction.
    v_unit: Dict[str, torch.Tensor] = {k: -torch.sign(mu_honest[k]) for k in float_keys}
    v_norm = max(sum(float(v_unit[k].norm().item()) ** 2 for k in float_keys) ** 0.5, 1e-9)
    for k in float_keys:
        v_unit[k] = v_unit[k] / v_norm

    def _flat(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([delta[k].reshape(-1) for k in float_keys])

    honest_flat = torch.stack([_flat(h.delta) for h in honest])  # (H, d)
    n_mal = len(malicious)
    m_total = len(honest) + n_mal
    k_nn = max(1, m_total - krum_f - 2)
    top_k = min(top_med, m_total)

    def _candidate(lam: float) -> Dict[str, torch.Tensor]:
        return {k: mu_honest[k] + (lam * mu_norm) * v_unit[k] for k in float_keys}

    def _survival(cand_flat: torch.Tensor) -> float:
        # All malicious copies identical for ranking; honest + n_mal copies.
        all_flat = torch.cat([honest_flat, cand_flat.unsqueeze(0).repeat(n_mal, 1)], dim=0)
        d2 = torch.cdist(all_flat, all_flat) ** 2
        d2.fill_diagonal_(float("inf"))
        scores = d2.topk(k_nn, largest=False).values.sum(dim=1)  # (m_total,)
        survivors = set(scores.topk(top_k, largest=False).indices.tolist())
        mal_idx = set(range(len(honest), m_total))
        return len(survivors & mal_idx) / n_mal

    lambda_grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    chosen_lam = lambda_grid[0]
    for lam in lambda_grid:
        cand_flat = _flat(_candidate(lam))
        if _survival(cand_flat) >= 1.0 - 1e-9:  # all malicious still survive
            chosen_lam = lam
        else:
            break  # survival is monotone-decreasing in lambda; stop at first failure

    # Apply the chosen perturbation to every malicious client, with tiny noise.
    base_seed = int(getattr(cfg, "mimicry_seed", 12345))
    cand_template = _candidate(chosen_lam)
    for u in malicious:
        new_delta: Dict[str, torch.Tensor] = {}
        for k, v in u.delta.items():
            if not v.is_floating_point():
                new_delta[k] = v
                continue
            seed = (base_seed * 100003 + round_idx * 1009 + u.client_id * 17 + hash(k)) & 0x7FFFFFFF
            g = torch.Generator(device=v.device).manual_seed(seed)
            xi = torch.randn(v.shape, generator=g, device=v.device, dtype=v.dtype)
            new_delta[k] = cand_template[k] + (1e-3 * mu_norm) * (xi / (xi.norm() + 1e-9))
        u.delta = new_delta

    # Final survival rate on the actually-applied (noised) updates.
    applied_flat = torch.cat([honest_flat, torch.stack([_flat(u.delta) for u in malicious])], dim=0)
    d2 = torch.cdist(applied_flat, applied_flat) ** 2
    d2.fill_diagonal_(float("inf"))
    scores = d2.topk(k_nn, largest=False).values.sum(dim=1)
    survivors = set(scores.topk(top_k, largest=False).indices.tolist())
    mal_idx = set(range(len(honest), m_total))
    survival = len(survivors & mal_idx) / n_mal

    diag = {"adaptive_lambda": float(chosen_lam), "top7_survival_rate": float(survival)}
    return updates, diag


def build_test_trigger(
    X_test: np.ndarray, dataset_name: str, cfg: AttackConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X_with_trigger, mask_of_eligible_samples)`` for ASR scoring.

    Eligible = samples whose true label is *not* the attacker target class.
    """
    name = dataset_name.lower()
    Xb = X_test.copy()
    if "backdoor" not in cfg.types:
        return Xb, np.zeros(Xb.shape[0], dtype=bool)
    if name == "ciciomt":
        for fi in (0, 1, 2):
            Xb[..., fi] = Xb[..., fi] + cfg.backdoor_amplitude
    else:
        T = Xb.shape[-1]
        n_inject = max(8, T // 6)
        t = np.arange(n_inject) / 360.0
        pulse = (cfg.backdoor_amplitude * np.sin(2 * np.pi * cfg.backdoor_freq * t)).astype(np.float32)
        Xb[..., -n_inject:] += pulse
    return Xb, np.ones(Xb.shape[0], dtype=bool)
