"""Aggregation defenses: Krum, Multi-Krum, Trimmed Mean, FoolsGold, FLTrust,
and **FEDShield** (the proposed method).

Every defense exposes the same interface::

    aggregate(updates, ctx) -> aggregated_delta_state_dict, info_dict

where ``updates`` is a list of per-client ``ClientUpdate`` records and
``ctx`` is the shared :class:`AggregationContext` (carrying global model,
trust scores, AE alarms, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import DefenseConfig
from .utils import flatten_state_dict, unflatten_to_state_dict


# --------------------------------------------------------------------------- #
#                              Data classes
# --------------------------------------------------------------------------- #
@dataclass
class ClientUpdate:
    client_id: int
    delta: Dict[str, torch.Tensor]    # parameter delta (theta_local - theta_global)
    n_samples: int
    edge_alarm: int = 0               # 1 if local AE flagged anomaly
    edge_scale: float = 1.0           # client-side soft down-scale factor
    is_malicious: bool = False        # ground truth (only used for FRR scoring)


@dataclass
class AggregationContext:
    global_state: Dict[str, torch.Tensor]
    trust_scores: Dict[int, float] = field(default_factory=dict)
    fl_trust_root_delta: Dict[str, torch.Tensor] | None = None
    round_idx: int = 0
    # FEDShield-v7 server-side anomaly state: rolling stats of past honest
    # delta-feature vectors (norm, signed-cosine to running mean, sparsity).
    # Lets the server detect updates that *don't fit the historical pattern*
    # of accepted updates — catches data-poisoning gradients that pass Krum's
    # Euclidean test (because their direction is consensus-aligned).
    delta_stats_running: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#                              Helpers
# --------------------------------------------------------------------------- #
def _stack_flat(updates: List[ClientUpdate]) -> torch.Tensor:
    return torch.stack([flatten_state_dict(u.delta) for u in updates], dim=0)


def _scale(delta: Dict[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: v * s for k, v in delta.items()}


def _zero_like(delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in delta.items()}


def _accumulate(out: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], w: float) -> None:
    for k in out:
        out[k] = out[k] + delta[k] * w


# --------------------------------------------------------------------------- #
#                              Baselines
# --------------------------------------------------------------------------- #
def fedavg(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    total = sum(u.n_samples for u in updates)
    out = _zero_like(updates[0].delta)
    for u in updates:
        _accumulate(out, u.delta, u.n_samples / total)
    info = {"weights": {u.client_id: u.n_samples / total for u in updates}}
    return out, info


def krum(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    flat = _stack_flat(updates)               # (m, d)
    m = flat.shape[0]
    f = max(1, min(cfg.krum_f, m - 3))
    # pairwise squared distances
    diff = flat.unsqueeze(0) - flat.unsqueeze(1)
    d2 = (diff ** 2).sum(-1)                  # (m, m)
    sorted_d2, _ = torch.sort(d2, dim=1)
    # sum of m - f - 2 nearest neighbours (excluding self at idx 0)
    k = max(1, m - f - 2)
    scores = sorted_d2[:, 1 : 1 + k].sum(dim=1)
    chosen = int(torch.argmin(scores).item())
    info = {"chosen": updates[chosen].client_id, "scores": scores.tolist()}
    return updates[chosen].delta, info


def multi_krum(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    flat = _stack_flat(updates)
    m_count = flat.shape[0]
    f = max(1, min(cfg.krum_f, m_count - 3))
    diff = flat.unsqueeze(0) - flat.unsqueeze(1)
    d2 = (diff ** 2).sum(-1)
    sorted_d2, _ = torch.sort(d2, dim=1)
    k = max(1, m_count - f - 2)
    scores = sorted_d2[:, 1 : 1 + k].sum(dim=1)
    n_keep = max(1, min(cfg.multi_krum_m, m_count - f))
    keep_idx = torch.topk(-scores, n_keep).indices.tolist()
    out = _zero_like(updates[0].delta)
    for idx in keep_idx:
        _accumulate(out, updates[idx].delta, 1.0 / n_keep)
    info = {"selected": [updates[i].client_id for i in keep_idx]}
    return out, info


def trimmed_mean(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    flat = _stack_flat(updates)               # (m, d)
    m_count = flat.shape[0]
    n_trim = int(np.floor(cfg.trim_beta * m_count))
    sorted_, _ = torch.sort(flat, dim=0)
    if n_trim > 0:
        kept = sorted_[n_trim:m_count - n_trim] if m_count - 2 * n_trim > 0 else sorted_
    else:
        kept = sorted_
    aggregated = kept.mean(dim=0)
    info = {"trimmed": int(n_trim)}
    out = unflatten_to_state_dict(aggregated, updates[0].delta)
    return out, info


def foolsgold(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    flat = _stack_flat(updates)
    m_count = flat.shape[0]
    norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (flat @ flat.T) / (norms * norms.T)
    cos.fill_diagonal_(0)
    cs_max = cos.max(dim=1).values.clamp(0, 1).numpy()         # max sim per client
    # FoolsGold pardoning
    pardon = cs_max.copy()
    for i in range(m_count):
        for j in range(m_count):
            if i == j:
                continue
            if pardon[i] < pardon[j]:
                cos[i, j] *= pardon[i] / max(pardon[j], 1e-12)
    cs_max = cos.max(dim=1).values.clamp(0, 1).numpy()
    # learning-rate scaling
    lr = 1 - cs_max
    lr[lr > 1] = 1
    lr[lr < 0] = 0
    eps = 1e-5
    lr = np.log((lr + eps) / (1 - lr + eps)) + 0.5
    lr = np.clip(lr, 0, 1)
    if lr.sum() == 0:
        lr = np.ones_like(lr)
    weights = lr / lr.sum()
    out = _zero_like(updates[0].delta)
    for w, u in zip(weights, updates):
        _accumulate(out, u.delta, float(w))
    info = {"weights": {u.client_id: float(w) for u, w in zip(updates, weights)}}
    return out, info


def fltrust(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    """Requires ``ctx.fl_trust_root_delta`` to be populated with the server's
    own clean update on its root dataset."""
    if ctx.fl_trust_root_delta is None:
        # graceful fallback: behave like FedAvg if root unavailable
        return fedavg(updates, ctx, cfg)
    g0 = flatten_state_dict(ctx.fl_trust_root_delta)
    norm0 = g0.norm().clamp_min(1e-12)
    flat = _stack_flat(updates)
    cs = (flat @ g0) / (flat.norm(dim=1).clamp_min(1e-12) * norm0)
    ts = torch.relu(cs)
    if ts.sum() == 0:
        ts = torch.ones_like(ts)
    rescale = norm0 / flat.norm(dim=1).clamp_min(1e-12)
    weights = ts / ts.sum()
    out = _zero_like(updates[0].delta)
    for w, r, u in zip(weights, rescale, updates):
        _accumulate(out, u.delta, float(w * r))
    info = {"weights": {u.client_id: float(w) for u, w in zip(updates, weights)}}
    return out, info


# --------------------------------------------------------------------------- #
#                              FEDShield  (proposed)
# --------------------------------------------------------------------------- #
def _geometric_median(flat: torch.Tensor, n_iter: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Weiszfeld iteration. Rotation-invariant; converges in 2-3 iters here."""
    y = flat.median(dim=0).values
    for _ in range(n_iter):
        d = (flat - y).norm(dim=1).clamp_min(eps)
        w = 1.0 / d
        y = (flat * w.unsqueeze(1)).sum(dim=0) / w.sum()
    return y


def fedshield(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    """FEDShield-v6 — Krum-augmented-with-edge-AE.

    Diagnosis from v4: under per-record non-IID on real MIT-BIH, honest clients
    have cos≈0.99 to the median direction but a sign-flipped malicious client
    can have cos as high as +0.86 because the *un-flipped* malicious direction
    was already nearly orthogonal to consensus. The soft-cosine gate cannot
    reliably reject such updates. Krum's pairwise Euclidean score handles
    sign-flip and scaling correctly, but Krum collapses against coordinated
    *data* poisoning (ρ_m=0.4 backdoor → ds=0) because backdoor gradients
    look directionally normal.

    v6 design: Krum's distance-based selection (correctly catches model-layer
    attacks) augmented with the edge-side AE alarm (catches data-layer
    attacks Krum can't see, since Krum only sees gradients, not raw data).

    Stage 1 — Edge AE alarm: if client raised AE alarm, add a large penalty
              to its Krum score so it falls outside the top-(m-f) selection.
    Stage 2 — Pairwise Euclidean Krum score on norm-clipped deltas.
    Stage 3 — Multi-Krum: average the (m-f-2) lowest-scored clients
              (m=cohort size, f=byzantine count from cfg).
    Trust EMA preserved as a stability prior between rounds.
    """
    flat = _stack_flat(updates)                                          # (m, d)
    norms = flat.norm(dim=1).clamp_min(1e-12)
    m_count = flat.shape[0]
    a = cfg.fedshield_alpha_ema
    k_mad = getattr(cfg, "fedshield_k_mad", 5.0)
    warmup_rounds = getattr(cfg, "fedshield_warmup_rounds", 2)
    use_krum_aug = getattr(cfg, "fedshield_use_krum_aug", True)
    # Ablation toggles (paper experiments). Each disables ONE stage of
    # FedShield while leaving the rest at v10 defaults.
    ablate_edge_ae   = bool(getattr(cfg, "fedshield_ablate_edge_ae", False))
    ablate_norm_clip = bool(getattr(cfg, "fedshield_ablate_norm_clip", False))
    ablate_trim_ref  = bool(getattr(cfg, "fedshield_ablate_trim_ref", False))
    ablate_buggy_f   = bool(getattr(cfg, "fedshield_ablate_buggy_f", False))

    # ----- Phase 1: warmup -> pure FedAvg ---------------------------------- #
    if ctx.round_idx < warmup_rounds:
        weights = torch.full((m_count,), 1.0 / m_count)
        for u in updates:
            ctx.trust_scores[u.client_id] = 1.0
        valid_mask = [True] * m_count
        flat_clipped = flat
        cos = torch.ones(m_count)
        n_med = float(norms.median().item())
    else:
        # ----- Stage B: norm test (MAD) ----------------------------------- #
        # MAD has high variance with small cohorts (m<8). We floor MAD at
        # 25% of the median norm so the outlier test does not trigger on
        # legitimate non-IID variance simply because clients happen to
        # produce nearly-identical norms.
        n_med = float(norms.median().item())
        mad = float((norms - n_med).abs().median().item())
        mad = max(mad, 0.25 * n_med)
        norm_outlier = (norms - n_med).abs() > k_mad * mad

        # Stage B (clipping). Ablation A2: skip → use raw flat downstream.
        if ablate_norm_clip:
            flat_clipped = flat
        else:
            clip_ceiling = n_med + k_mad * mad
            clip_factors = torch.clamp(clip_ceiling / norms, max=1.0)
            flat_clipped = flat * clip_factors.unsqueeze(1)

        # Stage C reference direction. Ablation A3: use mean-of-all instead
        # of trim-median over norm-test survivors.
        if ablate_trim_ref:
            ref = flat_clipped.mean(dim=0)
        elif (~norm_outlier).sum() >= 2:
            ref = flat_clipped[~norm_outlier].median(dim=0).values
        else:
            ref = flat_clipped.median(dim=0).values
        ref_norm = ref.norm().clamp_min(1e-12)

        # ----- Stage C: cosine -> SOFT weight ----------------------------- #
        cos = (flat_clipped @ ref) / (flat_clipped.norm(dim=1).clamp_min(1e-12) * ref_norm)

        # Hard rejection only when BOTH the cosine direction is adversarial
        # AND the AE alarm fires. The MAD-based norm outlier test is *not*
        # used as a hard rejection signal — empirically it false-positives
        # on heterogeneous-data honest clients whose gradient direction is
        # perfectly aligned (cos~0.999) but whose norm is naturally larger
        # because they hold a more diverse class mix. Norm clipping (Stage B)
        # already neutralises scaling attacks magnitude-wise; cosine soft-
        # weighting (below) already neutralises sign-flip direction-wise.
        # Hard rejection on norm alone adds false positives without adding
        # safety, so it has been removed.
        valid_mask = []
        reject_reasons = {}
        for i, u in enumerate(updates):
            ci = float(cos[i].item())
            reasons = []
            if ci < 0.0 and bool(u.edge_alarm):
                reasons.append("neg_cos_and_ae")
            reject = len(reasons) > 0
            if reject:
                reject_reasons[u.client_id] = reasons
            valid_mask.append(not reject)
            prev_ts = ctx.trust_scores.get(u.client_id, 1.0)
            instant = max(0.0, (ci + 1.0) / 2.0) if not reject else 0.0   # in [0,1]
            ctx.trust_scores[u.client_id] = a * prev_ts + (1.0 - a) * instant

        # ----- v7 server-side delta-pattern anomaly detector ---------- #
        use_server_ae = getattr(cfg, "fedshield_use_server_ae", False)
        server_ae_warmup = getattr(cfg, "fedshield_server_ae_warmup", 3)
        server_ae_z = getattr(cfg, "fedshield_server_ae_z", 3.0)
        server_ae_flags = [False] * m_count
        if use_server_ae and ctx.round_idx >= server_ae_warmup:
            # historical norm distribution stored in ctx.delta_stats_running
            stats = ctx.delta_stats_running.get("norms", [])
            if len(stats) >= 4:
                import statistics as _stats
                mu = _stats.mean(stats)
                sd = _stats.stdev(stats)
                if sd > 1e-9:
                    for i in range(m_count):
                        z = abs(float(norms[i].item()) - mu) / sd
                        if z > server_ae_z:
                            server_ae_flags[i] = True

        if use_krum_aug:
            # ----- v7.3: Krum distance on RAW flat (no clipping) ------- #
            # Norm clipping flattens the magnitude signal that Krum uses
            # to detect scaled attackers. Use raw distances for selection.
            # AE alarm penalty added below.
            d2 = ((flat.unsqueeze(0) - flat.unsqueeze(1)) ** 2).sum(-1)
            # v8: use cfg.krum_f (same as vanilla Krum, default 2) instead of
            # hardcoded 0.4*m_count. Setting f equal to the actual adversary
            # fraction makes malicious clients' mutual distances dominate the
            # score. Ablation A5 re-enables the buggy formula for the paper.
            if ablate_buggy_f:
                f = max(1, min(int(0.4 * m_count), m_count - 3))
            else:
                f = max(1, min(int(getattr(cfg, "krum_f", 2)), m_count - 3))
            k = max(1, m_count - f - 2)
            sorted_d2, _ = torch.sort(d2, dim=1)
            krum_score = sorted_d2[:, 1 : 1 + k].sum(dim=1)
            # Combine edge-AE alarm + server-AE flag into one penalty.
            # v8: also penalise hard-negative-cosine + norm-outlier clients
            # when those flags are enabled. At high malicious ratio AE signal
            # is contaminated, but cos<thresh and norm-MAD are clean
            # direction- and magnitude-based rejection signals.
            hard_neg_cos = bool(getattr(cfg, "fedshield_hard_neg_cos", False))
            hard_cos_thresh = float(getattr(cfg, "fedshield_hard_cos_thresh", 0.0))
            hard_norm = bool(getattr(cfg, "fedshield_hard_norm_outlier", False))
            ae_penalty = torch.tensor([
                1e12 if (
                    ((not ablate_edge_ae) and u.edge_alarm)
                    or server_ae_flags[i]
                    or (hard_neg_cos and float(cos[i].item()) < hard_cos_thresh)
                    or (hard_norm and bool(norm_outlier[i].item()))
                ) else 0.0
                for i, u in enumerate(updates)
            ], device=krum_score.device)
            score = krum_score + ae_penalty
            # v8: Multi-Krum — keep the m clients with smallest score.
            mk_m = max(1, int(getattr(cfg, "fedshield_multikrum_m", 1)))
            mk_m = min(mk_m, m_count)
            sel = torch.argsort(score)[:mk_m]
            best_idx = int(sel[0].item())
            keep = torch.zeros(m_count, dtype=torch.bool, device=krum_score.device)
            keep[sel] = True
            # v7: append the selected (winning) client's norm to running stats
            # so future rounds can detect distribution shift.
            if use_server_ae:
                stats = ctx.delta_stats_running.setdefault("norms", [])
                stats.append(float(norms[best_idx].item()))
                # keep last 30 rounds (rolling window)
                if len(stats) > 30:
                    del stats[: len(stats) - 30]
            # Trust EMA update from cosine for telemetry, no longer drives weight.
            ts = torch.tensor([ctx.trust_scores[u.client_id] for u in updates], device=keep.device)
            for i, u in enumerate(updates):
                ci = float(cos[i].item())
                prev_ts = ctx.trust_scores.get(u.client_id, 1.0)
                instant = max(0.0, (ci + 1.0) / 2.0) if keep[i].item() else 0.0
                ctx.trust_scores[u.client_id] = a * prev_ts + (1.0 - a) * instant
            # Uniform average over kept set, modulated by trust prior.
            # v8: optionally weight by max(cos(Δw_i, ref), 0) so survivors
            # with weak alignment contribute less.
            cos_weight = bool(getattr(cfg, "fedshield_topm_cos_weight", False))
            cos_pos = torch.clamp(cos, min=0.0)
            base = ts * keep.float()
            raw = base * cos_pos if cos_weight else base
            valid_mask = [bool(keep[i].item()) for i in range(m_count)]
            for i, u in enumerate(updates):
                if not valid_mask[i]:
                    reject_reasons[u.client_id] = (
                        ["ae_alarm"] if u.edge_alarm else ["krum_score"]
                    )
            if raw.sum() <= 1e-9:
                weights = keep.float() / max(int(keep.sum().item()), 1)
                if weights.sum() == 0:
                    weights = torch.ones(m_count, device=keep.device) / m_count
            else:
                weights = raw / raw.sum()
        else:
            # legacy v4 soft-cosine path (preserved for ablation)
            soft = torch.clamp((cos + 1.0) / 2.0, 0.0, 1.0)
            ts = torch.tensor([ctx.trust_scores[u.client_id] for u in updates], device=cos.device)
            keep = torch.tensor(valid_mask, dtype=torch.bool, device=cos.device)
            raw = soft * ts * keep.float()
            if raw.sum() <= 1e-9:
                weights = keep.float() / max(int(keep.sum().item()), 1)
                if weights.sum() == 0:
                    weights = torch.ones(m_count, device=cos.device) / m_count
            else:
                weights = raw / raw.sum()

    # v7.2: Aggregate using RAW flat (not clipped). Clipping was a Stage-B
    # safety net to bound malicious magnitude before aggregation. But once
    # selection has discarded malicious clients (Krum + AE penalty), the
    # surviving update is honest by construction — clipping it just
    # erases honest-client gradient information and starves learning.
    aggreg_mode = str(getattr(cfg, "fedshield_aggreg_mode", "topm"))
    edge_scales = torch.tensor([u.edge_scale for u in updates],
                               device=flat.device, dtype=flat.dtype)
    # v9: adaptive mode chooses between mean-of-clean_m and median-of-adv_m
    # based on the number of detected adversaries (edge_alarm or norm_outlier).
    if use_krum_aug and bool(getattr(cfg, "fedshield_adaptive", False)):
        adv_count = sum(
            1 for i, u in enumerate(updates)
            if u.edge_alarm or bool(norm_outlier[i].item())
        )
        adv_thresh = int(getattr(cfg, "fedshield_adaptive_threshold", 1))
        if adv_count < adv_thresh:
            aggreg_mode = "topm"
            mk_m = int(getattr(cfg, "fedshield_adaptive_clean_m", 2))
        else:
            aggreg_mode = "median"
            mk_m = int(getattr(cfg, "fedshield_adaptive_adv_m", 7))
        # Recompute keep mask for the adaptive m
        sel = torch.argsort(score)[:mk_m]
        keep = torch.zeros(m_count, dtype=torch.bool, device=krum_score.device)
        keep[sel] = True
        valid_mask = [bool(keep[i].item()) for i in range(m_count)]
        # Update weights for "topm" path so the trust-EMA-weighted average works
        cos_pos = torch.clamp(cos, min=0.0)
        base = ts * keep.float()
        raw = base
        if raw.sum() <= 1e-9:
            weights = keep.float() / max(int(keep.sum().item()), 1)
            if weights.sum() == 0:
                weights = torch.ones(m_count, device=keep.device) / m_count
        else:
            weights = raw / raw.sum()
    if use_krum_aug and aggreg_mode == "softmax":
        # v8: softmax-weighted average of ALL clients by negative Krum score.
        # No hard rejection — adversaries get small (not zero) weight, which
        # avoids hard-cutoff sensitivity to single noisy rounds.
        T = max(1e-6, float(getattr(cfg, "fedshield_softscore_temp", 0.3)))
        # Normalise scores so temperature has consistent meaning across rounds.
        s = score - score.min()
        s = s / s.max().clamp_min(1e-12)
        weights = torch.softmax(-s / T, dim=0)
        aggregated = (flat * (weights * edge_scales).unsqueeze(1)).sum(dim=0)
    elif use_krum_aug and aggreg_mode == "trimavg":
        # v8: coordinate-wise trim-mean over the top-m clients selected above.
        # Resilient to outlier coordinates contributed by a single rogue
        # selectee that slipped through Krum scoring.
        sel_idx = torch.argsort(score)[:mk_m]
        sel_flat = flat[sel_idx] * edge_scales[sel_idx].unsqueeze(1)
        beta = float(getattr(cfg, "fedshield_aggreg_trim_beta", 0.3))
        n_keep = sel_flat.shape[0]
        n_trim = int(beta * n_keep)
        sorted_, _ = torch.sort(sel_flat, dim=0)
        if n_keep - 2 * n_trim > 0:
            kept = sorted_[n_trim:n_keep - n_trim]
        else:
            kept = sorted_
        aggregated = kept.mean(dim=0)
    elif use_krum_aug and aggreg_mode == "median":
        # v8: coordinate-wise median over the top-m clients. Stronger
        # robustness than mean when the kept set may contain one bad client.
        sel_idx = torch.argsort(score)[:mk_m]
        sel_flat = flat[sel_idx] * edge_scales[sel_idx].unsqueeze(1)
        aggregated = sel_flat.median(dim=0).values
    elif use_krum_aug and aggreg_mode == "ensemble":
        # v10: convex combination of "mean of top-mk2_m" (good vs scaling)
        # and "coord-wise median of top-med_m" (good vs sign-flip).
        a = float(getattr(cfg, "fedshield_ensemble_alpha", 0.5))
        m2 = int(getattr(cfg, "fedshield_ensemble_mk2_m", 2))
        mm = int(getattr(cfg, "fedshield_ensemble_med_m", 7))
        m2 = max(1, min(m2, m_count))
        mm = max(1, min(mm, m_count))
        sorted_idx = torch.argsort(score)
        # Design-rationale: optionally disable the Krum filter to test whether
        # the pre-filter is load-bearing.
        if bool(getattr(cfg, "fedshield_disable_krum_filter", False)):
            sel_a = torch.arange(m_count, device=flat.device)
            sel_b = torch.arange(m_count, device=flat.device)
        else:
            sel_a = sorted_idx[:m2]
            sel_b = sorted_idx[:mm]
        beta_trim = float(getattr(cfg, "fedshield_aggreg_trim_beta", 0.3))

        def _aggregate(kind: str, sel: torch.Tensor) -> torch.Tensor:
            sub = flat[sel] * edge_scales[sel].unsqueeze(1)
            n = sub.shape[0]
            if kind == "mean":
                return sub.mean(dim=0)
            if kind == "median":
                return sub.median(dim=0).values
            if kind == "trim":
                k = int(beta_trim * n)
                if n - 2 * k <= 0:
                    return sub.mean(dim=0)
                sorted_, _ = torch.sort(sub, dim=0)
                return sorted_[k:n - k].mean(dim=0)
            if kind == "geomed":
                # Weiszfeld iterations for geometric median
                gm = sub.mean(dim=0)
                for _ in range(int(getattr(cfg, "fedshield_geo_median_iters", 3))):
                    d = (sub - gm).norm(dim=1).clamp_min(1e-9)
                    w_ = (1.0 / d).unsqueeze(1)
                    gm = (sub * w_).sum(dim=0) / w_.sum()
                return gm
            return sub.mean(dim=0)

        a_kind = str(getattr(cfg, "fedshield_ens_a_kind", "mean"))
        b_kind = str(getattr(cfg, "fedshield_ens_b_kind", "median"))
        agg_a = _aggregate(a_kind, sel_a)
        agg_b = _aggregate(b_kind, sel_b)

        # Design-rationale: optionally test alternate combination forms.
        combine = str(getattr(cfg, "fedshield_combine_form", "convex"))
        if combine == "max":
            aggregated = torch.maximum(agg_a, agg_b)
        elif combine == "min":
            aggregated = torch.minimum(agg_a, agg_b)
        elif combine == "alarm_gated":
            n_alarms = sum(1 for u in updates if u.edge_alarm)
            aggregated = agg_a if n_alarms == 0 else agg_b
        else:  # "convex" (default)
            aggregated = a * agg_a + (1.0 - a) * agg_b
    else:
        aggregated = torch.zeros_like(flat[0])
        for i, u in enumerate(updates):
            aggregated = aggregated + flat[i] * float(weights[i].item()) * u.edge_scale

    out = unflatten_to_state_dict(aggregated, updates[0].delta)
    info = {
        "weights": {u.client_id: float(weights[i].item()) for i, u in enumerate(updates)},
        "cosines": {u.client_id: float(cos[i].item()) for i, u in enumerate(updates)},
        "alarms": {u.client_id: int(u.edge_alarm) for u in updates},
        "norms": {u.client_id: float(norms[i].item()) for i, u in enumerate(updates)},
        "norm_median": n_med,
        "rejected": [updates[i].client_id for i, k in enumerate(valid_mask) if not k],
        "reject_reasons": locals().get("reject_reasons", {}),
    }
    return out, info


# --------------------------------------------------------------------------- #
#                              Registry
# --------------------------------------------------------------------------- #
def coord_median(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    """Per-coordinate median across all clients (no Krum filter).

    Domain reviewers in healthcare-FL often request a comparison against the
    plain coordinate-wise median as a simple, threat-model-light reference.
    """
    flat = _stack_flat(updates)
    aggregated = flat.median(dim=0).values
    out = unflatten_to_state_dict(aggregated, updates[0].delta)
    info = {"selected": [u.client_id for u in updates]}
    return out, info


def rfa(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    """Robust Federated Aggregation (RFA)~\\cite{pillutla2022robust}: the
    geometric median of the client update vectors, computed by smoothed
    Weiszfeld iteration. Unlike coordinate-wise median, RFA is rotation
    invariant and operates on whole vectors; it is a standard recent
    distance-based robust aggregator and is run here as a clean baseline
    (no Krum prefilter, no mean--median fusion)."""
    flat = _stack_flat(updates)
    aggregated = _geometric_median(flat, n_iter=int(getattr(cfg, "rfa_iters", 4)))
    out = unflatten_to_state_dict(aggregated, updates[0].delta)
    info = {"selected": [u.client_id for u in updates]}
    return out, info


def dnc(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    """Divide-and-Conquer (DnC) spectral robust aggregation~\\cite{shejwalkar2021manipulating}.
    Centers the client updates, projects each onto the top right singular
    vector of the centered matrix, scores each update by its squared
    projection (outlier score), removes the $f$ highest-scoring updates, and
    averages the remainder. This is a recent (2021) detection/filtering-based
    aggregator that targets the dominant manipulation direction rather than
    pairwise distances."""
    flat = _stack_flat(updates)                       # (m, d)
    m = flat.shape[0]
    mu = flat.mean(dim=0)
    centered = flat - mu
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        v = vh[0]
    except Exception:
        v = centered[0] / (centered[0].norm() + 1e-12)
    scores = (centered @ v) ** 2                       # outlier score per update
    f = int(getattr(cfg, "krum_f", 2))
    n_keep = max(1, m - f)
    keep = torch.argsort(scores)[:n_keep]              # lowest scores: kept as honest
    aggregated = flat[keep].mean(dim=0)
    out = unflatten_to_state_dict(aggregated, updates[0].delta)
    info = {"selected": [updates[i].client_id for i in keep.tolist()]}
    return out, info


def bulyan(updates: List[ClientUpdate], ctx: AggregationContext, cfg: DefenseConfig):
    """Bulyan (Mhamdi, Guerraoui, Rouault, ICML 2018) — cascaded
    Krum→trimmed-mean.

    Stage 1: iteratively pick the lowest-Krum-score client, append to
             the survivor set S, remove from candidate pool, repeat until
             |S| = θ = m − 2f.
    Stage 2: per-coordinate trimmed mean over S with trim-fraction β
             selecting the m−4f central values per coordinate.

    Robust if |B| ≤ f and m ≥ 4f + 3. We use the same f as Krum
    (cfg.krum_f, default 2) for fair comparison.
    """
    flat = _stack_flat(updates)
    m = flat.shape[0]
    f = max(1, min(cfg.krum_f, (m - 3) // 4))  # ensure m >= 4f+3
    theta = max(1, m - 2 * f)
    # Stage 1: iteratively pick lowest-Krum-score client
    diff = flat.unsqueeze(0) - flat.unsqueeze(1)
    d2 = (diff ** 2).sum(-1)
    survivors: List[int] = []
    candidate_mask = torch.ones(m, dtype=torch.bool, device=flat.device)
    for _ in range(theta):
        # Krum score among current candidates
        cand_idx = candidate_mask.nonzero(as_tuple=True)[0]
        if len(cand_idx) <= 2:
            survivors.extend(cand_idx.tolist())
            break
        sub = d2[cand_idx][:, cand_idx]
        sorted_, _ = torch.sort(sub, dim=1)
        k = max(1, len(cand_idx) - f - 2)
        scores = sorted_[:, 1:1 + k].sum(dim=1)
        winner_local = int(scores.argmin().item())
        winner_global = int(cand_idx[winner_local].item())
        survivors.append(winner_global)
        candidate_mask[winner_global] = False
    # Stage 2: per-coordinate trim-mean over survivors
    surv_flat = flat[torch.tensor(survivors, device=flat.device)]
    n_surv = surv_flat.shape[0]
    n_trim = max(0, (n_surv - max(1, n_surv - 4 * f)) // 2)
    if n_surv - 2 * n_trim > 0:
        sorted_, _ = torch.sort(surv_flat, dim=0)
        kept = sorted_[n_trim:n_surv - n_trim]
    else:
        kept = surv_flat
    aggregated = kept.mean(dim=0)
    out = unflatten_to_state_dict(aggregated, updates[0].delta)
    info = {"survivors": [updates[i].client_id for i in survivors],
            "f": f, "theta": theta, "n_trim": n_trim}
    return out, info


DEFENSES = {
    "fedavg": fedavg,
    "krum": krum,
    "multi_krum": multi_krum,
    "trimmed_mean": trimmed_mean,
    "foolsgold": foolsgold,
    "fltrust": fltrust,
    "median": coord_median,
    "rfa": rfa,
    "dnc": dnc,
    "bulyan": bulyan,
    "fedshield": fedshield,
}


def get_defense(name: str):
    if name not in DEFENSES:
        raise ValueError(f"unknown defense {name}; available: {sorted(DEFENSES)}")
    return DEFENSES[name]
