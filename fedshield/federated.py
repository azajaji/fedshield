"""Federated training orchestration.

Combines :mod:`data_loader`, :mod:`models`, :mod:`attacks`, and :mod:`defenses`
into a single training loop. Each round:

  1. server samples a cohort
  2. each client trains locally; if malicious, may corrupt data and/or update
  3. each client runs its local autoencoder for edge-side gating (FEDShield)
  4. server runs the configured aggregation defense
  5. evaluator records accuracy / ASR / FRR / latency
"""
from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import (
    AttackPlan,
    apply_data_attack,
    apply_update_attack,
    apply_post_collection_mimicry,
    apply_post_collection_adaptive_krum,
    build_test_trigger,
    select_malicious_clients,
)
from .config import ExperimentConfig
from .data_loader import ArrayDataset, FederatedSplit, build_federated_data, make_loader
from .defenses import AggregationContext, ClientUpdate, get_defense
from .models import build_autoencoder, build_classifier, model_param_count, TinyAE  # noqa: F401
from .utils import (
    CSVLogger,
    Timer,
    flatten_state_dict,
    save_json,
    set_seed,
    state_dict_sub,
)


# --------------------------------------------------------------------------- #
#                              Local training
# --------------------------------------------------------------------------- #
def train_local(
    model: nn.Module,
    loader,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: str,
) -> nn.Module:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
    return model


def train_autoencoder_one_epoch(ae: TinyAE, loader, lr: float, device: str) -> None:
    ae.train()
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for X, _ in loader:
        X = X.to(device)
        opt.zero_grad()
        out = ae(X)
        loss = F.mse_loss(out, X)
        loss.backward()
        opt.step()


def autoencoder_anomaly_score(ae: TinyAE, loader, device: str) -> Tuple[float, float]:
    """Return ``(mean, std)`` of per-sample reconstruction error."""
    ae.eval()
    errs: List[float] = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            errs.extend(ae.reconstruction_error(X).cpu().tolist())
    if not errs:
        return 0.0, 0.0
    return float(np.mean(errs)), float(np.std(errs))


# --------------------------------------------------------------------------- #
#                              Evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader, device: str) -> Tuple[float, float]:
    model.eval()
    n_total = 0
    n_correct = 0
    all_y, all_p = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        pred = out.argmax(1)
        n_correct += int((pred == y).sum().item())
        n_total += y.numel()
        all_y.extend(y.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
    acc = n_correct / max(n_total, 1)
    # macro-F1
    classes = sorted(set(all_y) | set(all_p))
    f1s = []
    for c in classes:
        tp = sum(1 for yi, pi in zip(all_y, all_p) if yi == c and pi == c)
        fp = sum(1 for yi, pi in zip(all_y, all_p) if yi != c and pi == c)
        fn = sum(1 for yi, pi in zip(all_y, all_p) if yi == c and pi != c)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    return acc, float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate_asr(model: nn.Module, X_trig: np.ndarray, y_clean: np.ndarray,
                 target: int, device: str, batch_size: int = 256
                 ) -> Tuple[float, int, str]:
    """Attack Success Rate = P(predicted == target | true != target).

    Returns ``(asr, n_eligible, status)`` where ``status`` is one of:
        ``ok``           — at least 50 eligible samples, value is reliable.
        ``few_samples``  — fewer than 50 eligible samples; value is reported but flagged.
        ``no_eligible``  — every test sample's true label equals the target;
                            ASR is undefined and reported as ``nan`` instead of 0.
    Reviewers can filter on ``status`` to exclude flagged cells from headline tables.
    """
    model.eval()
    elig = y_clean != target
    n = int(elig.sum())
    if n == 0:
        return float("nan"), 0, "no_eligible"
    Xt = torch.from_numpy(X_trig[elig].astype(np.float32)).to(device)
    preds = []
    for i in range(0, len(Xt), batch_size):
        out = model(Xt[i : i + batch_size])
        preds.extend(out.argmax(1).cpu().tolist())
    preds = np.array(preds)
    asr = float((preds == target).mean())
    status = "ok" if n >= 50 else "few_samples"
    return asr, n, status


# --------------------------------------------------------------------------- #
#                              Per-client state
# --------------------------------------------------------------------------- #
class ClientState:
    """Holds a client's persistent local state across rounds (its autoencoder
    and its rolling reconstruction-error baseline)."""

    def __init__(self, idx: int, train_ds: ArrayDataset, val_ds: ArrayDataset,
                 ae: TinyAE, batch_size: int, local_epochs: int) -> None:
        self.idx = idx
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.ae = ae
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.recon_mu = 0.0
        self.recon_sigma = 1.0
        self.last_cos = 1.0     # server feedback; controls lazy AE activation


# --------------------------------------------------------------------------- #
#                              Trainer
# --------------------------------------------------------------------------- #
class FederatedTrainer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[fedshield] WARNING: device={self.device} requested but CUDA unavailable; falling back to cpu")
            self.device = "cpu"
        if self.device.startswith("cuda"):
            # Determinism > speed for benchmarks. cudnn.benchmark auto-tunes
            # kernels per-shape per-process and produces ±0.15 score drift
            # between identical-seed runs in different processes.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print(f"[fedshield] using device={self.device} ({torch.cuda.get_device_name(0)})")

        # data
        self.split: FederatedSplit = build_federated_data(cfg.data, seed=cfg.seed)
        cfg.model.num_classes = self.split.num_classes
        cfg.model.input_dim = int(np.prod(self.split.input_shape))

        # global model
        self.global_model = build_classifier(cfg.model, self.split.input_shape).to(self.device)
        self.input_shape = self.split.input_shape

        # per-client state
        rng = np.random.default_rng(cfg.seed)
        self.malicious_ids = select_malicious_clients(
            cfg.data.num_clients, cfg.attack.malicious_ratio, rng
        )
        self.attack_plan = AttackPlan(self.malicious_ids, cfg.attack)

        self.clients: List[ClientState] = []
        for i, (tr, va) in enumerate(zip(self.split.client_train, self.split.client_val)):
            # data corruption for malicious clients (data-layer)
            if self.attack_plan.is_malicious(i):
                tr = apply_data_attack(tr, self.attack_plan, self.split.num_classes,
                                       np.random.default_rng(cfg.seed + i),
                                       cfg.data.name)
            ae = build_autoencoder(cfg.model, self.split.input_shape).to(self.device)
            bs = int(rng.choice(cfg.data.batch_size_choices))
            ep = int(rng.choice(cfg.federated.local_epochs_choices))
            self.clients.append(ClientState(i, tr, va, ae, bs, ep))

        # FLTrust root set (server-side)
        self.fltrust_loader = None
        if cfg.defense.name == "fltrust":
            n_root = cfg.defense.fltrust_root_size
            X = self.split.test.X.numpy()[:n_root]
            y = self.split.test.y.numpy()[:n_root]
            self.fltrust_loader = make_loader(ArrayDataset(X, y), batch_size=64)

        # global trust state
        self.trust_scores: Dict[int, float] = {i: 1.0 for i in range(cfg.data.num_clients)}

        self._cur_round = 0

        # logger
        # defense_score = acc * (1 - asr) is the canonical "Backdoor Accuracy"
        # combined metric from Bagdasaryan'20 / FLTrust'21 — robust to either
        # mode of failure (collapse OR poisoning).
        self.logger = CSVLogger(
            os.path.join(cfg.out_dir, f"{cfg.name}_metrics.csv"),
            fieldnames=[
                "round", "acc", "f1", "asr", "asr_status", "n_asr_eligible",
                "defense_score", "frr",
                "latency_edge_ms", "latency_server_ms",
                "latency_train_ms", "latency_ae_ms",
                "comm_bytes_per_client", "edge_ram_mb",
                "rejected", "defense", "dataset", "attack",
                "malicious_ratio", "n_params", "n_ae_params",
                "backdoor_target", "top7_survival_rate",
            ],
        )

    # ----------------------------- bootstrap AE ----------------------------- #
    def warmup_autoencoders(self) -> None:
        for c in self.clients:
            loader = make_loader(c.train_ds, batch_size=c.batch_size, shuffle=True)
            for _ in range(2):
                train_autoencoder_one_epoch(c.ae, loader, lr=1e-3, device=self.device)
            val_loader = make_loader(c.val_ds, batch_size=c.batch_size, shuffle=False)
            mu, sigma = autoencoder_anomaly_score(c.ae, val_loader, self.device)
            c.recon_mu, c.recon_sigma = mu, max(sigma, 1e-6)

    # ----------------------------- one round -------------------------------- #
    def _client_round(self, c: ClientState) -> ClientUpdate:
        # take a private copy of global parameters
        local = copy.deepcopy(self.global_model)
        loader = make_loader(c.train_ds, batch_size=c.batch_size, shuffle=True)
        with Timer() as t_train:
            train_local(
                local, loader, c.local_epochs,
                lr=self.cfg.federated.lr,
                momentum=self.cfg.federated.momentum,
                weight_decay=self.cfg.federated.weight_decay,
                device=self.device,
            )
        self._last_train_ms = 1000.0 * t_train.elapsed
        delta = state_dict_sub(local.state_dict(), self.global_model.state_dict())

        # update-layer attack for malicious clients
        is_mal = self.attack_plan.is_malicious(c.idx)
        if is_mal:
            delta = apply_update_attack(delta, self.attack_plan)

        # edge-side AE gating — *lazy*: bootstrap for first 3 rounds, then
        # only run when the last server-reported cosine was low.
        run_ae = (
            self._cur_round < 3
            or c.last_cos < self.cfg.defense.fedshield_ae_lazy_phi
        )
        with Timer() as t_ae:
            if run_ae:
                val_loader = make_loader(c.val_ds, batch_size=c.batch_size, shuffle=False)
                cur_mu, _ = autoencoder_anomaly_score(c.ae, val_loader, self.device)
                threshold = c.recon_mu + self.cfg.defense.fedshield_threshold_k * c.recon_sigma
                alarm = int(cur_mu > threshold)
                edge_scale = float(min(1.0, threshold / max(cur_mu, 1e-9))) if alarm else 1.0
                if not alarm:
                    c.recon_mu = 0.9 * c.recon_mu + 0.1 * cur_mu
            else:
                alarm = 0
                edge_scale = 1.0
        self._last_ae_ms = 1000.0 * t_ae.elapsed

        return ClientUpdate(
            client_id=c.idx,
            delta=delta,
            n_samples=len(c.train_ds),
            edge_alarm=alarm,
            edge_scale=edge_scale,
            is_malicious=is_mal,
        )

    def _maybe_compute_fltrust_root_delta(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.cfg.defense.name != "fltrust" or self.fltrust_loader is None:
            return None
        local = copy.deepcopy(self.global_model)
        train_local(
            local, self.fltrust_loader, epochs=1,
            lr=self.cfg.federated.lr,
            momentum=self.cfg.federated.momentum,
            weight_decay=self.cfg.federated.weight_decay,
            device=self.device,
        )
        return state_dict_sub(local.state_dict(), self.global_model.state_dict())

    # ----------------------------- full training ---------------------------- #
    def train(self) -> Dict:
        self.warmup_autoencoders()
        defense_fn = get_defense(self.cfg.defense.name)
        rng = np.random.default_rng(self.cfg.seed)
        n_params = model_param_count(self.global_model)
        history: List[Dict] = []
        self._adaptive_survival: List[float] = []  # E2 adaptive-attack survival rates

        # build trigger test set once
        X_clean, y_clean = self.split.test.X.numpy(), self.split.test.y.numpy()
        X_trig, _ = build_test_trigger(X_clean, self.cfg.data.name, self.cfg.attack)
        target = self.cfg.attack.backdoor_target
        if target is None:
            # Hard error — silently picking majority class conflates "defended"
            # with "model destroyed by collapse to a non-target class".
            raise ValueError(
                f"attack.backdoor_target must be set explicitly for dataset"
                f" '{self.cfg.data.name}'; received None. The argmax-of-majority"
                f" fallback was removed because it produced misleading ASR=0"
                f" rows for collapsed models."
            )
        # Sanity: target must actually appear in the test set.
        if int((y_clean == target).sum()) == 0:
            raise ValueError(
                f"backdoor_target={target} is absent from the test set;"
                f" ASR is undefined."
            )

        test_loader = make_loader(self.split.test, batch_size=128, shuffle=False)

        for r in range(self.cfg.federated.rounds):
            self._cur_round = r
            # cohort sampling
            n_sample = max(2, int(self.cfg.federated.sample_ratio * len(self.clients)))
            cohort = sorted(rng.choice(len(self.clients), size=n_sample, replace=False).tolist())

            # client rounds
            self._last_train_ms = 0.0
            self._last_ae_ms = 0.0
            train_ms_acc = 0.0
            ae_ms_acc = 0.0
            with Timer() as t_edge:
                updates = []
                for i in cohort:
                    u = self._client_round(self.clients[i])
                    updates.append(u)
                    train_ms_acc += getattr(self, "_last_train_ms", 0.0)
                    ae_ms_acc += getattr(self, "_last_ae_ms", 0.0)
            latency_edge_ms = 1000.0 * t_edge.elapsed / max(n_sample, 1)
            latency_train_ms = train_ms_acc / max(n_sample, 1)
            latency_ae_ms = ae_ms_acc / max(n_sample, 1)
            # communication: bytes per client = float32 (4) * total params + 2 (alarm + scale uint16)
            comm_bytes_per_client = 4 * n_params + 2

            # Semi-adaptive Krum-aware mimicry stress test (no-op unless
            # "mimicry" is in attack.types). Runs server-side AFTER all
            # client updates are collected because it needs mu_honest.
            updates, _mimicry_diag = apply_post_collection_mimicry(
                updates, self.attack_plan, round_idx=r
            )
            # Adaptive Krum-survival attack (no-op unless "adaptive_krum" in
            # attack.types). Replaces malicious updates with the strongest
            # perturbation that still survives the top-7 Krum filter (E2).
            updates, _adaptive_diag = apply_post_collection_adaptive_krum(
                updates, self.attack_plan, round_idx=r
            )
            if _adaptive_diag.get("top7_survival_rate") is not None:
                self._adaptive_survival.append(_adaptive_diag["top7_survival_rate"])

            # server aggregation
            ctx = AggregationContext(
                global_state=self.global_model.state_dict(),
                trust_scores=self.trust_scores,
                fl_trust_root_delta=self._maybe_compute_fltrust_root_delta(),
                round_idx=r,
            )
            with Timer() as t_srv:
                agg, info = defense_fn(updates, ctx, self.cfg.defense)
            latency_server_ms = 1000.0 * t_srv.elapsed

            # apply aggregated delta
            new_state = {k: v + agg[k] for k, v in self.global_model.state_dict().items()}
            self.global_model.load_state_dict(new_state)

            # feed back cosines for next round's lazy-AE decision
            for cid, c in info.get("cosines", {}).items():
                self.clients[int(cid)].last_cos = float(c)

            # FRR: fraction of honest clients **hard-rejected** by the defense.
            # Soft down-weighting (e.g., FEDShield's cosine soft weight, FoolsGold's
            # similarity penalty) is *not* a rejection and is excluded — those
            # clients still contribute to the global update.
            honest_ids = [u.client_id for u in updates if not u.is_malicious]
            rejected = set(info.get("rejected", []))
            frr = (sum(1 for cid in honest_ids if cid in rejected)
                   / len(honest_ids)) if honest_ids else 0.0

            # evaluate
            if (r % self.cfg.log_every) == 0 or r == self.cfg.federated.rounds - 1:
                acc, f1 = evaluate_classifier(self.global_model, test_loader, self.device)
                asr, n_asr_eligible, asr_status = evaluate_asr(
                    self.global_model, X_trig, y_clean, target, self.device
                )
                # edge memory footprint: classifier + AE working set in MB
                ae_params = model_param_count(self.clients[0].ae)
                edge_ram_mb = 4.0 * (n_params + ae_params) / (1024.0 ** 2)
                # Canonical defense score = main-task acc * (1 - ASR).
                # NaN-safe: if ASR undefined, propagate as NaN.
                if np.isnan(asr):
                    defense_score = float("nan")
                else:
                    defense_score = float(acc * (1.0 - asr))
                row = dict(
                    round=r, acc=acc, f1=f1,
                    asr=asr, asr_status=asr_status, n_asr_eligible=n_asr_eligible,
                    defense_score=defense_score, frr=frr,
                    latency_edge_ms=latency_edge_ms,
                    latency_server_ms=latency_server_ms,
                    latency_train_ms=latency_train_ms,
                    latency_ae_ms=latency_ae_ms,
                    comm_bytes_per_client=comm_bytes_per_client,
                    edge_ram_mb=edge_ram_mb,
                    rejected=len(rejected),
                    defense=self.cfg.defense.name,
                    dataset=self.cfg.data.name,
                    attack="+".join(self.cfg.attack.types),
                    malicious_ratio=self.cfg.attack.malicious_ratio,
                    n_params=n_params,
                    n_ae_params=ae_params,
                    backdoor_target=target,
                    top7_survival_rate=(float(np.mean(self._adaptive_survival))
                                        if self._adaptive_survival else float("nan")),
                )
                self.logger.log(row)
                history.append(row)

        # persist final config + history
        save_json(os.path.join(self.cfg.out_dir, f"{self.cfg.name}_config.json"), self.cfg.to_dict())
        save_json(os.path.join(self.cfg.out_dir, f"{self.cfg.name}_history.json"), history)
        return {"history": history, "n_params": n_params, "input_shape": list(self.input_shape)}
