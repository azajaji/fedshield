"""Shared utilities: seeding, logging, parameter (de)serialisation."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_state_dict(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in sd.values()])


def unflatten_to_state_dict(flat: torch.Tensor, ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    idx = 0
    for k, v in ref.items():
        n = v.numel()
        out[k] = flat[idx : idx + n].reshape(v.shape).to(v.dtype)
        idx += n
    return out


def state_dict_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k] for k in a}


def state_dict_add(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] + b[k] for k in a}


def state_dict_scale(a: Dict[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: a[k] * s for k in a}


class CSVLogger:
    """Append-only CSV logger; opens file lazily on first record."""

    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self._initialised = False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, row: Dict) -> None:
        write_header = not self._initialised and not os.path.exists(self.path)
        mode = "a" if self._initialised or os.path.exists(self.path) else "w"
        with open(self.path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._initialised = True


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


class Timer:
    """Context manager for wall-clock timing."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
