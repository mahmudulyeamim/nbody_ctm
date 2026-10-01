from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


META_KEYS = [
    "event_step", "event_body_a", "event_body_b", "outcome_margin",
    "min_hidden_separation", "max_hidden_radius", "max_hidden_specific_energy",
    "initial_energy", "final_energy", "proposal", "virial_ratio_target",
    "pair_id", "is_counterfactual",
]


@dataclass
class NormalizationStats:
    state_mean: np.ndarray  # [4]
    state_std: np.ndarray   # [4]
    mass_mean: float
    mass_std: float

    def as_dict(self):
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "mass_mean": float(self.mass_mean),
            "mass_std": float(self.mass_std),
        }


def compute_train_stats(npz_path: str | Path) -> NormalizationStats:
    d = np.load(npz_path, allow_pickle=False)
    x = d["train_state"].astype(np.float64)
    m = d["train_mass"].astype(np.float64)
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    std = np.maximum(std, 1e-6)
    return NormalizationStats(mean.astype(np.float32), std.astype(np.float32), float(m.mean()), float(max(m.std(), 1e-6)))


class NBodyDataset(Dataset):
    """Dataset returning normalized trajectories plus raw metadata.

    State layout is [time, body, 4] with channels [x, y, vx, vy].
    Mass is [body].
    """
    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        stats: NormalizationStats,
        *,
        augment: bool = False,
        rotation_augmentation: bool = True,
        body_permutation_augmentation: bool = True,
        seed: int = 0,
    ):
        self.path = str(npz_path)
        self.split = split
        self.stats = stats
        self.augment = augment
        self.rotation_augmentation = rotation_augmentation
        self.body_permutation_augmentation = body_permutation_augmentation
        self.rng = np.random.default_rng(seed)
        d = np.load(npz_path, allow_pickle=False)
        self.state = d[f"{split}_state"]
        self.mass = d[f"{split}_mass"]
        self.label = d[f"{split}_label"]
        self.meta = {k: d[f"{split}_{k}"] for k in META_KEYS if f"{split}_{k}" in d.files}

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        state = self.state[idx].astype(np.float32, copy=True)
        mass = self.mass[idx].astype(np.float32, copy=True)
        if self.augment:
            if self.rotation_augmentation:
                theta = float(self.rng.uniform(0, 2 * np.pi))
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)
                state[..., :2] = state[..., :2] @ R.T
                state[..., 2:] = state[..., 2:] @ R.T
            if self.body_permutation_augmentation:
                p = self.rng.permutation(state.shape[1])
                state = state[:, p]
                mass = mass[p]

        state = (state - self.stats.state_mean) / self.stats.state_std
        mass_n = (mass - self.stats.mass_mean) / self.stats.mass_std
        out = {
            "state": torch.from_numpy(state),
            "mass": torch.from_numpy(mass_n),
            "label": torch.tensor(int(self.label[idx]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        for k, v in self.meta.items():
            value = v[idx]
            if np.issubdtype(np.asarray(value).dtype, np.integer):
                out[k] = torch.tensor(int(value), dtype=torch.long)
            elif np.issubdtype(np.asarray(value).dtype, np.bool_):
                out[k] = torch.tensor(bool(value), dtype=torch.bool)
            else:
                out[k] = torch.tensor(float(value), dtype=torch.float32)
        return out


def make_datasets(npz_path: str | Path, cfg: Dict):
    stats = compute_train_stats(npz_path)
    aug = cfg.get("augmentation", {})
    train = NBodyDataset(
        npz_path, "train", stats, augment=bool(aug.get("enabled", True)),
        rotation_augmentation=bool(aug.get("rotation", True)),
        body_permutation_augmentation=bool(aug.get("permute_bodies", True)),
        seed=int(cfg.get("seed", 42)),
    )
    val = NBodyDataset(npz_path, "val", stats, augment=False)
    test = NBodyDataset(npz_path, "test", stats, augment=False)
    return train, val, test, stats
