import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .transforms import get_train_transforms, get_val_transforms


def mask_to_bbox(mask: np.ndarray, noise_px: int = 0) -> list[int]:
    """Derive a bounding box from a binary mask.

    Returns ``[x_min, y_min, x_max, y_max]`` in pixel coordinates.
    If the mask is empty (no foreground), returns the full image extent.
    ``noise_px`` adds random uniform perturbation to each edge — use at
    train time to prevent the model from over-fitting to exact GT boxes.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        h, w = mask.shape
        return [0, 0, w, h]

    h, w = mask.shape
    r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    if noise_px > 0:
        r_min = max(0, r_min - np.random.randint(0, noise_px + 1))
        c_min = max(0, c_min - np.random.randint(0, noise_px + 1))
        r_max = min(h - 1, r_max + np.random.randint(0, noise_px + 1))
        c_max = min(w - 1, c_max + np.random.randint(0, noise_px + 1))

    # Return as [x_min, y_min, x_max, y_max] (SAM prompt convention)
    return [c_min, r_min, c_max, r_max]


def build_dataloaders(cfg: DictConfig):
    """Construct train and validation DataLoaders from a Hydra config."""
    from .dataset import BUSIDataset  # local import avoids circular dependency

    train_transform = get_train_transforms(cfg.data.augmentation)
    val_transform = get_val_transforms()

    common = dict(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        classes=list(cfg.data.classes),
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        seed=cfg.seed,
    )

    train_ds = BUSIDataset(
        **common,
        split="train",
        transform=train_transform,
        bbox_noise_px=cfg.data.bbox_noise_px,
    )
    val_ds = BUSIDataset(**common, split="val", transform=val_transform)

    loader_kwargs = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader
