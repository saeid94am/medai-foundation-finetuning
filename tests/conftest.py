"""Shared pytest fixtures.

All fixtures create purely synthetic data so the test suite runs in CI
without any downloaded datasets or model checkpoints.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def busi_root(tmp_path_factory) -> Path:
    """Minimal BUSI-like directory: 10 images per class, 200×200 px.

    10 per class (20 total) satisfies sklearn's stratified-split requirement
    that each split contains at least n_classes=2 samples at 80/10/10 fractions.
    """
    root = tmp_path_factory.mktemp("busi")
    rng = np.random.default_rng(0)

    for cls in ["benign", "malignant"]:
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(1, 11):
            # Grayscale ultrasound-like image
            img = rng.integers(0, 255, (200, 200), dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"{cls} ({i}).png"), img)

            # Binary mask — filled rectangle in centre
            mask = np.zeros((200, 200), dtype=np.uint8)
            mask[60:140, 60:140] = 255
            cv2.imwrite(str(cls_dir / f"{cls} ({i})_mask.png"), mask)

    return root


@pytest.fixture
def random_logits() -> torch.Tensor:
    """[2, 1, 64, 64] float logits."""
    torch.manual_seed(0)
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def random_mask() -> torch.Tensor:
    """[2, 1, 64, 64] binary mask in {0, 1}."""
    torch.manual_seed(1)
    return (torch.rand(2, 1, 64, 64) > 0.5).float()
