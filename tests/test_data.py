import numpy as np

from medai_medsam.data.dataset import BUSIDataset
from medai_medsam.data.transforms import get_train_transforms, get_val_transforms
from medai_medsam.data.utils import mask_to_bbox

# ── mask_to_bbox ──────────────────────────────────────────────────────────────


def test_mask_to_bbox_basic():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 1
    bbox = mask_to_bbox(mask)
    assert bbox == [30, 20, 69, 59], f"Unexpected bbox: {bbox}"


def test_mask_to_bbox_empty_returns_full_extent():
    mask = np.zeros((80, 120), dtype=np.uint8)
    bbox = mask_to_bbox(mask)
    assert bbox == [0, 0, 120, 80]


def test_mask_to_bbox_noise_stays_in_bounds():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    for _ in range(20):
        bbox = mask_to_bbox(mask, noise_px=10)
        x_min, y_min, x_max, y_max = bbox
        assert x_min >= 0 and y_min >= 0
        assert x_max <= 99 and y_max <= 99


# ── transforms ───────────────────────────────────────────────────────────────


def test_train_transforms_not_none():
    from omegaconf import OmegaConf

    aug_cfg = OmegaConf.create(
        {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.3,
            "rotate_limit": 15,
            "brightness_contrast_prob": 0.3,
            "gaussian_noise_prob": 0.2,
        }
    )
    t = get_train_transforms(aug_cfg)
    assert t is not None


def test_val_transforms_is_none():
    assert get_val_transforms() is None


# ── BUSIDataset ───────────────────────────────────────────────────────────────


def test_dataset_length(busi_root):
    ds = BUSIDataset(root=str(busi_root), split="train", image_size=64, seed=0)
    # 10 images × 2 classes = 20 total; 80% train = 16 samples
    assert len(ds) > 0


def test_dataset_item_keys(busi_root):
    ds = BUSIDataset(root=str(busi_root), split="train", image_size=64, seed=0)
    item = ds[0]
    for key in ("image", "mask", "bbox", "label", "class_name", "path"):
        assert key in item, f"Missing key: {key}"


def test_dataset_image_shape(busi_root):
    ds = BUSIDataset(root=str(busi_root), split="train", image_size=64, seed=0)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert item["bbox"].shape == (4,)


def test_dataset_splits_are_disjoint(busi_root):
    common = dict(root=str(busi_root), image_size=64, seed=0)
    train_paths = {s["path"] for s in BUSIDataset(**common, split="train").samples}
    val_paths = {s["path"] for s in BUSIDataset(**common, split="val").samples}
    test_paths = {s["path"] for s in BUSIDataset(**common, split="test").samples}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_build_dataloaders_returns_two_loaders(busi_root):
    from omegaconf import OmegaConf

    from medai_medsam.data.utils import build_dataloaders

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "data": {
                "root": str(busi_root),
                "image_size": 64,
                "classes": ["benign", "malignant"],
                "train_split": 0.8,
                "val_split": 0.1,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "bbox_noise_px": 0,
                "augmentation": {
                    "horizontal_flip_prob": 0.5,
                    "vertical_flip_prob": 0.0,
                    "rotate_limit": 10,
                    "brightness_contrast_prob": 0.2,
                    "gaussian_noise_prob": 0.1,
                },
            },
        }
    )
    train_loader, val_loader = build_dataloaders(cfg)
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
