from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import mask_to_bbox


class BUSIDataset(Dataset):
    """BUSI breast ultrasound dataset for MedSAM prompt-guided segmentation.

    Expected directory layout::

        root/
        ├── benign/
        │   ├── benign (1).png
        │   ├── benign (1)_mask.png
        │   └── ...
        ├── malignant/
        │   └── ...
        └── normal/          # excluded by default (no lesion)
            └── ...

    Args:
        root: Path to the BUSI root directory.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        image_size: Spatial size fed to the model (1024 for MedSAM, 256 for UNet).
        classes: Subset of classes to include. Defaults to benign + malignant.
        transform: Albumentations transform applied before resizing.
        bbox_noise_px: Random perturbation (±px) added to the GT bounding box at train time.
        train_split: Fraction of data used for training.
        val_split: Fraction used for validation (remainder is test).
        seed: Random seed for the stratified split.
    """

    CLASSES = ["benign", "malignant"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 1024,
        classes: Optional[List[str]] = None,
        transform=None,
        bbox_noise_px: int = 0,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.classes = classes or self.CLASSES
        self.transform = transform
        self.bbox_noise_px = bbox_noise_px if split == "train" else 0

        all_samples = self._build_sample_list()
        self.samples = self._stratified_split(all_samples, split, train_split, val_split, seed)

    # ------------------------------------------------------------------
    def _build_sample_list(self) -> List[Dict]:
        samples = []
        for label_idx, cls in enumerate(self.classes):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            image_files = sorted(f for f in cls_dir.glob("*.png") if "_mask" not in f.name)
            for img_path in image_files:
                mask_path = img_path.with_name(img_path.stem + "_mask.png")
                if not mask_path.exists():
                    continue
                samples.append(
                    {
                        "image": img_path,
                        "mask": mask_path,
                        "label": label_idx,
                        "class_name": cls,
                    }
                )
        return samples

    def _stratified_split(
        self,
        samples: List[Dict],
        split: str,
        train_frac: float,
        val_frac: float,
        seed: int,
    ) -> List[Dict]:
        labels = [s["label"] for s in samples]
        test_frac = 1.0 - train_frac - val_frac
        train_val, test = train_test_split(
            samples, test_size=test_frac, stratify=labels, random_state=seed
        )
        adjusted_val = val_frac / (train_frac + val_frac)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val,
            stratify=[s["label"] for s in train_val],
            random_state=seed,
        )
        return {"train": train, "val": val, "test": test}[split]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = cv2.imread(str(sample["image"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # SAM expects 3-channel input

        mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        bbox = mask_to_bbox(mask, noise_px=self.bbox_noise_px)

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "label": sample["label"],
            "class_name": sample["class_name"],
            "path": str(sample["image"]),
        }
