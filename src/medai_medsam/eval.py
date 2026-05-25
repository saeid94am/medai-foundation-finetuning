"""Evaluation entry point.

Run with:
    python -m medai_medsam.eval checkpoint=results/checkpoints/medsam_lora_best.pth
"""

import csv
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from medai_medsam.data.dataset import BUSIDataset
from medai_medsam.data.transforms import get_val_transforms
from medai_medsam.metrics import SegmentationMetrics
from medai_medsam.models import build_model


@hydra.main(config_path="../../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and reconstruct model
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    train_cfg = OmegaConf.create(ckpt["cfg"])
    model = build_model(train_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build test dataset
    test_ds = BUSIDataset(
        root=cfg.data.root,
        split=cfg.data.split,
        image_size=cfg.data.image_size,
        classes=list(cfg.data.classes),
        transform=get_val_transforms(),
        bbox_noise_px=0,
        train_split=train_cfg.data.train_split,
        val_split=train_cfg.data.val_split,
        seed=train_cfg.seed,
    )
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    metrics = SegmentationMetrics(device=device)
    per_class_metrics = {cls: SegmentationMetrics(device=device) for cls in cfg.data.classes}

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(cfg.output.predictions_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            bbox = batch["bbox"].to(device)

            logits = model(image, bbox)
            preds = (torch.sigmoid(logits) > 0.5).long()

            metrics.update(preds, mask.long())

            for i, cls_name in enumerate(batch["class_name"]):
                per_class_metrics[cls_name].update(
                    preds[i].unsqueeze(0), mask[i].unsqueeze(0).long()
                )

            if cfg.output.save_predictions:
                _save_prediction_overlays(batch, preds, pred_dir)

            for i in range(len(batch["path"])):
                rows.append({"path": batch["path"][i], "class": batch["class_name"][i]})

    overall = metrics.compute()
    print("\n=== Overall Test Results ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    if cfg.metrics.per_class:
        for cls_name, m in per_class_metrics.items():
            r = m.compute()
            print(f"\n=== {cls_name} ===")
            for k, v in r.items():
                print(f"  {k}: {v:.4f}")

    # Save CSV
    results_path = Path(cfg.output.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class"] + list(overall.keys()))
        writer.writeheader()
        for row in rows:
            row.update(overall)
            writer.writerow(row)

    print(f"\nResults saved to {results_path}")


def _save_prediction_overlays(batch, preds, pred_dir: Path) -> None:
    import cv2
    import numpy as np

    for i in range(len(batch["path"])):
        img = (batch["image"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = (batch["mask"][i, 0].numpy() * 255).astype(np.uint8)
        pred = (preds[i, 0].cpu().numpy() * 255).astype(np.uint8)

        overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB), 0.3, 0)
        stem = Path(batch["path"][i]).stem
        cv2.imwrite(str(pred_dir / f"{stem}_pred.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pred_dir / f"{stem}_gt.png"), gt)


if __name__ == "__main__":
    main()
