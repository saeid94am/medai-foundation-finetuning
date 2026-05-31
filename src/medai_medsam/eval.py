"""Evaluation entry point.

Run with:
    python -m medai_medsam.eval checkpoint=results/checkpoints/medsam_lora_best.pth
"""

import csv
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from medai_medsam.data.dataset import BUSIDataset
from medai_medsam.data.transforms import get_val_transforms
from medai_medsam.metrics import SegmentationMetrics
from medai_medsam.models import build_model


def keep_largest_component(pred: torch.Tensor) -> torch.Tensor:
    """Replace each prediction mask with its largest connected component.

    Spurious blobs far from the main lesion inflate HD95 without meaningfully
    changing Dice.  Keeping only the largest component removes them.

    Args:
        pred: ``[B, 1, H, W]`` binary tensor (values 0 or 1, dtype long).

    Returns:
        Same shape tensor with only the largest component per sample retained.
    """
    out = pred.clone()
    for b in range(pred.shape[0]):
        mask_np = pred[b, 0].cpu().numpy().astype(np.uint8)
        if mask_np.max() == 0:
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
        if n_labels <= 1:
            continue
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out[b, 0] = torch.from_numpy((labels == largest).astype(np.int64)).to(pred.device)
    return out


def _dice_per_sample(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    intersection = (pred_np * gt_np).sum()
    return float(2 * intersection + 1e-5) / float(pred_np.sum() + gt_np.sum() + 1e-5)


def _iou_per_sample(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    intersection = (pred_np * gt_np).sum()
    union = pred_np.sum() + gt_np.sum() - intersection
    return float(intersection + 1e-5) / float(union + 1e-5)


def _hd95_per_sample(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt

    if pred_np.max() == 0 and gt_np.max() == 0:
        return 0.0
    if pred_np.max() == 0 or gt_np.max() == 0:
        return float(max(pred_np.shape))

    pred_border = pred_np ^ cv2.erode(pred_np.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
        bool
    )
    gt_border = gt_np ^ cv2.erode(gt_np.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    d1 = dt_gt[pred_border] if pred_border.any() else np.array([0.0])
    d2 = dt_pred[gt_border] if gt_border.any() else np.array([0.0])
    return float(np.percentile(np.concatenate([d1, d2]), 95))


@hydra.main(config_path="../../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.checkpoint, map_location=device)
    train_cfg = OmegaConf.create(ckpt["cfg"])
    model = build_model(train_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    # Collect per-sample data for CSV and visualisation
    samples = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            bbox = batch["bbox"].to(device)

            logits = model(image, bbox)
            preds = (torch.sigmoid(logits) > cfg.postprocess.threshold).long()
            if cfg.postprocess.keep_largest_component:
                preds = keep_largest_component(preds)

            metrics.update(preds, mask.long())

            for i, cls_name in enumerate(batch["class_name"]):
                per_class_metrics[cls_name].update(
                    preds[i].unsqueeze(0), mask[i].unsqueeze(0).long()
                )

            for i in range(preds.shape[0]):
                pred_np = preds[i, 0].cpu().numpy()
                gt_np = mask[i, 0].cpu().numpy()
                img_np = (batch["image"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                samples.append(
                    {
                        "path": batch["path"][i],
                        "class": batch["class_name"][i],
                        "image": img_np,
                        "gt": gt_np,
                        "pred": pred_np,
                        "dice": _dice_per_sample(pred_np, gt_np),
                        "iou": _iou_per_sample(pred_np, gt_np),
                        "hd95": _hd95_per_sample(pred_np, gt_np),
                    }
                )

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

    # Save CSV — one row per test image with per-sample metrics
    results_path = Path(cfg.output.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "dice", "hd95", "iou"])
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "path": s["path"],
                    "class": s["class"],
                    "dice": f"{s['dice']:.4f}",
                    "hd95": f"{s['hd95']:.4f}",
                    "iou": f"{s['iou']:.4f}",
                }
            )
    print(f"\nResults saved to {results_path}")

    if cfg.output.save_predictions:
        pred_dir = Path(cfg.output.predictions_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        n_samples = cfg.output.get("n_overlay_samples", 5)
        worst_samples = sorted(samples, key=lambda x: x["dice"])[:n_samples]
        _save_prediction_overlays(worst_samples, pred_dir)
        _save_prediction_grid(
            samples,
            output_dir / "prediction_grid.png",
            n_worst=cfg.output.get("n_worst", 6),
            n_best=cfg.output.get("n_best", 6),
        )
        print(f"Visualisations saved to {output_dir}")


def _save_prediction_overlays(samples: list, pred_dir: Path) -> None:
    """Save individual overlay PNGs for the given samples (worst-Dice cases)."""
    for s in samples:
        pred_rgb = cv2.cvtColor((s["pred"] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(s["image"], 0.7, pred_rgb, 0.3, 0)
        stem = Path(s["path"]).stem
        cv2.imwrite(str(pred_dir / f"{stem}_pred.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(pred_dir / f"{stem}_gt.png"),
            (s["gt"] * 255).astype(np.uint8),
        )


def _save_prediction_grid(
    samples: list,
    out_path: Path,
    n_worst: int = 6,
    n_best: int = 6,
) -> None:
    """Save a matplotlib figure grid: worst-Dice cases on top, best on bottom.

    Each row shows one case: Image | GT contour | Prediction contour | Overlay.
    This is the figure to include in a portfolio or paper supplementary.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    sorted_samples = sorted(samples, key=lambda x: x["dice"])
    worst = sorted_samples[:n_worst]
    best = sorted_samples[-n_best:][::-1]
    groups = [("Worst predictions (lowest Dice)", worst), ("Best predictions (highest Dice)", best)]

    cols = 4
    col_titles = ["Ultrasound", "Ground truth", "Prediction", "Overlay"]
    fig_rows = sum(len(g) for _, g in groups)

    fig, axes = plt.subplots(
        fig_rows, cols, figsize=(cols * 3, fig_rows * 3), constrained_layout=True
    )
    if fig_rows == 1:
        axes = axes[np.newaxis, :]

    row_idx = 0
    for group_title, group in groups:
        for s in group:
            img = s["image"]
            gt_contours, _ = cv2.findContours(
                s["gt"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pred_contours, _ = cv2.findContours(
                s["pred"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            gt_overlay = img.copy()
            cv2.drawContours(gt_overlay, gt_contours, -1, (0, 255, 0), 2)

            pred_overlay = img.copy()
            cv2.drawContours(pred_overlay, pred_contours, -1, (255, 0, 0), 2)

            both_overlay = img.copy()
            cv2.drawContours(both_overlay, gt_contours, -1, (0, 255, 0), 2)
            cv2.drawContours(both_overlay, pred_contours, -1, (255, 0, 0), 2)

            panels = [img, gt_overlay, pred_overlay, both_overlay]
            for col, panel in enumerate(panels):
                ax = axes[row_idx, col]
                ax.imshow(panel)
                ax.axis("off")
                if col == 0:
                    ax.set_title(
                        f"{s['class']}  Dice={s['dice']:.3f}",
                        fontsize=8,
                        loc="left",
                        pad=2,
                    )
                elif row_idx == 0:
                    ax.set_title(col_titles[col], fontsize=8)

            row_idx += 1

    # Group section labels
    row_idx = 0
    for group_title, group in groups:
        axes[row_idx, 0].annotate(
            group_title,
            xy=(0, 1),
            xycoords="axes fraction",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="steelblue", alpha=0.8),
            va="bottom",
        )
        row_idx += len(group)

    gt_patch = mpatches.Patch(color=(0, 1, 0), label="Ground truth")
    pred_patch = mpatches.Patch(color=(1, 0, 0), label="Prediction")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center", ncol=2, fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
