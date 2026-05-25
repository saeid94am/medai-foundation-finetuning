import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class SegmentationMetrics:
    """Aggregates Dice, HD95, and IoU over a validation / test epoch.

    Uses MONAI's ``DiceMetric`` and ``HausdorffDistanceMetric`` so results
    are directly comparable to other MONAI-based segmentation papers.

    Usage::

        metrics = SegmentationMetrics(device=device)
        for batch in loader:
            pred_mask = model(...)
            pred_binary = (torch.sigmoid(pred_mask) > 0.5).long()
            metrics.update(pred_binary, batch["mask"].long())
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self._dice = DiceMetric(include_background=False, reduction="mean")
        self._hd95 = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )
        self._iou_sum = 0.0
        self._count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            preds:   ``[B, 1, H, W]`` integer tensor with values in {0, 1}.
            targets: ``[B, 1, H, W]`` integer tensor with values in {0, 1}.
        """
        preds = preds.to(self.device)
        targets = targets.to(self.device)

        self._dice(y_pred=preds, y=targets)
        try:
            self._hd95(y_pred=preds, y=targets)
        except Exception:
            pass  # HD95 raises if a batch has no foreground in pred or target

        # Micro IoU
        intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
        union = (preds | targets).sum(dim=(1, 2, 3)).float()
        valid = union > 0
        if valid.any():
            self._iou_sum += (intersection[valid] / union[valid]).sum().item()
            self._count += valid.sum().item()

    def compute(self) -> dict[str, float]:
        dice = self._dice.aggregate().item()
        hd95_agg = self._hd95.aggregate()
        hd95 = hd95_agg.item() if hd95_agg is not None else float("nan")
        iou = self._iou_sum / self._count if self._count > 0 else float("nan")
        return {"dice": dice, "hd95": hd95, "iou": iou}

    def reset(self) -> None:
        self._dice.reset()
        self._hd95.reset()
        self._iou_sum = 0.0
        self._count = 0
