import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def _boundary_weight_map(targets: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Return a weight map that is 2.0 on boundary pixels and 1.0 elsewhere.

    Boundary pixels are identified by the difference between a morphological
    dilation and erosion of the GT mask — a purely PyTorch operation with no
    extra dependencies.
    """
    pad = kernel_size // 2
    dilated = F.max_pool2d(targets, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-targets, kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return 1.0 + boundary  # 1.0 everywhere, 2.0 on the boundary ring


class DiceBCELoss(nn.Module):
    """Weighted sum of soft Dice loss and boundary-aware binary cross-entropy.

    Both terms operate on raw logits.  The Dice term is computed on the
    sigmoid-activated probabilities; BCE uses ``F.binary_cross_entropy_with_logits``
    for numerical stability.

    When ``boundary_weight > 0`` the BCE term is multiplied by a per-pixel
    weight map that is ``(1 + boundary_weight)`` on GT-mask boundary pixels and
    1.0 elsewhere.  This forces the model to pay extra attention to contour
    accuracy, which directly reduces HD95 on irregular malignant tumors.

    Args:
        dice_weight: Weight applied to the Dice term.
        bce_weight: Weight applied to the BCE term.
        smooth: Laplace smoothing constant to avoid division by zero.
        boundary_weight: Extra weight on boundary pixels in BCE (0 = disabled).
        boundary_kernel_size: Morphological kernel size for boundary extraction.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-5,
        boundary_weight: float = 0.0,
        boundary_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.boundary_weight = boundary_weight
        self.boundary_kernel_size = boundary_kernel_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits:  ``[B, 1, H, W]`` raw model output (before sigmoid).
            targets: ``[B, 1, H, W]`` binary ground-truth mask in {0, 1}.

        Returns:
            Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)

        flat_p = probs.reshape(-1)
        flat_t = targets.reshape(-1)
        intersection = (flat_p * flat_t).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            flat_p.sum() + flat_t.sum() + self.smooth
        )

        if self.boundary_weight > 0.0:
            # Boundary pixels get (1 + boundary_weight)× the BCE loss weight.
            # Gradient flows only through logits; targets is treated as constant.
            weight_map = _boundary_weight_map(targets.detach(), self.boundary_kernel_size)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weight_map
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def build_loss(cfg: DictConfig) -> nn.Module:
    """Instantiate loss function from Hydra config ``cfg.loss``."""
    if cfg.loss.name == "dice_bce":
        return DiceBCELoss(
            dice_weight=cfg.loss.dice_weight,
            bce_weight=cfg.loss.bce_weight,
            smooth=cfg.loss.smooth,
            boundary_weight=cfg.loss.get("boundary_weight", 0.0),
            boundary_kernel_size=cfg.loss.get("boundary_kernel_size", 5),
        )
    raise ValueError(f"Unknown loss '{cfg.loss.name}'.")
