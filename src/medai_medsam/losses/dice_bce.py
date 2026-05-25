import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class DiceBCELoss(nn.Module):
    """Weighted sum of soft Dice loss and binary cross-entropy.

    Both terms operate on raw logits.  The Dice term is computed on the
    sigmoid-activated probabilities; BCE uses ``F.binary_cross_entropy_with_logits``
    for numerical stability.

    Args:
        dice_weight: Weight applied to the Dice term.
        bce_weight: Weight applied to the BCE term.
        smooth: Laplace smoothing constant to avoid division by zero.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

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

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def build_loss(cfg: DictConfig) -> nn.Module:
    """Instantiate loss function from Hydra config ``cfg.loss``."""
    if cfg.loss.name == "dice_bce":
        return DiceBCELoss(
            dice_weight=cfg.loss.dice_weight,
            bce_weight=cfg.loss.bce_weight,
            smooth=cfg.loss.smooth,
        )
    raise ValueError(f"Unknown loss '{cfg.loss.name}'.")
