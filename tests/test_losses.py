import torch

from medai_medsam.losses.dice_bce import DiceBCELoss, _boundary_weight_map


def test_loss_is_scalar(random_logits, random_mask):
    criterion = DiceBCELoss()
    loss = criterion(random_logits, random_mask)
    assert loss.shape == ()


def test_loss_perfect_prediction_is_near_zero():
    # When logits are very large positives where mask=1 and very large negatives where mask=0,
    # Dice loss → 0 and BCE loss → 0.
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 8:24, 8:24] = 1.0
    logits = (mask * 20.0) - ((1 - mask) * 20.0)  # +20 where foreground, -20 where background
    criterion = DiceBCELoss()
    loss = criterion(logits, mask)
    assert loss.item() < 0.05, f"Expected loss < 0.05, got {loss.item():.4f}"


def test_loss_all_zero_prediction_is_positive(random_mask):
    logits = torch.full_like(random_mask, -10.0)  # all background prediction
    criterion = DiceBCELoss()
    loss = criterion(logits, random_mask)
    assert loss.item() > 0.0


def test_loss_weights_sum_effect():
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    logits = torch.zeros_like(mask)

    dice_only = DiceBCELoss(dice_weight=1.0, bce_weight=0.0)(logits, mask)
    bce_only = DiceBCELoss(dice_weight=0.0, bce_weight=1.0)(logits, mask)
    combined = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)(logits, mask)

    assert abs(combined.item() - 0.5 * (dice_only.item() + bce_only.item())) < 1e-5


def test_build_loss_factory_returns_criterion():
    from omegaconf import OmegaConf

    from medai_medsam.losses import build_loss

    cfg = OmegaConf.create(
        {"loss": {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5, "smooth": 1e-5}}
    )
    criterion = build_loss(cfg)
    assert isinstance(criterion, DiceBCELoss)


def test_boundary_weight_map_highlights_edges():
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 8:24, 8:24] = 1.0
    weight_map = _boundary_weight_map(mask, kernel_size=3)
    # Interior pixels should have weight 1.0, boundary pixels 2.0
    interior = weight_map[0, 0, 12:20, 12:20]
    assert interior.max().item() == 1.0
    assert weight_map.max().item() == 2.0


def test_boundary_loss_reduces_hd_sensitive_errors():
    # A prediction that gets the blob right but shifts the boundary outward
    # should incur a higher loss with boundary_weight=1 than without.
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 8:24, 8:24] = 1.0
    shifted = torch.zeros(1, 1, 32, 32)
    shifted[0, 0, 6:26, 6:26] = 1.0  # 2px dilation — correct interior, wrong boundary
    logits = (shifted * 10.0) - ((1 - shifted) * 10.0)

    loss_no_boundary = DiceBCELoss(boundary_weight=0.0)(logits, mask)
    loss_with_boundary = DiceBCELoss(boundary_weight=1.0)(logits, mask)
    assert loss_with_boundary.item() > loss_no_boundary.item()


def test_build_loss_unknown_raises():
    import pytest
    from omegaconf import OmegaConf

    from medai_medsam.losses import build_loss

    cfg = OmegaConf.create({"loss": {"name": "unknown_loss"}})
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(cfg)
