import pytest
import torch

from medai_medsam.models.prompt_utils import bbox_to_sam_format, build_model
from medai_medsam.models.unet import UNet

# ── UNet ─────────────────────────────────────────────────────────────────────

def test_unet_output_shape():
    model = UNet(in_channels=1, out_channels=1, features=[16, 32])
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_unet_forward_cpu_no_crash():
    model = UNet(in_channels=1, out_channels=1, features=[8, 16])
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    assert not torch.isnan(out).any()


def test_unet_output_is_logits_not_probabilities():
    model = UNet(in_channels=1, out_channels=1, features=[8, 16])
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    # Raw logits can exceed [0, 1]; sigmoid output cannot
    assert out.min().item() < 0.0 or out.max().item() > 1.0


# ── prompt_utils ─────────────────────────────────────────────────────────────

def test_bbox_to_sam_format_shape():
    bbox = torch.tensor([10.0, 20.0, 100.0, 150.0])
    result = bbox_to_sam_format(bbox, torch.device("cpu"))
    assert result.shape == (1, 1, 4)


def test_bbox_to_sam_format_values_preserved():
    bbox = torch.tensor([5.0, 15.0, 80.0, 95.0])
    result = bbox_to_sam_format(bbox, torch.device("cpu"))
    assert torch.allclose(result[0, 0], bbox)


# ── build_model factory ───────────────────────────────────────────────────────

def test_build_model_unet(tmp_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "seed": 42,
        "model": {
            "name": "unet_baseline",
            "architecture": {
                "in_channels": 1,
                "out_channels": 1,
                "features": [8, 16],
                "bilinear": False,
            },
        },
    })
    model = build_model(cfg)
    assert isinstance(model, UNet)


def test_build_model_unknown_name_raises():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"model": {"name": "nonexistent_model"}})
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model(cfg)
