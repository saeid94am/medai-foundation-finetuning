import torch
from omegaconf import DictConfig


def bbox_to_sam_format(bbox: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a ``[4]`` bbox tensor to the shape SAM's prompt encoder expects.

    SAM prompt encoder requires boxes as ``[B, 1, 4]`` with values in
    ``[x_min, y_min, x_max, y_max]`` pixel coordinates in the 1024-px space.

    Args:
        bbox: ``[4]`` float tensor ``[x_min, y_min, x_max, y_max]``.
        device: Target device.

    Returns:
        ``[1, 1, 4]`` float tensor.
    """
    return bbox.to(device).unsqueeze(0).unsqueeze(0)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Factory that instantiates the correct model from a Hydra config.

    The ``cfg.model.name`` field selects which class to instantiate:
    - ``medsam_lora``         → MedSAMLoRA
    - ``medsam_linear_probe`` → MedSAMLoRA with all decoder layers frozen except head
    - ``medsam_full_finetune``→ MedSAMLoRA with no frozen layers
    - ``unet_baseline``       → UNet

    Args:
        cfg: Top-level Hydra config (not just the model sub-node).

    Returns:
        Instantiated, un-compiled model on CPU.
    """
    from medai_medsam.models.medsam_lora import MedSAMLoRA
    from medai_medsam.models.unet import UNet

    name = cfg.model.name

    if name == "unet_baseline":
        return UNet(
            in_channels=cfg.model.architecture.in_channels,
            out_channels=cfg.model.architecture.out_channels,
            features=list(cfg.model.architecture.features),
            bilinear=cfg.model.architecture.bilinear,
        )

    if name in {"medsam_lora", "medsam_linear_probe", "medsam_full_finetune"}:
        return MedSAMLoRA(checkpoint=cfg.model.checkpoint, cfg=cfg.model)

    raise ValueError(
        f"Unknown model name '{name}'. Expected one of: medsam_lora, medsam_linear_probe, medsam_full_finetune, unet_baseline."
    )
