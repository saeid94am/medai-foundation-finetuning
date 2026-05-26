from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry

from .prompt_utils import bbox_to_sam_format


class MedSAMLoRA(nn.Module):
    """MedSAM with LoRA adapters injected into the mask decoder attention layers.

    Architecture summary:
        Image Encoder (ViT-B) — FROZEN
        Prompt Encoder          — FROZEN
        Mask Decoder            — LoRA on Q and V projections of every attention layer

    Only the LoRA matrices (A, B per targeted linear layer) and the mask
    prediction head are updated during training.  This keeps trainable
    parameter count under 2 M regardless of backbone size.

    Args:
        checkpoint: Path to ``medsam_vit_b.pth`` weight file.
        cfg: Hydra model config node (``cfg.model``).
    """

    def __init__(self, checkpoint: str, cfg: DictConfig) -> None:
        super().__init__()

        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"MedSAM checkpoint not found at '{checkpoint}'.\n"
                "Download medsam_vit_b.pth from the official MedSAM repository:\n"
                "  https://github.com/bowang-lab/MedSAM\n"
                "The README links to the Google Drive folder containing the checkpoint."
            )

        # MedSAM is a fine-tuned SAM vit_b — same architecture, different weights
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        # Freeze image encoder — 86 M params; no gradients flow through it
        for param in sam.image_encoder.parameters():
            param.requires_grad = False

        # Freeze prompt encoder — encodes bbox / point prompts
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = False

        # Inject LoRA into mask decoder attention Q and V projections
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=list(cfg.lora.target_modules),
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
        )
        sam.mask_decoder = get_peft_model(sam.mask_decoder, lora_config)

        self.sam = sam
        self.img_size = 1024  # SAM always works at 1024×1024

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return a binary logit mask.

        Args:
            image: ``[B, 3, 1024, 1024]`` float tensor in [0, 1].
            bbox:  ``[B, 4]`` bounding box in ``[x_min, y_min, x_max, y_max]``
                   pixel coordinates (1024-px space).

        Returns:
            ``[B, 1, 1024, 1024]`` logit mask (before sigmoid).
        """
        B = image.shape[0]
        device = image.device

        # SAM image encoder expects values in [0, 255]; apply built-in normalisation
        image_255 = image * 255.0

        # Extract image embeddings (no grad — encoder is frozen)
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image_255)  # [B, 256, 64, 64]

        pred_masks = []
        for i in range(B):
            # Encode bounding-box prompt
            box_prompt = bbox_to_sam_format(bbox[i], device)
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=box_prompt,
                masks=None,
            )

            # Decode mask — LoRA params live here
            low_res_mask, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_masks.append(low_res_mask)  # [1, 1, 256, 256]

        # Upsample to 1024×1024 to match GT mask resolution
        low_res = torch.cat(pred_masks, dim=0)  # [B, 1, 256, 256]
        return F.interpolate(
            low_res, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

    # ------------------------------------------------------------------
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
