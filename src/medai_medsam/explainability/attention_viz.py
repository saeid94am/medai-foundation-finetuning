import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class AttentionVisualizer:
    """Captures and visualises cross-attention maps from MedSAM's mask decoder.

    The mask decoder's ``TwoWayTransformer`` contains cross-attention layers
    where token queries attend to image patch keys.  These attention weights
    show *which image regions influenced each predicted mask token* — a
    built-in explainability signal that requires no post-hoc approximation.

    Usage::

        viz = AttentionVisualizer(model.sam.mask_decoder)
        with viz:
            logits = model(image, bbox)
        heatmap = viz.get_heatmap()   # [H, W] numpy array

    Args:
        decoder: The SAM ``MaskDecoder`` module (after LoRA wrapping).
        layer_idx: Which ``TwoWayAttentionBlock`` to hook.  Defaults to the
                   last layer, which has the most task-specific attention.
    """

    def __init__(self, decoder: nn.Module, layer_idx: int = -1) -> None:
        self.decoder = decoder
        self.layer_idx = layer_idx
        self._hooks: list = []
        self._attn_weights: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        def _hook(module, input, output):
            # output is (attn_output, attn_weights) for nn.MultiheadAttention
            if isinstance(output, tuple) and len(output) == 2:
                self._attn_weights = output[1].detach().cpu()

        target_layer = self._get_target_layer()
        if target_layer is not None:
            self._hooks.append(target_layer.register_forward_hook(_hook))

    def _get_target_layer(self) -> nn.Module | None:
        # Navigate: decoder → transformer → layers[idx] → cross_attn_token_to_image
        try:
            layers = self.decoder.transformer.layers
            block = layers[self.layer_idx]
            return block.cross_attn_token_to_image
        except (AttributeError, IndexError):
            return None

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    def get_heatmap(self, spatial_size: int = 1024) -> np.ndarray | None:
        """Return a ``[spatial_size, spatial_size]`` attention heatmap.

        Averages attention weights across heads and mask tokens, then
        reshapes from the 64×64 SAM patch grid to ``spatial_size``.

        Returns ``None`` if no attention was captured.
        """
        if self._attn_weights is None:
            return None

        # attn_weights: [B, heads, n_tokens, n_patches]
        attn = self._attn_weights[0]  # first batch item
        attn = attn.mean(dim=0).mean(dim=0)  # average over heads and tokens → [n_patches]

        grid_size = int(attn.shape[0] ** 0.5)
        heatmap = attn.reshape(grid_size, grid_size).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Upsample to spatial_size using simple repeat
        scale = spatial_size // grid_size
        heatmap = np.kron(heatmap, np.ones((scale, scale)))
        return heatmap

    def save_overlay(
        self,
        image: np.ndarray,
        save_path: str,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> None:
        """Overlay the attention heatmap on the original image and save.

        Args:
            image:     ``[H, W, 3]`` uint8 RGB image.
            save_path: Output PNG path.
            alpha:     Heatmap opacity (0 = invisible, 1 = fully opaque).
        """
        heatmap = self.get_heatmap(spatial_size=image.shape[0])
        if heatmap is None:
            return

        cmap = plt.get_cmap(colormap)
        colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        overlay = (alpha * colored + (1 - alpha) * image).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[1].imshow(heatmap, cmap=colormap)
        axes[1].set_title("Attention heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
