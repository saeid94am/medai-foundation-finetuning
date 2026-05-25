# MedSAM LoRA — BUSI Segmentation

> Phase 3 of the Medical AI PhD Portfolio — Foundation Model Fine-Tuning

## What this project does

This repository fine-tunes **MedSAM** (Ma et al., *Nature Communications* 2024) on the
**BUSI** breast ultrasound dataset using **LoRA** (Low-Rank Adaptation), a parameter-efficient
fine-tuning technique that updates fewer than 1% of model weights.

The central question: *can LoRA-adapted MedSAM overcome the label-scarcity bottleneck while
retaining the interactive, bounding-box-prompted workflow required in clinical practice?*

## Key results

| Model | Trainable params | Dice ↑ | HD95 ↓ |
|---|---|---|---|
| Zero-shot MedSAM | 0 | TBD | TBD |
| UNet (scratch) | ~31 M | TBD | TBD |
| MedSAM LoRA (r=8) | ~2 M | TBD | TBD |
| MedSAM Full fine-tune | ~308 M | TBD | TBD |

## Quick links

- [GitHub repository](https://github.com/saeid-amini/medai-foundation-finetuning)
- [W&B experiment report](#) *(link added after training)*
- [Quickstart guide](quickstart.md)
