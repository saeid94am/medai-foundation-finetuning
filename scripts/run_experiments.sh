#!/usr/bin/env bash
# End-to-end experiment reproduction script.
#
# Runs all four model variants across 3 seeds, then evaluates each checkpoint
# and writes results/metrics/test_results.csv.
#
# Prerequisites:
#   1. conda activate medai-medsam   (or Docker image)
#   2. data/BUSI/  populated via scripts/download_data.sh
#   3. checkpoints/medsam_vit_b.pth  present
#
# Usage:
#   bash scripts/run_experiments.sh          # full suite
#   bash scripts/run_experiments.sh --quick  # LoRA only, 1 seed

set -euo pipefail

QUICK=false
[[ "${1:-}" == "--quick" ]] && QUICK=true

SEEDS="0,1,2"
[[ "$QUICK" == true ]] && SEEDS="0"

echo "======================================================="
echo " MedSAM LoRA — BUSI Experiment Suite"
echo " Seeds: $SEEDS  |  Quick mode: $QUICK"
echo "======================================================="

# ── Helper ────────────────────────────────────────────────────────────────
run_train() {
    local MODEL=$1
    echo ""
    echo "--- Training: $MODEL (seeds=$SEEDS) ---"
    python -m medai_medsam.train \
        model="$MODEL" \
        seed="$SEEDS" \
        --multirun
}

run_eval() {
    local MODEL=$1
    local CKPT="results/checkpoints/${MODEL}_best.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo "[WARN] Checkpoint not found: $CKPT — skipping eval for $MODEL"
        return
    fi
    echo ""
    echo "--- Evaluating: $MODEL ---"
    python -m medai_medsam.eval checkpoint="$CKPT"
}

# ── Experiments ───────────────────────────────────────────────────────────

# 1. UNet from scratch (no checkpoint needed)
run_train unet_baseline
run_eval  unet_baseline

if [[ "$QUICK" == false ]]; then
    # 2. Linear probe (fastest MedSAM variant)
    run_train medsam_linear_probe
    run_eval  medsam_linear_probe
fi

# 3. LoRA fine-tuning (primary experiment)
run_train medsam_lora
run_eval  medsam_lora

if [[ "$QUICK" == false ]]; then
    # 4. Full fine-tuning (upper bound; requires ≥10 GB VRAM)
    VRAM=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory/1e9))" 2>/dev/null || echo 0)
    if [[ "$VRAM" -ge 10 ]]; then
        run_train medsam_full_finetune
        run_eval  medsam_full_finetune
    else
        echo ""
        echo "[SKIP] Full fine-tune skipped — detected ${VRAM} GB VRAM (need ≥10 GB)."
        echo "       Run on Kaggle (P100 16 GB) or Colab (A100) to reproduce this result."
    fi
fi

echo ""
echo "======================================================="
echo " All experiments complete."
echo " Results: results/metrics/test_results.csv"
echo " W&B runs: results/metrics/wandb_runs.txt"
echo "======================================================="
