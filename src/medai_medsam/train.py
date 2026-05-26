"""Training entry point.

Run with:
    python -m medai_medsam.train
    python -m medai_medsam.train model=unet_baseline
    python -m medai_medsam.train training.max_epochs=50 seed=0,1,2 --multirun
    python -m medai_medsam.train training.resume_from=results/checkpoints/medsam_lora_last.pth
"""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from medai_medsam.data import build_dataloaders
from medai_medsam.losses import build_loss
from medai_medsam.metrics import SegmentationMetrics
from medai_medsam.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    o = cfg.optimizer
    if o.name == "adamw":
        return torch.optim.AdamW(params, lr=o.lr, weight_decay=o.weight_decay, betas=tuple(o.betas))
    raise ValueError(f"Unknown optimizer '{o.name}'.")


def build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    s = cfg.scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=s.warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s.T_max, eta_min=s.eta_min)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[s.warmup_epochs]
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, cfg, device, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        bbox = batch["bbox"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
            logits = model(image, bbox)
            loss = criterion(logits, mask) / cfg.training.accumulate_grad_batches

        scaler.scale(loss).backward()

        if (step + 1) % cfg.training.accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.training.gradient_clip_val,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.training.accumulate_grad_batches

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, metrics, device, cfg):
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        bbox = batch["bbox"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
            logits = model(image, bbox)
            loss = criterion(logits, mask)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        metrics.update(preds, mask.long())

    results = metrics.compute()
    results["loss"] = total_loss / len(loader)
    return results


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.logging.wandb.tags),
    )

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    if cfg.training.gradient_checkpointing and hasattr(model, "sam"):
        model.sam.image_encoder.gradient_checkpointing_enable()

    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)
    val_metrics = SegmentationMetrics(device=device)

    ckpt_dir = Path(cfg.checkpointing.dirpath)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    patience_counter = 0
    start_epoch = 0

    if cfg.training.resume_from:
        ckpt = torch.load(cfg.training.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_dice = ckpt["best_dice"]
        patience_counter = ckpt["patience_counter"]
        start_epoch = ckpt["epoch"] + 1
        print(
            f"Resumed from '{cfg.training.resume_from}' (epoch {ckpt['epoch']}). Best val Dice so far: {best_dice:.4f}"
        )

    for epoch in range(start_epoch, cfg.training.max_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, cfg, device, epoch
        )
        val_results = validate(model, val_loader, criterion, val_metrics, device, cfg)
        scheduler.step()

        log = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_results["loss"],
            "val/dice": val_results["dice"],
            "val/hd95": val_results["hd95"],
            "val/iou": val_results["iou"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(log)

        if val_results["dice"] > best_dice:
            best_dice = val_results["dice"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_dice,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_dir / f"{cfg.model.name}_best.pth",
            )
        else:
            patience_counter += 1

        # Always overwrite the last checkpoint so training can be resumed after
        # a Kaggle session timeout or connection drop.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_dice": best_dice,
                "patience_counter": patience_counter,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_dir / f"{cfg.model.name}_last.pth",
        )

        if patience_counter >= cfg.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best val Dice: {best_dice:.4f}")
            break

    wandb.finish()
    # Record W&B run URL for reproducibility
    with open("results/metrics/wandb_runs.txt", "a") as f:
        f.write(f"{wandb.run.url}\n")


if __name__ == "__main__":
    main()
