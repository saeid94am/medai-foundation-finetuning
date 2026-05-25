import torch
import pytest

from medai_medsam.metrics.segmentation import SegmentationMetrics


def test_metrics_perfect_dice():
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.long)
    mask[0, 0, 16:48, 16:48] = 1
    metrics = SegmentationMetrics()
    metrics.update(mask, mask)
    results = metrics.compute()
    assert abs(results["dice"] - 1.0) < 1e-4, f"Expected dice=1.0, got {results['dice']:.4f}"


def test_metrics_zero_prediction_dice_is_zero():
    target = torch.zeros(1, 1, 64, 64, dtype=torch.long)
    target[0, 0, 16:48, 16:48] = 1
    pred = torch.zeros_like(target)
    metrics = SegmentationMetrics()
    metrics.update(pred, target)
    results = metrics.compute()
    assert results["dice"] < 0.1


def test_metrics_reset_clears_state():
    mask = torch.zeros(1, 1, 32, 32, dtype=torch.long)
    mask[0, 0, 8:24, 8:24] = 1
    metrics = SegmentationMetrics()
    metrics.update(mask, mask)
    metrics.reset()
    metrics.update(mask, mask)
    results = metrics.compute()
    # After reset and one update, dice should still be 1.0
    assert abs(results["dice"] - 1.0) < 1e-4


def test_metrics_iou_perfect():
    mask = torch.zeros(1, 1, 32, 32, dtype=torch.long)
    mask[0, 0, 4:28, 4:28] = 1
    metrics = SegmentationMetrics()
    metrics.update(mask, mask)
    results = metrics.compute()
    assert abs(results["iou"] - 1.0) < 1e-4
