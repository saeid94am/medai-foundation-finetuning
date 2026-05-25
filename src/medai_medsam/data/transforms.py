def get_train_transforms(cfg):
    """Albumentations pipeline for the training split."""
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.horizontal_flip_prob),
            A.VerticalFlip(p=cfg.vertical_flip_prob),
            A.Rotate(limit=cfg.rotate_limit, p=0.5),
            A.RandomBrightnessContrast(p=cfg.brightness_contrast_prob),
            A.GaussNoise(p=cfg.gaussian_noise_prob),
        ]
    )


def get_val_transforms():
    """No augmentation for val/test — only the resize in BUSIDataset.__getitem__."""
    return None
