"""Image transforms / augmentations."""
from __future__ import annotations

from PIL import Image
import torch
import torchvision.transforms as T


# DINOv2 normalization statistics (ImageNet)
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD  = (0.229, 0.224, 0.225)


def build_train_transform(cfg) -> T.Compose:
    """Training-time transform with augmentation."""
    aug = cfg.augmentation
    img_size = cfg.data.image_size

    ops = [
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
    ]
    if aug.enabled:
        ops.append(T.RandomHorizontalFlip(p=aug.horizontal_flip_prob))
        ops.append(T.RandAugment(num_ops=aug.randaugment_n, magnitude=aug.randaugment_m))
        cj = aug.color_jitter
        ops.append(T.ColorJitter(
            brightness=cj.brightness, contrast=cj.contrast, saturation=cj.saturation
        ))
    ops += [
        T.ToTensor(),
        T.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ]
    return T.Compose(ops)


def build_eval_transform(cfg) -> T.Compose:
    """Eval/inference transform — no augmentation."""
    img_size = cfg.data.image_size
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


def build_inference_transform(image_size: int = 224) -> T.Compose:
    """Stand-alone inference transform for the web app."""
    return T.Compose([
        T.Resize(int(image_size * 1.15)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])
