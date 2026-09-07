"""Extract and load one shared frozen-ResNet feature cache for all ablations."""
from __future__ import annotations

import os
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from data_protocol import dataset_fingerprint
from dataset import get_transform, parse_captions, resolve_dataset_split
from visual_features import SpatialResNetBackbone
from vocabulary import Vocabulary


FEATURE_CACHE_VERSION = 1
FEATURE_DTYPE = torch.float16


class _ImageDataset(Dataset):
    def __init__(self, image_keys: list[str], images_dir: str):
        self.image_keys = image_keys
        self.images_dir = images_dir
        # Feature extraction is deterministic. Caption training augmentation is
        # deliberately not baked into a cache shared by all ablations.
        self.transform = get_transform("validation")

    def __len__(self) -> int:
        return len(self.image_keys)

    def __getitem__(self, index: int):
        image_name = self.image_keys[index]
        path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        image = self.transform(Image.open(path).convert("RGB"))
        return image_name, image


def expected_cache_metadata(
    image_captions: dict[str, list[str]], image_keys: list[str]
) -> dict[str, Any]:
    return {
        "cache_version": FEATURE_CACHE_VERSION,
        "dataset_fingerprint": dataset_fingerprint(image_captions),
        "image_keys": list(image_keys),
        "backbone": "torchvision.resnet50",
        "weights": "IMAGENET1K_V1",
        "image_size": config.IMAGE_SIZE,
        "image_mean": list(config.IMAGE_MEAN),
        "image_std": list(config.IMAGE_STD),
        "feature_layout": "locations_channels",
        "feature_dtype": "float16",
    }


@torch.no_grad()
def extract_feature_cache(
    image_captions: dict[str, list[str]],
    output_path: str = config.FEATURE_CACHE_PATH,
    *,
    images_dir: str = config.IMAGES_DIR,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device = config.DEVICE,
) -> dict[str, Any]:
    """Create a deterministic float16 spatial-feature tensor for every image."""
    image_keys = sorted(image_captions)
    if not image_keys:
        raise ValueError("cannot extract features for an empty dataset")
    loader = DataLoader(
        _ImageDataset(image_keys, images_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    encoder = SpatialResNetBackbone(fine_tune=False, pretrained=True).to(device).eval()
    cached: torch.Tensor | None = None
    offset = 0

    for names, images in tqdm(loader, desc="Extracting frozen ResNet features", unit="batch"):
        spatial = encoder(images.to(device, non_blocking=True)).cpu().to(FEATURE_DTYPE)
        if cached is None:
            cached = torch.empty(
                len(image_keys), spatial.size(1), spatial.size(2), dtype=FEATURE_DTYPE
            )
        end = offset + spatial.size(0)
        cached[offset:end].copy_(spatial)
        if list(names) != image_keys[offset:end]:
            raise RuntimeError("feature extraction order no longer matches the image index")
        offset = end

    if cached is None or offset != len(image_keys):
        raise RuntimeError("feature extraction did not cover every image")
    metadata = expected_cache_metadata(image_captions, image_keys)
    payload = {"metadata": metadata, "features": cached}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)
    return metadata


def load_feature_cache(
    path: str,
    image_captions: dict[str, list[str]],
) -> tuple[torch.Tensor, dict[str, int], dict[str, Any]]:
    """Load a cache only when its dataset and extraction settings still match."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frozen feature cache not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # PyTorch versions before mmap/weights_only support.
        payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    image_keys = sorted(image_captions)
    if metadata != expected_cache_metadata(image_captions, image_keys):
        raise RuntimeError("Feature cache does not match the current dataset or encoder protocol")
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise RuntimeError("Feature cache tensor must have shape (images, locations, channels)")
    if features.size(0) != len(image_keys) or features.size(2) != config.CNN_FEAT_DIM:
        raise RuntimeError("Feature cache tensor dimensions do not match its metadata")
    return features, {name: index for index, name in enumerate(image_keys)}, metadata


class CachedFeatureCaptionDataset(Dataset):
    """Caption pairs backed by the shared memory-mappable feature tensor."""

    def __init__(
        self,
        image_keys: list[str],
        image_captions: dict[str, list[str]],
        vocabulary: Vocabulary,
        cache_path: str = config.FEATURE_CACHE_PATH,
        max_length: int = config.MAX_CAPTION_LENGTH,
    ):
        self.features, self.image_to_index, self.cache_metadata = load_feature_cache(
            cache_path, image_captions
        )
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.samples = [
            (image_name, caption)
            for image_name in image_keys
            for caption in image_captions[image_name]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_name, caption = self.samples[index]
        feature = self.features[self.image_to_index[image_name]].float()
        encoded = self.vocabulary.encode_caption(caption, self.max_length)
        caption_tensor = torch.tensor(encoded, dtype=torch.long)
        pad_idx = self.vocabulary[config.PAD_TOKEN]
        length = sum(token != pad_idx for token in encoded)
        return feature, caption_tensor, torch.tensor(length, dtype=torch.long)


def build_cached_feature_dataloaders(
    vocabulary: Vocabulary,
    cache_path: str = config.FEATURE_CACHE_PATH,
    *,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    """Build all loaders from one shared cache and the frozen image split."""
    image_captions = parse_captions()
    manifest = resolve_dataset_split(image_captions)
    datasets = {
        name: CachedFeatureCaptionDataset(
            manifest["splits"][name], image_captions, vocabulary, cache_path
        )
        for name in ("train", "validation", "test")
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_references = {
        key: image_captions[key] for key in manifest["splits"]["test"]
    }
    return train_loader, validation_loader, test_loader, test_references
