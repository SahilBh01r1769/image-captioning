"""Flickr8k dataset loading and image-level train/validation/test splitting."""
from __future__ import annotations

import os
import random
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import config
from vocabulary import Vocabulary


def get_transform(split: str = "train") -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(config.IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
    ])


def parse_captions(captions_file: str = config.CAPTIONS_FILE) -> dict[str, list[str]]:
    """Parse the Kaggle/UCI-style Flickr8k ``image,caption`` file."""
    if not os.path.exists(captions_file):
        raise FileNotFoundError(
            f"Flickr8k captions file not found: {captions_file}. "
            "See README.md for dataset setup."
        )

    image_captions: dict[str, list[str]] = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as handle:
        lines = handle.read().strip().split("\n")
    if not lines:
        return {}

    start = 1 if lines[0].lower().startswith("image") else 0
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        image_name, caption = parts
        image_name = image_name.split("#")[0].strip()
        caption = caption.strip()
        if image_name and caption:
            image_captions[image_name].append(caption)
    return dict(image_captions)


def split_dataset(
    image_captions: dict[str, list[str]],
    train: float = config.TRAIN_SPLIT,
    val: float = config.VAL_SPLIT,
    seed: int = config.SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Split by image, never by caption, preventing cross-split image leakage."""
    if train <= 0 or val < 0 or train + val >= 1:
        raise ValueError("train and val fractions must leave a non-empty test split")

    keys = sorted(image_captions)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * train)
    n_val = int(n * val)
    return (
        keys[:n_train],
        keys[n_train:n_train + n_val],
        keys[n_train + n_val:],
    )


class Flickr8kDataset(Dataset):
    """Flatten image-caption pairs while retaining image-level references."""

    def __init__(
        self,
        image_keys: list[str],
        image_captions: dict[str, list[str]],
        vocabulary: Vocabulary,
        images_dir: str = config.IMAGES_DIR,
        split: str = "train",
        max_length: int = config.MAX_CAPTION_LENGTH,
    ):
        self.images_dir = images_dir
        self.vocabulary = vocabulary
        self.transform = get_transform(split)
        self.max_length = max_length
        self.split = split
        self.samples: list[tuple[str, str]] = []
        self.image_to_captions: dict[str, list[str]] = {}

        for image_name in image_keys:
            captions = image_captions.get(image_name, [])
            self.image_to_captions[image_name] = captions
            self.samples.extend((image_name, caption) for caption in captions)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_name, caption = self.samples[idx]
        image_path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = self.transform(Image.open(image_path).convert("RGB"))
        encoded = self.vocabulary.encode_caption(caption, self.max_length)
        caption_tensor = torch.tensor(encoded, dtype=torch.long)
        pad_idx = self.vocabulary[config.PAD_TOKEN]
        length = sum(token != pad_idx for token in encoded)
        return image, caption_tensor, torch.tensor(length, dtype=torch.long)


def build_dataloaders(
    vocabulary: Vocabulary,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    image_captions = parse_captions()
    train_keys, val_keys, test_keys = split_dataset(image_captions)
    train_ds = Flickr8kDataset(train_keys, image_captions, vocabulary, split="train")
    val_ds = Flickr8kDataset(val_keys, image_captions, vocabulary, split="val")
    test_ds = Flickr8kDataset(test_keys, image_captions, vocabulary, split="test")
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, test_ds.image_to_captions


def download_glove() -> None:
    """Print the authoritative GloVe setup location used by this project."""
    print("Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip")
    print(f"Extract glove.6B.200d.txt to: {config.GLOVE_FILE}")
