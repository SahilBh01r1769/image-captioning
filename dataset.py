# utils/dataset.py
"""
Flickr8k Dataset loader.

Expected layout:
  data/Flickr8k/Images/      ← all 8,091 .jpg images
  data/Flickr8k/captions.txt ← one line per (image, caption) pair

captions.txt format (Kaggle version):
  image,caption
  1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
  ...
"""

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.vocabulary import Vocabulary


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transform(split: str = "train") -> T.Compose:
    """
    Returns image transforms.
    Training: random crop, flip, colour jitter for augmentation.
    Val/Test: deterministic centre crop only.
    """
    if split == "train":
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(config.IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(config.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ])


# ── Caption parsing ───────────────────────────────────────────────────────────

def parse_captions(captions_file: str = config.CAPTIONS_FILE
                   ) -> dict[str, list[str]]:
    """
    Parse the Flickr8k captions file.

    Returns
    -------
    dict mapping image filename → list of 5 caption strings
    """
    image_captions: dict[str, list[str]] = defaultdict(list)

    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    # Skip header if present
    start = 1 if lines[0].lower().startswith("image") else 0

    for line in lines[start:]:
        if not line.strip():
            continue
        # Split only on first comma — caption may contain commas
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        img_name, caption = parts
        img_name = img_name.strip()
        caption  = caption.strip()
        # Some Kaggle versions append #0–#4 to the image name
        img_name = img_name.split("#")[0].strip()
        image_captions[img_name].append(caption)

    return dict(image_captions)


def split_dataset(image_captions: dict[str, list[str]],
                  train: float = config.TRAIN_SPLIT,
                  val:   float = config.VAL_SPLIT,
                  seed:  int   = 42
                  ) -> tuple[list[str], list[str], list[str]]:
    """Split image keys into train / val / test lists."""
    keys = sorted(image_captions.keys())
    random.seed(seed)
    random.shuffle(keys)

    n       = len(keys)
    n_train = int(n * train)
    n_val   = int(n * val)

    train_keys = keys[:n_train]
    val_keys   = keys[n_train : n_train + n_val]
    test_keys  = keys[n_train + n_val :]

    print(f"Dataset split → train: {len(train_keys)} | "
          f"val: {len(val_keys)} | test: {len(test_keys)} images")
    return train_keys, val_keys, test_keys


# ── Dataset ───────────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k.

    Each item is (image_tensor, caption_tensor, caption_length).
    During training we flatten so each (image, caption) pair is one sample.
    During evaluation we keep all 5 captions per image for BLEU scoring.
    """

    def __init__(
        self,
        image_keys:      list[str],
        image_captions:  dict[str, list[str]],
        vocabulary:      Vocabulary,
        images_dir:      str   = config.IMAGES_DIR,
        split:           str   = "train",
        max_length:      int   = config.MAX_CAPTION_LENGTH,
    ):
        self.images_dir      = images_dir
        self.vocabulary      = vocabulary
        self.transform       = get_transform(split)
        self.max_length      = max_length
        self.split           = split

        # Build flat list of (image_name, caption) pairs
        self.samples: list[tuple[str, str]] = []
        # Also store all captions per image for evaluation
        self.image_to_captions: dict[str, list[str]] = {}

        for img_name in image_keys:
            captions = image_captions.get(img_name, [])
            self.image_to_captions[img_name] = captions
            for cap in captions:
                self.samples.append((img_name, cap))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load & transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Encode caption
        encoded = self.vocabulary.encode_caption(caption, self.max_length)
        caption_tensor = torch.tensor(encoded, dtype=torch.long)

        # Caption length (including START and END, excluding PAD)
        length = min(
            len([w for w in encoded if w != self.vocabulary[config.PAD_TOKEN]]),
            self.max_length
        )

        return image, caption_tensor, torch.tensor(length, dtype=torch.long)


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(
    vocabulary: Vocabulary,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Parse captions, split dataset, return train/val/test DataLoaders
    and a dict mapping image_name → list of reference captions (for BLEU).
    """
    image_captions = parse_captions()
    train_keys, val_keys, test_keys = split_dataset(image_captions)

    train_ds = Flickr8kDataset(train_keys, image_captions, vocabulary, split="train")
    val_ds   = Flickr8kDataset(val_keys,   image_captions, vocabulary, split="val")
    test_ds  = Flickr8kDataset(test_keys,  image_captions, vocabulary, split="test")

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    # Reference captions for BLEU evaluation
    test_references = test_ds.image_to_captions

    print(f"Train samples : {len(train_ds):,}")
    print(f"Val   samples : {len(val_ds):,}")
    print(f"Test  samples : {len(test_ds):,}")

    return train_loader, val_loader, test_loader, test_references


# ── GloVe download helper ─────────────────────────────────────────────────────

def download_glove():
    """Convenience helper — prints instructions to download GloVe."""
    print("GloVe download instructions:")
    print("  1. Visit https://nlp.stanford.edu/data/glove.6B.zip")
    print("  2. Download and extract glove.6B.200d.txt")
    print(f"  3. Place it at: {config.GLOVE_FILE}")
    print("\nOr run in terminal:")
    print("  wget https://nlp.stanford.edu/data/glove.6B.zip")
    print("  unzip glove.6B.zip glove.6B.200d.txt -d data/glove/")
