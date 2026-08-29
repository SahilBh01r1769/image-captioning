"""Vocabulary and GloVe utilities for Flickr8k captioning."""
from __future__ import annotations

import os
import pickle
import re
from collections import Counter

import numpy as np
from tqdm import tqdm

import config


class Vocabulary:
    """Word/index mapping with explicit special-token handling."""

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self._idx = 0
        for token in [config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN, config.UNK_TOKEN]:
            self._add_word(token)

    def _add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = self._idx
            self.idx2word[self._idx] = word
            self._idx += 1
        return self.word2idx[word]

    def build_from_captions(self, captions: list[str], min_freq: int = config.MIN_WORD_FREQ) -> None:
        self.word_freq.clear()
        for caption in captions:
            self.word_freq.update(tokenize(caption))
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                self._add_word(word)

    def __len__(self) -> int:
        return len(self.word2idx)

    def __getitem__(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])

    def decode(self, indices: list[int]) -> str:
        words: list[str] = []
        for idx in indices:
            word = self.idx2word.get(int(idx), config.UNK_TOKEN)
            if word == config.END_TOKEN:
                break
            if word not in (config.PAD_TOKEN, config.START_TOKEN):
                words.append(word)
        return " ".join(words)

    def encode_caption(self, caption: str, max_length: int = config.MAX_CAPTION_LENGTH) -> list[int]:
        tokens = tokenize(caption)[: max_length - 2]
        ids = [self.word2idx[config.START_TOKEN]]
        ids.extend(self[word] for word in tokens)
        ids.append(self.word2idx[config.END_TOKEN])
        ids.extend([self.word2idx[config.PAD_TOKEN]] * (max_length - len(ids)))
        return ids

    def build_glove_matrix(
        self,
        glove_path: str = config.GLOVE_FILE,
        glove_dim: int = config.GLOVE_DIM,
    ) -> np.ndarray:
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")

        needed = set(self.word2idx)
        vectors: dict[str, np.ndarray] = {}
        with open(glove_path, "r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc="Reading GloVe"):
                parts = line.rstrip().split()
                if len(parts) != glove_dim + 1 or parts[0] not in needed:
                    continue
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

        rng = np.random.default_rng(config.SEED)
        scale = 0.6 / np.sqrt(glove_dim)
        matrix = rng.normal(0, scale, size=(len(self), glove_dim)).astype(np.float32)
        for word, idx in self.word2idx.items():
            if word in vectors:
                matrix[idx] = vectors[word]
        matrix[self.word2idx[config.PAD_TOKEN]] = 0.0
        return matrix

    def save(self, path: str = config.VOCAB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str = config.VOCAB_PATH) -> "Vocabulary":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary not found: {path}. Train/build the vocabulary first.")
        with open(path, "rb") as handle:
            return pickle.load(handle)


_PUNCT = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> list[str]:
    """Lower-case, remove punctuation and split on whitespace."""
    return _PUNCT.sub("", text.lower()).strip().split()
