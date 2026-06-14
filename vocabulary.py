# utils/vocabulary.py
"""
Vocabulary builder for image captioning.
Handles word-to-index mapping, GloVe embedding loading,
and special token management.
"""

import os
import re
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Vocabulary:
    """
    Manages the word vocabulary used for tokenizing captions.

    Special tokens:
      <PAD>   → index 0  (padding)
      <START> → index 1  (beginning of caption)
      <END>   → index 2  (end of caption)
      <UNK>   → index 3  (unknown / rare word)
    """

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_freq: Counter       = Counter()
        self._idx                     = 0

        # Add special tokens first
        for tok in [config.PAD_TOKEN, config.START_TOKEN,
                    config.END_TOKEN, config.UNK_TOKEN]:
            self._add_word(tok)

    # ── internal ──────────────────────────────────────────────────────────
    def _add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = self._idx
            self.idx2word[self._idx] = word
            self._idx += 1
        return self.word2idx[word]

    # ── public ────────────────────────────────────────────────────────────
    def build_from_captions(self, captions: list[str],
                             min_freq: int = config.MIN_WORD_FREQ) -> None:
        """Build vocab from a flat list of raw caption strings."""
        print("Building vocabulary …")
        for caption in captions:
            for word in tokenize(caption):
                self.word_freq[word] += 1

        added = 0
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                self._add_word(word)
                added += 1

        print(f"  Total unique words : {len(self.word_freq):,}")
        print(f"  Vocab size (≥{min_freq}): {added:,}  (+4 special tokens)")
        print(f"  Final vocab size   : {len(self):,}")

    def __len__(self) -> int:
        return len(self.word2idx)

    def __getitem__(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[config.UNK_TOKEN])

    def decode(self, indices: list[int]) -> str:
        """Convert a list of indices back to a caption string."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, config.UNK_TOKEN)
            if word == config.END_TOKEN:
                break
            if word not in (config.PAD_TOKEN, config.START_TOKEN):
                words.append(word)
        return " ".join(words)

    # ── caption encoding ──────────────────────────────────────────────────
    def encode_caption(self, caption: str,
                        max_length: int = config.MAX_CAPTION_LENGTH
                        ) -> list[int]:
        """
        Tokenize + encode a caption into a fixed-length integer sequence.
        Format: [<START>, w1, w2, …, <END>, <PAD>, <PAD>, …]
        """
        tokens = tokenize(caption)
        # Truncate body to leave room for START and END
        tokens = tokens[: max_length - 2]
        ids = (
            [self.word2idx[config.START_TOKEN]]
            + [self[w] for w in tokens]
            + [self.word2idx[config.END_TOKEN]]
        )
        # Pad
        ids += [self.word2idx[config.PAD_TOKEN]] * (max_length - len(ids))
        return ids

    # ── GloVe ─────────────────────────────────────────────────────────────
    def build_glove_matrix(self, glove_path: str = config.GLOVE_FILE,
                            glove_dim: int = config.GLOVE_DIM) -> np.ndarray:
        """
        Load GloVe vectors and return an embedding matrix aligned to this vocab.
        Words not found in GloVe are randomly initialised (unit normal, scaled).

        Returns
        -------
        np.ndarray of shape (vocab_size, glove_dim)
        """
        if not os.path.exists(glove_path):
            raise FileNotFoundError(
                f"GloVe file not found at {glove_path}.\n"
                "Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip "
                "and extract glove.6B.200d.txt into data/glove/"
            )

        print(f"Loading GloVe from {glove_path} …")
        glove: dict[str, np.ndarray] = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="  Reading GloVe"):
                parts = line.strip().split()
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec

        vocab_size  = len(self)
        matrix      = np.zeros((vocab_size, glove_dim), dtype=np.float32)
        found        = 0
        scale        = 0.6 / np.sqrt(glove_dim)  # Xavier-like scale for OOV

        for word, idx in self.word2idx.items():
            if word in glove:
                matrix[idx] = glove[word]
                found += 1
            else:
                matrix[idx] = np.random.randn(glove_dim).astype(np.float32) * scale

        # Zero-out <PAD>
        matrix[self.word2idx[config.PAD_TOKEN]] = 0.0

        print(f"  GloVe coverage: {found}/{vocab_size} words "
              f"({100*found/vocab_size:.1f}%)")
        return matrix

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path: str = config.VOCAB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Vocabulary saved → {path}")

    @staticmethod
    def load(path: str = config.VOCAB_PATH) -> "Vocabulary":
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded ← {path}  (size={len(vocab):,})")
        return vocab


# ── helpers ───────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^a-z0-9\s]")

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = _PUNCT.sub("", text)
    return text.strip().split()
