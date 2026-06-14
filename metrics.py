# utils/metrics.py
"""
BLEU and METEOR evaluation utilities for image captioning.

BLEU  — n-gram precision with brevity penalty (corpus-level)
METEOR— recall-weighted harmonic mean with stemming & synonymy
"""

import re
import math
from collections import Counter
from itertools import chain


# ══════════════════════════════════════════════════════════════════════════════
#  Tokenisation helper
# ══════════════════════════════════════════════════════════════════════════════

_PUNCT = re.compile(r"[^a-z0-9\s]")

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = _PUNCT.sub("", text)
    return text.strip().split()


# ══════════════════════════════════════════════════════════════════════════════
#  BLEU
# ══════════════════════════════════════════════════════════════════════════════

def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(hypothesis: list[str],
                        references: list[list[str]],
                        n: int) -> tuple[int, int]:
    """
    Compute clipped n-gram precision counts for one sentence.
    Returns (clipped_matches, hypothesis_ngram_count).
    """
    hyp_counts = _ngram_counts(hypothesis, n)
    if not hyp_counts:
        return 0, 0

    # Max reference count for each n-gram
    max_ref_counts: Counter = Counter()
    for ref in references:
        ref_counts = _ngram_counts(ref, n)
        for ngram, cnt in ref_counts.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], cnt)

    clipped = sum(min(cnt, max_ref_counts[ngram])
                  for ngram, cnt in hyp_counts.items())
    return clipped, sum(hyp_counts.values())


def sentence_bleu(hypothesis: str,
                   references: list[str],
                   max_n: int = 4,
                   smooth: bool = True) -> dict[str, float]:
    """
    Compute sentence-level BLEU-1 through BLEU-4 for one hypothesis.

    Parameters
    ----------
    hypothesis  : generated caption string
    references  : list of reference caption strings
    smooth      : apply +1 smoothing to avoid zero BLEU on short sentences
    """
    hyp_tokens  = _tokenize(hypothesis)
    ref_tokens  = [_tokenize(r) for r in references]

    scores: dict[str, float] = {}
    log_avg = 0.0
    valid_n = 0

    for n in range(1, max_n + 1):
        match, total = _clipped_precision(hyp_tokens, ref_tokens, n)
        if smooth:
            match += 1
            total += 1
        if total == 0:
            precision = 0.0
        else:
            precision = match / total

        if precision > 0:
            log_avg += math.log(precision)
            valid_n += 1

        scores[f"bleu{n}"] = precision

    # Brevity penalty
    ref_len = min(len(r) for r in ref_tokens)
    hyp_len = len(hyp_tokens)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    # Cumulative BLEU scores
    for n in range(1, max_n + 1):
        cum_log = sum(math.log(max(scores[f"bleu{k}"], 1e-10))
                      for k in range(1, n + 1)) / n
        scores[f"bleu{n}_cumulative"] = bp * math.exp(cum_log)

    return scores


def corpus_bleu(hypotheses: list[str],
                references_list: list[list[str]],
                max_n: int = 4) -> dict[str, float]:
    """
    Compute corpus-level BLEU (standard evaluation for captioning).

    Parameters
    ----------
    hypotheses       : list of generated captions (one per image)
    references_list  : list of reference-caption lists (one list per image)
    """
    clipped_matches = [0] * (max_n + 1)
    totals          = [0] * (max_n + 1)
    hyp_total_len   = 0
    ref_total_len   = 0

    for hyp, refs in zip(hypotheses, references_list):
        hyp_tok  = _tokenize(hyp)
        ref_toks = [_tokenize(r) for r in refs]

        hyp_total_len += len(hyp_tok)
        # Pick closest reference length
        ref_total_len += min(
            len(r) for r in ref_toks
        )

        for n in range(1, max_n + 1):
            m, t = _clipped_precision(hyp_tok, ref_toks, n)
            clipped_matches[n] += m
            totals[n] += t

    # Brevity penalty
    bp = (1.0 if hyp_total_len >= ref_total_len
          else math.exp(1 - ref_total_len / max(hyp_total_len, 1)))

    scores: dict[str, float] = {}
    for n in range(1, max_n + 1):
        p = clipped_matches[n] / max(totals[n], 1)
        scores[f"bleu{n}"] = p

    for n in range(1, max_n + 1):
        cum_log = sum(math.log(max(scores[f"bleu{k}"], 1e-10))
                      for k in range(1, n + 1)) / n
        scores[f"bleu{n}_cumulative"] = bp * math.exp(cum_log)

    scores["brevity_penalty"] = bp
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  METEOR  (simplified — no WordNet synonymy, uses Porter-like stemming)
# ══════════════════════════════════════════════════════════════════════════════

def _stem(word: str) -> str:
    """
    Minimal Porter-like stemming (handles most English suffixes).
    For full METEOR, use nltk.stem.PorterStemmer.
    """
    if len(word) <= 3:
        return word
    for suffix in ("ational", "tional", "enci", "anci", "izer", "ising",
                   "izing", "ated", "ating", "alism", "ness", "ment",
                   "tion", "ous", "ive", "ful", "ing", "ies", "ers",
                   "est", "er", "es", "ly", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word


def sentence_meteor(hypothesis: str,
                     references: list[str],
                     alpha: float = 0.9,
                     beta:  float = 3.0,
                     gamma: float = 0.5) -> float:
    """
    Sentence-level METEOR score.

    Uses unigram exact match + stemmed match (no synonym match for simplicity).
    Standard weights: alpha=0.9, beta=3, gamma=0.5
    """
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0

    best = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue

        # Exact matches
        hyp_set = Counter(hyp_tokens)
        ref_set = Counter(ref_tokens)
        exact   = sum((hyp_set & ref_set).values())

        # Stem matches on unmatched tokens
        hyp_unmatched = list(hyp_tokens)
        ref_unmatched = list(ref_tokens)
        for w in list(hyp_set & ref_set):
            count = min(hyp_set[w], ref_set[w])
            for _ in range(count):
                if w in hyp_unmatched: hyp_unmatched.remove(w)
                if w in ref_unmatched: ref_unmatched.remove(w)

        hyp_stems = Counter(_stem(w) for w in hyp_unmatched)
        ref_stems = Counter(_stem(w) for w in ref_unmatched)
        stem_match = sum((hyp_stems & ref_stems).values())

        m = exact + stem_match  # total matched unigrams

        precision = m / len(hyp_tokens)
        recall    = m / len(ref_tokens)

        if precision + recall == 0:
            continue

        f_mean = precision * recall / (alpha * precision + (1 - alpha) * recall)

        # Chunk penalty
        chunks = _count_chunks(hyp_tokens, ref_tokens)
        penalty = gamma * (chunks / max(m, 1)) ** beta

        score = f_mean * (1 - penalty)
        best  = max(best, score)

    return best


def _count_chunks(hyp: list[str], ref: list[str]) -> int:
    """Count the number of contiguous matched chunks (for fragmentation penalty)."""
    ref_pos: dict[str, list[int]] = {}
    for i, w in enumerate(ref):
        ref_pos.setdefault(w, []).append(i)

    matched_ref_indices: list[int] = []
    for word in hyp:
        if word in ref_pos and ref_pos[word]:
            matched_ref_indices.append(ref_pos[word].pop(0))

    if not matched_ref_indices:
        return 0

    chunks = 1
    for i in range(1, len(matched_ref_indices)):
        if matched_ref_indices[i] != matched_ref_indices[i - 1] + 1:
            chunks += 1
    return chunks


def corpus_meteor(hypotheses: list[str],
                  references_list: list[list[str]]) -> float:
    """Average sentence METEOR across the corpus."""
    if not hypotheses:
        return 0.0
    return sum(
        sentence_meteor(h, refs)
        for h, refs in zip(hypotheses, references_list)
    ) / len(hypotheses)


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty-print summary
# ══════════════════════════════════════════════════════════════════════════════

def print_scores(bleu: dict[str, float], meteor: float) -> None:
    print("\n" + "═" * 40)
    print("  Evaluation Results")
    print("═" * 40)
    print(f"  BLEU-1  : {bleu['bleu1_cumulative']:.4f}")
    print(f"  BLEU-2  : {bleu['bleu2_cumulative']:.4f}")
    print(f"  BLEU-3  : {bleu['bleu3_cumulative']:.4f}")
    print(f"  BLEU-4  : {bleu['bleu4_cumulative']:.4f}")
    print(f"  METEOR  : {meteor:.4f}")
    print("═" * 40 + "\n")
