"""Dependency-light captioning metrics.

BLEU is implemented at corpus level with clipped n-gram precision and the
standard brevity penalty. ``meteor_lite`` is intentionally named as an
approximation: it uses exact/stem matching but not WordNet synonym matching.
"""
from __future__ import annotations

import math
import re
from collections import Counter


_PUNCT = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> list[str]:
    return _PUNCT.sub("", text.lower()).strip().split()


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def _clipped_precision(hypothesis: list[str], references: list[list[str]], n: int) -> tuple[int, int]:
    hyp_counts = _ngram_counts(hypothesis, n)
    if not hyp_counts:
        return 0, 0
    max_ref_counts: Counter = Counter()
    for reference in references:
        counts = _ngram_counts(reference, n)
        for ngram, count in counts.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
    clipped = sum(min(count, max_ref_counts[ngram]) for ngram, count in hyp_counts.items())
    return clipped, sum(hyp_counts.values())


def _closest_reference_length(hyp_len: int, references: list[list[str]]) -> int:
    lengths = [len(reference) for reference in references]
    if not lengths:
        return 0
    return min(lengths, key=lambda length: (abs(length - hyp_len), length))


def corpus_bleu(
    hypotheses: list[str],
    references_list: list[list[str]],
    max_n: int = 4,
) -> dict[str, float]:
    if len(hypotheses) != len(references_list):
        raise ValueError("hypotheses and references_list must have equal length")
    if not hypotheses:
        return {**{f"bleu{n}_cumulative": 0.0 for n in range(1, max_n + 1)}, "brevity_penalty": 0.0}

    matches = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hyp_total = 0
    ref_total = 0

    for hypothesis, references in zip(hypotheses, references_list):
        hyp_tokens = tokenize(hypothesis)
        ref_tokens = [tokenize(reference) for reference in references]
        hyp_total += len(hyp_tokens)
        ref_total += _closest_reference_length(len(hyp_tokens), ref_tokens)
        for n in range(1, max_n + 1):
            clipped, total = _clipped_precision(hyp_tokens, ref_tokens, n)
            matches[n] += clipped
            totals[n] += total

    if hyp_total == 0:
        brevity_penalty = 0.0
    elif hyp_total >= ref_total:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - ref_total / hyp_total)

    precisions = {
        n: matches[n] / totals[n] if totals[n] else 0.0
        for n in range(1, max_n + 1)
    }
    scores: dict[str, float] = {"brevity_penalty": brevity_penalty}
    for n in range(1, max_n + 1):
        selected = [precisions[k] for k in range(1, n + 1)]
        if any(value <= 0 for value in selected):
            cumulative = 0.0
        else:
            cumulative = brevity_penalty * math.exp(sum(math.log(value) for value in selected) / n)
        scores[f"bleu{n}"] = precisions[n]
        scores[f"bleu{n}_cumulative"] = cumulative
    return scores


def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix in (
        "ational", "tional", "enci", "anci", "izer", "ising", "izing",
        "ated", "ating", "alism", "ness", "ment", "tion", "ous", "ive",
        "ful", "ing", "ies", "ers", "est", "er", "es", "ly", "ed", "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word


def _greedy_matches(hypothesis: list[str], reference: list[str]) -> list[tuple[int, int]]:
    """Match exact words first, then unmatched stem-equivalent words."""
    used_ref: set[int] = set()
    matches: list[tuple[int, int]] = []
    unmatched_hyp: list[int] = []

    for h_idx, word in enumerate(hypothesis):
        found = next((r_idx for r_idx, ref_word in enumerate(reference) if r_idx not in used_ref and ref_word == word), None)
        if found is None:
            unmatched_hyp.append(h_idx)
        else:
            used_ref.add(found)
            matches.append((h_idx, found))

    for h_idx in unmatched_hyp:
        stem = _stem(hypothesis[h_idx])
        found = next(
            (r_idx for r_idx, ref_word in enumerate(reference) if r_idx not in used_ref and _stem(ref_word) == stem),
            None,
        )
        if found is not None:
            used_ref.add(found)
            matches.append((h_idx, found))
    return sorted(matches)


def _chunks(matches: list[tuple[int, int]]) -> int:
    if not matches:
        return 0
    count = 1
    for previous, current in zip(matches, matches[1:]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            count += 1
    return count


def sentence_meteor_lite(hypothesis: str, references: list[str]) -> float:
    """Approximate METEOR using exact/stem matches and fragmentation penalty."""
    hyp = tokenize(hypothesis)
    if not hyp:
        return 0.0
    best = 0.0
    for reference_text in references:
        ref = tokenize(reference_text)
        if not ref:
            continue
        matches = _greedy_matches(hyp, ref)
        matched = len(matches)
        if matched == 0:
            continue
        precision = matched / len(hyp)
        recall = matched / len(ref)
        f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)
        penalty = 0.5 * (_chunks(matches) / matched) ** 3
        best = max(best, f_mean * (1.0 - penalty))
    return best


def corpus_meteor_lite(hypotheses: list[str], references_list: list[list[str]]) -> float:
    if not hypotheses:
        return 0.0
    return sum(
        sentence_meteor_lite(hypothesis, references)
        for hypothesis, references in zip(hypotheses, references_list)
    ) / len(hypotheses)


def distinct_n(hypotheses: list[str], n: int = 1) -> float:
    """Ratio of unique generated n-grams; a simple caption-diversity signal."""
    all_ngrams: list[tuple[str, ...]] = []
    for caption in hypotheses:
        tokens = tokenize(caption)
        all_ngrams.extend(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def caption_statistics(hypotheses: list[str]) -> dict[str, float]:
    lengths = [len(tokenize(caption)) for caption in hypotheses]
    return {
        "mean_caption_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "distinct_1": distinct_n(hypotheses, 1),
        "distinct_2": distinct_n(hypotheses, 2),
    }
