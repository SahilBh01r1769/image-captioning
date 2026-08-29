from dataset import split_dataset
from metrics import caption_statistics, corpus_bleu, corpus_meteor_lite
from vocabulary import Vocabulary


def test_image_level_split_is_disjoint_and_deterministic():
    captions = {f"img_{idx}.jpg": [f"caption {idx}"] * 5 for idx in range(100)}
    first = split_dataset(captions, seed=42)
    second = split_dataset(captions, seed=42)
    assert first == second
    train, val, test = map(set, first)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(captions)


def test_vocabulary_round_trip():
    vocab = Vocabulary()
    vocab.build_from_captions(["A dog runs fast", "A dog sleeps"], min_freq=1)
    encoded = vocab.encode_caption("A dog runs fast", max_length=8)
    decoded = vocab.decode(encoded)
    assert decoded == "a dog runs fast"
    assert encoded[0] == vocab["<START>"]


def test_bleu_is_high_for_exact_match_and_lower_for_unrelated_text():
    references = [["a dog runs through grass", "a dog is running"]]
    exact = corpus_bleu(["a dog runs through grass"], references)
    unrelated = corpus_bleu(["two people sit indoors"], references)
    assert exact["bleu1_cumulative"] > unrelated["bleu1_cumulative"]
    assert exact["bleu4_cumulative"] > unrelated["bleu4_cumulative"]


def test_meteor_lite_and_diversity_are_bounded():
    hypotheses = ["a dog runs", "a cat sleeps"]
    references = [["a dog is running"], ["a cat is sleeping"]]
    meteor = corpus_meteor_lite(hypotheses, references)
    stats = caption_statistics(hypotheses)
    assert 0 <= meteor <= 1
    assert 0 <= stats["distinct_1"] <= 1
    assert 0 <= stats["distinct_2"] <= 1
