import torch
import torch.nn as nn

import config
from attention_model import ExplainableCaptioningModel, attention_coverage_loss
from feature_cache import (
    CachedFeatureCaptionDataset,
    expected_cache_metadata,
    load_feature_cache,
)
from model import ImageCaptioningModel
from visual_features import SpatialResNetBackbone, global_average_pool
from vocabulary import Vocabulary


def test_global_baseline_pools_the_shared_spatial_tensor():
    spatial = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    pooled = global_average_pool(spatial)
    assert torch.equal(pooled, spatial.mean(dim=1))


def test_baseline_and_attention_accept_the_same_cached_features():
    spatial = torch.randn(2, 4, config.CNN_FEAT_DIM)
    captions = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0]])
    baseline = ImageCaptioningModel(
        vocab_size=12,
        embed_dim=8,
        hidden_dim=10,
        num_layers=1,
        dropout=0.0,
        pretrained_encoder=False,
    )
    attention = ExplainableCaptioningModel(
        vocab_size=12,
        encoder_dim=8,
        embed_dim=8,
        hidden_dim=10,
        attention_dim=6,
        dropout=0.0,
        pretrained_encoder=False,
    )

    baseline_logits = baseline.forward_from_features(spatial, captions)
    attention_logits, alphas = attention.forward_from_features(spatial, captions)

    assert baseline.decoder.num_layers == 1
    assert baseline_logits.shape == attention_logits.shape == (2, 4, 12)
    assert alphas.shape == (2, 4, 4)


def test_frozen_backbone_keeps_batch_norm_statistics_frozen():
    encoder = SpatialResNetBackbone(fine_tune=False, pretrained=False)
    encoder.train()
    batch_norms = [module for module in encoder.backbone.modules() if isinstance(module, nn.BatchNorm2d)]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())


def test_cache_is_bound_to_dataset_and_loads_caption_pairs(tmp_path):
    image_captions = {
        "a.jpg": ["a small dog", "the dog runs"],
        "b.jpg": ["a blue bicycle"],
    }
    image_keys = sorted(image_captions)
    features = torch.randn(2, 4, config.CNN_FEAT_DIM).half()
    path = tmp_path / "features.pt"
    torch.save(
        {
            "metadata": expected_cache_metadata(image_captions, image_keys),
            "features": features,
        },
        path,
    )
    loaded, index, _ = load_feature_cache(str(path), image_captions)
    assert torch.equal(loaded, features)
    assert index == {"a.jpg": 0, "b.jpg": 1}

    vocab = Vocabulary()
    vocab.build_from_captions([caption for captions in image_captions.values() for caption in captions], min_freq=1)
    dataset = CachedFeatureCaptionDataset(
        ["a.jpg"], image_captions, vocab, str(path), max_length=8
    )
    feature, caption, length = dataset[0]
    assert len(dataset) == 2
    assert feature.dtype == torch.float32
    assert feature.shape == (4, config.CNN_FEAT_DIM)
    assert caption.shape == (8,)
    assert int(length) > 0


def test_padding_steps_do_not_change_coverage_loss():
    alphas = torch.rand(1, 4, 5)
    valid = torch.tensor([[True, True, False, False]])
    first = attention_coverage_loss(alphas, valid)
    alphas[:, 2:] = torch.rand_like(alphas[:, 2:]) * 100
    second = attention_coverage_loss(alphas, valid)
    assert torch.allclose(first, second)
