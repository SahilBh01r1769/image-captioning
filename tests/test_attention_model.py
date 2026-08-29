import torch

from attention_model import AdditiveAttention, AttentionLSTMDecoder, SpatialCNNEncoder, attention_coverage_loss


def test_additive_attention_normalizes_over_locations():
    attention = AdditiveAttention(encoder_dim=16, hidden_dim=12, attention_dim=8)
    encoder = torch.randn(3, 9, 16)
    hidden = torch.randn(3, 12)
    context, alpha = attention(encoder, hidden)
    assert context.shape == (3, 16)
    assert alpha.shape == (3, 9)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(3), atol=1e-5)


def test_attention_decoder_returns_logits_and_maps():
    decoder = AttentionLSTMDecoder(
        vocab_size=20,
        encoder_dim=16,
        embed_dim=10,
        hidden_dim=12,
        attention_dim=8,
        dropout=0.0,
        pad_idx=0,
    )
    encoder = torch.randn(2, 9, 16)
    captions = torch.tensor([
        [1, 4, 5, 2, 0],
        [1, 6, 2, 0, 0],
    ])
    logits, alphas = decoder(encoder, captions)
    assert logits.shape == (2, 4, 20)
    assert alphas.shape == (2, 4, 9)
    assert torch.allclose(alphas.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_coverage_loss_accepts_padding_mask():
    alphas = torch.full((2, 4, 9), 1 / 9)
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])
    loss = attention_coverage_loss(alphas, valid)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_spatial_encoder_preserves_locations_without_pretrained_download():
    encoder = SpatialCNNEncoder(encoder_dim=32, pretrained=False)
    encoder.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        features = encoder(image)
    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[-1] == 32
    assert features.shape[1] > 1
