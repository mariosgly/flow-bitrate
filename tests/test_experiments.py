import csv
import itertools
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from flow_bitrate.allocation import allocate, pack_codes, stream_bits, uniform_choices, unpack_codes
from flow_bitrate.codec import BLOCK_SAMPLES, UPSTREAM, FlowCodec
from flow_bitrate.data import load_cache, read_manifest
from flow_bitrate.metrics import si_sdr, spectral_loss


@pytest.mark.parametrize("seed", range(5))
def test_exact_allocation_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    costs = rng.normal(size=(4, 4))  # deliberately non-monotonic
    budget = 5.1
    choices = allocate(costs, budget)
    candidates = [x for x in itertools.product(range(4), repeat=4) if stream_bits(x) <= budget * 8000]
    best = min(sum(costs[i, k] for i, k in enumerate(x)) for x in candidates)
    assert sum(costs[i, k] for i, k in enumerate(choices)) == pytest.approx(best)
    assert stream_bits(choices) <= budget * 8000
    assert stream_bits(uniform_choices(4, budget)) <= budget * 8000


def test_budget_and_bitstream_roundtrip():
    codes = np.random.default_rng(0).integers(0, 1024, size=(4, 10, 150))
    choices = np.arange(4)
    stream = pack_codes(codes, choices, seed=12345)
    decoded, rates, seed = unpack_codes(stream)
    assert seed == 12345
    assert len(stream) * 8 == stream_bits(choices)
    np.testing.assert_array_equal(rates, choices)
    for block, choice, original in zip(decoded, choices, codes):
        np.testing.assert_array_equal(block, original[:4 + choice * 2])
    with pytest.raises(ValueError):
        unpack_codes(stream[:-1])
    with pytest.raises(ValueError):
        allocate(np.zeros((2, 4)), 3.0)
    with pytest.raises(ValueError):
        allocate([[float("nan")] * 4], 4.55)
    # More bits really can be worse; the solver must be allowed to leave budget unused.
    np.testing.assert_array_equal(allocate(np.tile([0, 1, 2, 3], (3, 1)), 8), [0, 0, 0])


def test_split_leakage_is_rejected(tmp_path):
    audio = tmp_path / "a.wav"
    other = tmp_path / "b.wav"
    sf.write(audio, np.zeros(1000), 48000)
    sf.write(other, np.zeros(1000), 48000)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([["path", "split", "group"], [audio.name, "train", "speaker1"],
                          [other.name, "test", "speaker1"]])
    with pytest.raises(ValueError, match="leaks"):
        read_manifest(manifest)


def test_incomplete_or_mixed_cache_is_rejected(tmp_path):
    torch.save({"metadata": {"num_shards": 2}, "shard": 0}, tmp_path / "shard_00000.pt")
    with pytest.raises(ValueError, match="incomplete"):
        load_cache(tmp_path)
    torch.save({"metadata": {"num_shards": 2, "changed": True}, "shard": 1}, tmp_path / "shard_00001.pt")
    with pytest.raises(ValueError, match="Mixed"):
        load_cache(tmp_path)


class IdentityFeatures(nn.Module):
    def forward(self, x):
        return torch.complex(x[:, :, None], torch.zeros_like(x[:, :, None]))

    def invert(self, x, orig_length):
        return x[:, :, 0, :orig_length].real


class ZeroVelocity(nn.Module):
    def forward(self, state, condition, time):
        return torch.zeros_like(state)


def test_shifted_prior_and_seeded_sampler():
    codec = FlowCodec.__new__(FlowCodec)
    nn.Module.__init__(codec)
    codec.features = IdentityFeatures()
    codec.backbone = ZeroVelocity()
    codec.register_buffer("sigma", torch.tensor(0.25))
    coarse = torch.ones(2, 1, 64)
    original = coarse * 0.5
    error, power, speed = codec.flow_loss(original, coarse, seed=9, time=0.5)
    generator = torch.Generator().manual_seed(9)
    noise = torch.randn(2, 1, 1, 64, dtype=torch.complex64, generator=generator)
    expected = (-0.5 - 0.25 * noise).abs().square().flatten(1).mean(1)
    torch.testing.assert_close(error, expected)
    torch.testing.assert_close(power, expected)
    assert not speed.any()
    torch.testing.assert_close(codec.enhance(coarse, seed=2), codec.enhance(coarse, seed=2))
    assert not torch.equal(codec.enhance(coarse, seed=2), codec.enhance(coarse, seed=3))
    assert codec.probes(original, coarse, seed=1).shape == (2, 10)


def test_real_upstream_backbone_backward_and_bfloat16():
    if not UPSTREAM.exists():
        pytest.skip("Run scripts/setup.sh to install the pinned source")
    sys.path.insert(0, str(UPSTREAM))
    from flowdec.backbones.ncsnpp import NCSNpp
    from flowdec.data.feature_extractors import AmplitudeCompressedComplexSTFT

    torch.set_num_threads(2)
    model = NCSNpp(image_size=64, nf=8, ch_mult=(1, 1), num_res_blocks=1,
                  attn_resolutions=(), bottleneck_attn=False, num_channels=4,
                  output_layer_kwargs={"kernel_size": 1, "bias": False, "padding": "same"})
    state = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
    prediction = model(state, state, torch.tensor([0.5]))
    assert prediction.shape == state.shape
    prediction.abs().square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(model(state, state, torch.tensor([0.5]))).all()
    features = AmplitudeCompressedComplexSTFT("hann", 1534, 48000, n_hops=4, alpha=0.3, beta=0.33)
    audio = torch.randn(1, 1, BLOCK_SAMPLES) * 0.1
    restored = features.invert(features(audio), orig_length=BLOCK_SAMPLES)
    torch.testing.assert_close(restored, audio, atol=1e-5, rtol=1e-4)


def test_real_dac_variable_depth_shapes():
    from dac import DAC

    torch.set_num_threads(2)
    codec = FlowCodec.__new__(FlowCodec)
    nn.Module.__init__(codec)
    codec.ndac = DAC(encoder_dim=4, encoder_rates=[2, 4, 8, 10], latent_dim=1024,
                     decoder_dim=64, decoder_rates=[10, 8, 4, 2], n_codebooks=10,
                     codebook_size=1024, codebook_dim=8, sample_rate=48000).eval()
    with torch.no_grad():
        latent, codes = codec.encode(torch.randn(1, 1, BLOCK_SAMPLES) * 0.1)
        assert latent.shape == (1, 1024, 150)
        assert codes.shape == (1, 10, 150)
        coarse, quantized = codec.coarse(codes, 4)
        assert coarse.shape == (1, 1, BLOCK_SAMPLES)
        assert quantized.shape == latent.shape
        torch.testing.assert_close(codec.coarse_at_rates(codes, torch.tensor([0])), coarse)


def test_metrics_handle_silence_and_identity():
    x = torch.zeros(1, 1, BLOCK_SAMPLES)
    assert spectral_loss(x, x).item() == 0
    assert torch.isfinite(si_sdr(x, x)).all()
    x = torch.randn_like(x) * 0.1
    assert spectral_loss(x, x).item() == 0
    assert spectral_loss(x, x * 0.5).item() > 0
