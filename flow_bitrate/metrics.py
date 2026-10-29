import torch


def spectral_loss(original, reconstructed):
    """Mean log-magnitude error, three resolutions; one value per block."""
    if original.shape != reconstructed.shape:
        raise ValueError(f"Spectral loss needs equal waveform shapes: {original.shape} vs {reconstructed.shape}")
    original, reconstructed = original.flatten(0, 1), reconstructed.flatten(0, 1)
    losses = []
    for size in (256, 1024, 4096):
        window = torch.hann_window(size, device=original.device)
        x = torch.stft(original, size, size // 4, window=window, return_complex=True).abs()
        y = torch.stft(reconstructed, size, size // 4, window=window, return_complex=True).abs()
        losses.append((x.clamp_min(1e-4).log() - y.clamp_min(1e-4).log()).abs().flatten(1).mean(1))
    return torch.stack(losses).mean(0)


def si_sdr(original, reconstructed):
    if original.shape != reconstructed.shape:
        raise ValueError(f"SI-SDR needs equal waveform shapes: {original.shape} vs {reconstructed.shape}")
    x, y = original.flatten(1), reconstructed.flatten(1)
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    projection = (x * y).sum(1, keepdim=True) / x.square().sum(1, keepdim=True).clamp_min(1e-8) * x
    return 10 * (projection.square().sum(1).clamp_min(1e-8) /
                 (y - projection).square().sum(1).clamp_min(1e-8)).log10()


def signal_features(audio):
    window = torch.hann_window(1024, device=audio.device)
    spectrum = torch.stft(audio[:, 0], 1024, 256, window=window, return_complex=True).abs()
    spectrum = spectrum / spectrum.sum(1, keepdim=True).clamp_min(1e-8)
    flux = (spectrum[..., 1:] - spectrum[..., :-1]).clamp_min(0).sum(1).mean(1)
    rms = audio.square().flatten(1).mean(1).sqrt().clamp_min(1e-8).log()
    return torch.stack((rms, flux), 1)
