"""Bridges and phase weights in FlowDec's compressed complex STFT coordinates."""

import torch


PHASE_FLOOR = 0.05


def noise_scale(sigma, multiplier):
    return (sigma * multiplier).clamp_min(1e-6)


def phase_weight(amplitude, tau):
    power = amplitude.square()
    return PHASE_FLOOR + (1 - PHASE_FLOOR) * power / (power + tau.square())


def matched_draw(condition, sigma, seeds):
    # CPU draws match across devices and batch shapes, including DDP padding slots.
    noise, times = [], []
    for seed in seeds:
        generator = torch.Generator().manual_seed(max(seed, 0))
        noise.append(torch.randn(condition.shape[1:], dtype=condition.dtype, generator=generator))
        times.append(torch.rand((), generator=generator))
    return condition + sigma * torch.stack(noise).to(condition.device), torch.stack(times).to(condition.device)


def warped_disk_bridge(start, end, t, tau):
    """Return the geodesic and its time derivative in Cartesian coordinates.

    Minimize |exp(-i*w) end - start|² + tau² w² over the shortest rotation.
    Reflect negative phase gaps to [0, pi]. For a nonzero gap below pi there
    is one minimum in this interval. At pi the through-origin stationary
    point is optimal iff |start| |end| <= tau². Choose the positive rotation
    when the two paths around the origin tie.
    """
    cross = start.conj() * end
    imaginary = cross.imag.abs()
    upper = torch.atan2(imaginary, cross.real)
    lower = torch.zeros_like(upper)
    tau_squared = tau.square()
    for _ in range(22):
        angle = (lower + upper) * 0.5
        # Half the derivative of the rotation cost, without subtracting near-pi angles.
        decreasing = tau_squared * angle < imaginary * angle.cos() - cross.real * angle.sin()
        lower = torch.where(decreasing, angle, lower)
        upper = torch.where(decreasing, upper, angle)
    turn = (lower + upper) * 0.5
    straight = (imaginary == 0) & (cross.real >= -tau_squared)
    turn = torch.where(straight, 0.0, torch.where(cross.imag < 0, -turn, turn))
    unit = torch.ones_like(turn)
    aligned_end = torch.polar(unit, -turn) * end
    rotation = torch.polar(unit, t * turn)
    state = rotation * ((1 - t) * start + t * aligned_end)
    velocity = rotation * (aligned_end - start) + 1j * turn * state
    return state, velocity
