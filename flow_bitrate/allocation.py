"""Exact allocation for four RVQ depths. Costs include a two-bit rate flag per block."""

import struct

import numpy as np

from .codec import CODEBOOKS

FRAMES = 150
BASE_BITS = 4 * FRAMES * 10 + 2
UPGRADE_BITS = 2 * FRAMES * 10
HEADER_BITS = 96  # FB01 + uint32 block count + uint32 sampling seed


def stream_bits(choices):
    bits = HEADER_BITS + len(choices) * BASE_BITS + UPGRADE_BITS * np.asarray(choices).sum()
    return int((bits + 7) // 8 * 8)


def allocate(distortion, budget_kbps):
    distortion = np.asarray(distortion, dtype=np.float64)
    if distortion.ndim != 2 or distortion.shape[1] != 4 or not len(distortion) or not np.isfinite(distortion).all():
        raise ValueError("Expected finite [blocks, 4] distortions")
    count = len(distortion)
    available = int(np.floor(budget_kbps * 1000 * 2 * count / 8)) * 8
    upgrades = (available - HEADER_BITS - count * BASE_BITS) // UPGRADE_BITS
    if upgrades < 0:
        raise ValueError("Budget cannot fit the minimum rate plus stream overhead")
    upgrades = min(upgrades, 3 * count)
    previous = np.full(upgrades + 1, np.inf)
    previous[0] = 0
    back = np.zeros((count, upgrades + 1), dtype=np.int8)
    for block, costs in enumerate(distortion):
        candidates = np.full((4, upgrades + 1), np.inf)
        for level in range(min(3, upgrades) + 1):
            candidates[level, level:] = previous[:upgrades + 1 - level] + costs[level]
        back[block] = candidates.argmin(0)
        previous = candidates.min(0)
    remaining = int(previous.argmin())  # <= budget; more bits need not improve a flow reconstruction
    choices = np.zeros(count, dtype=np.int64)
    for block in range(count - 1, -1, -1):
        choices[block] = back[block, remaining]
        remaining -= choices[block]
    return choices


def uniform_choices(count, budget_kbps):
    feasible = [i for i in range(4) if stream_bits(np.full(count, i)) <= budget_kbps * 1000 * 2 * count]
    if not feasible:
        raise ValueError("Budget too small")
    return np.full(count, max(feasible), dtype=np.int64)


def pack_codes(codes, choices, seed=0):
    """codes [blocks, 10, 150] -> experimental fixed-length bitstream."""
    codes, choices = np.asarray(codes), np.asarray(choices)
    if codes.shape != (len(choices), 10, FRAMES) or len(choices) == 0:
        raise ValueError("Expected [blocks, 10, 150] codes")
    if np.any((choices < 0) | (choices > 3)) or np.any((codes < 0) | (codes >= 1024)):
        raise ValueError("Invalid RVQ index or rate")
    parts = []
    for block, choice in zip(codes, choices):
        parts.append(((int(choice) >> np.arange(1, -1, -1)) & 1).astype(np.uint8))
        values = block[:CODEBOOKS[choice]].reshape(-1)
        parts.append(((values[:, None] >> np.arange(9, -1, -1)) & 1).astype(np.uint8).reshape(-1))
    return b"FB01" + struct.pack(">II", len(choices), seed) + np.packbits(np.concatenate(parts)).tobytes()


def unpack_codes(stream):
    if len(stream) < 12 or stream[:4] != b"FB01":
        raise ValueError("Not a flow-bitrate stream")
    count, seed = struct.unpack(">II", stream[4:12])
    if count == 0 or count > (len(stream) * 8 - HEADER_BITS) // BASE_BITS:
        raise ValueError("Truncated/invalid stream")
    bits = np.unpackbits(np.frombuffer(stream[12:], dtype=np.uint8))
    offset, codes, choices = 0, [], []
    for _ in range(count):
        choice = int(bits[offset:offset + 2] @ np.array([2, 1]))
        offset += 2
        size = CODEBOOKS[choice] * FRAMES * 10
        if offset + size > len(bits):
            raise ValueError("Truncated RVQ codes")
        values = bits[offset:offset + size].reshape(-1, 10) @ (2 ** np.arange(9, -1, -1))
        codes.append(values.reshape(CODEBOOKS[choice], FRAMES))
        choices.append(choice)
        offset += size
    if len(bits) - offset > 7 or bits[offset:].any():
        raise ValueError("Unexpected trailing bits")
    return codes, np.asarray(choices), seed
