"""Fixed recipes for the first six-codebook Jamendo ablation round."""

import math
from dataclasses import dataclass, replace

FIRST_CROPS = 192_000
CACHE_LOOKAHEAD = 1024
CALIBRATION_FILE = "start-noise.json"


@dataclass(frozen=True)
class Arm:
    name: str
    width: int = 96
    geometry: str = "cylindrical"
    phase_loss: str = "disk"
    architecture: str = "unet"
    noise: str = "original"
    schedule: str = "wsd"
    batch: int = 64


REFERENCE = Arm("medium-disk")
ARMS = dict(enumerate((
    Arm("small-disk", width=64),
    Arm("small-disk-epoch-cosine", width=64, schedule="epoch-cosine"),
    REFERENCE,
    replace(REFERENCE, name="medium-cylindrical-unit", phase_loss="unit"),
    replace(REFERENCE, name="medium-cartesian", geometry="cartesian", phase_loss="unit"),
    replace(REFERENCE, name="large-disk", width=160),
    replace(REFERENCE, name="medium-disk-sigma-half", noise="scale:0.5"),
    replace(REFERENCE, name="medium-disk-sigma-three-quarters", noise="scale:0.75"),
    replace(REFERENCE, name="medium-disk-sigma-five-quarters", noise="scale:1.25"),
    replace(REFERENCE, name="medium-disk-sigma-flat-rms", noise="flat-rms"),
    replace(REFERENCE, name="medium-disk-sigma-jamendo-matched", noise="jamendo-matched"),
    replace(REFERENCE, name="medium-disk-sigma-jamendo-raw", noise="jamendo-raw"),
    replace(REFERENCE, name="medium-disk-sigma-flat-066", noise="flat:0.66"),
    replace(REFERENCE, name="medium-disk-frequency-first", architecture="frequency-first"),
    replace(REFERENCE, name="medium-disk-dilated", architecture="dilated"),
    replace(REFERENCE, name="medium-disk-film", architecture="film"),
    replace(REFERENCE, name="medium-disk-multiscale", architecture="multiscale"),
    replace(REFERENCE, name="medium-disk-convnext", architecture="convnext"),
    replace(REFERENCE, name="medium-disk-band-tcn", architecture="band-tcn"),
    replace(REFERENCE, name="medium-disk-band-attention", architecture="band-attention"),
    replace(REFERENCE, name="medium-disk-ncsnpp-polar", architecture="ncsnpp"),
    Arm("small-disk-sigma-three-quarters", width=64, noise="scale:0.75"),
    replace(REFERENCE, name="medium-cartesian-band-tcn", geometry="cartesian",
            phase_loss="unit", architecture="band-tcn"),
    Arm("small-disk-batch128", width=64, batch=128),
    replace(REFERENCE, name="medium-disk-separate-heads", architecture="separate-heads"),
), start=1))


def real_crops(track_count, batch, updates):
    """Number of real sampler entries consumed, excluding the epoch-tail padding."""
    epochs, updates_in_epoch = divmod(updates, math.ceil(track_count / batch))
    return epochs * track_count + updates_in_epoch * batch


def wsd_factor(step, effective_batch):
    equivalent_step = step * effective_batch / 64
    if equivalent_step < 500:
        fraction = equivalent_step / 500
        return 0.05 + 0.95 * (1 - math.cos(math.pi * fraction)) / 2
    fraction = min(1.0, max(0.0, (equivalent_step - 10_000) / 2_000))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * fraction)) / 2
