"""Fixed recipes for the six-codebook Jamendo ablations."""

import math
from dataclasses import dataclass, replace

FIRST_CROPS = 192_000
LONG_FIRST_CROPS = 1_536_000
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
    band_width: int = 256
    highband_loss_weight: float = 0.0


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
    Arm("small-disk-sigma-three-quarters-epoch-cosine", width=64, noise="scale:0.75",
        schedule="epoch-cosine"),
    replace(REFERENCE, name="medium-disk-multiscale-sigma-jamendo-raw", architecture="multiscale",
            noise="jamendo-raw"),
    Arm("small-disk-multiscale", width=64, architecture="multiscale"),
    replace(REFERENCE, name="medium-disk-band-attention-sigma-jamendo-raw", architecture="band-attention",
            noise="jamendo-raw"),
), start=1))

LONG_UNET = replace(ARMS[12], schedule="wsd-long")
LONG_BAND_ORIGINAL = replace(ARMS[20], schedule="wsd-long")
LONG_BAND_RAW = replace(ARMS[29], schedule="wsd-long")
HIGHBAND_WEIGHT = 0.05

ARMS.update({
    30: LONG_UNET,
    31: replace(LONG_UNET, name=ARMS[27].name, architecture="multiscale"),
    32: replace(LONG_UNET, name=LONG_UNET.name + "-highband", highband_loss_weight=HIGHBAND_WEIGHT),
    33: replace(LONG_UNET, name=ARMS[27].name + "-highband", architecture="multiscale",
                highband_loss_weight=HIGHBAND_WEIGHT),
    34: LONG_BAND_ORIGINAL,
    35: replace(LONG_BAND_ORIGINAL, name=LONG_BAND_ORIGINAL.name + "-w320", band_width=320),
    36: replace(LONG_BAND_ORIGINAL, name=LONG_BAND_ORIGINAL.name + "-highband", highband_loss_weight=HIGHBAND_WEIGHT),
    37: replace(LONG_BAND_ORIGINAL, name=LONG_BAND_ORIGINAL.name + "-w320-highband", band_width=320,
                highband_loss_weight=HIGHBAND_WEIGHT),
    38: LONG_BAND_RAW,
    39: replace(LONG_BAND_RAW, name=LONG_BAND_RAW.name + "-w320", band_width=320),
    40: replace(LONG_BAND_RAW, name=LONG_BAND_RAW.name + "-highband", highband_loss_weight=HIGHBAND_WEIGHT),
    41: replace(LONG_BAND_RAW, name=LONG_BAND_RAW.name + "-w320-highband", band_width=320,
                highband_loss_weight=HIGHBAND_WEIGHT),
})


def real_crops(track_count, batch, updates):
    """Number of real sampler entries consumed, excluding the epoch-tail padding."""
    epochs, updates_in_epoch = divmod(updates, math.ceil(track_count / batch))
    return epochs * track_count + updates_in_epoch * batch


def wsd_factor(step, effective_batch, schedule="wsd"):
    warmup, decay_start, decay_end = (1_000, 32_000, 48_000) if schedule == "wsd-long" else (500, 10_000, 12_000)
    equivalent_step = step * effective_batch / 64
    if equivalent_step < warmup:
        fraction = equivalent_step / warmup
        return 0.05 + 0.95 * (1 - math.cos(math.pi * fraction)) / 2
    fraction = min(1.0, max(0.0, (equivalent_step - decay_start) / (decay_end - decay_start)))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * fraction)) / 2
