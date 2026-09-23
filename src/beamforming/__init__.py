"""
src/beamforming

Phase 4A: MRT and RZF beamforming for the physical-layer (per-antenna,
complex) channel. See src.beamforming.precoders for implementations.

Phase 4B: SINR-constrained (QoS) joint beamforming with per-AP power
constraints and SINR-target bisection. See src.beamforming.constrained
for implementation and full solver documentation. Additive only --
mrt_weights / rzf_weights / normalize_per_ap_power below are untouched.

This package was a placeholder (empty __init__.py) prior to Phase 4A.
"""

from src.beamforming.precoders import (
    mrt_weights,
    rzf_weights,
    normalize_per_ap_power,
)
from src.beamforming.constrained import (
    constrained_sinr_beamforming,
    bisection_sinr_beamforming,
)

__all__ = [
    "mrt_weights",
    "rzf_weights",
    "normalize_per_ap_power",
    "constrained_sinr_beamforming",
    "bisection_sinr_beamforming",
]
