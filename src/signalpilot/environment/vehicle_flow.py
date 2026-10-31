"""Seeded vehicle arrival and discharge mechanics."""

from collections.abc import Sequence

import numpy as np

from signalpilot.environment.state import SignalPhase


def sample_arrivals(
    rng: np.random.Generator,
    arrival_rates: Sequence[float],
) -> tuple[int, ...]:
    """Sample one Poisson arrival count for each approach."""

    return tuple(int(rng.poisson(rate)) for rate in arrival_rates)


def discharge_queues(
    phase: SignalPhase,
    queues: tuple[int, int, int, int],
    saturation_flow: int,
) -> tuple[int, int, int, int]:
    """Discharge vehicles only from the movement permitted by ``phase``."""

    if phase not in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
        return (0, 0, 0, 0)
    allowed = (0, 1) if phase is SignalPhase.NS_GREEN else (2, 3)
    remaining = saturation_flow
    served = [0, 0, 0, 0]
    for index in allowed:
        amount = min(queues[index], remaining)
        served[index] = amount
        remaining -= amount
    return tuple(served)
