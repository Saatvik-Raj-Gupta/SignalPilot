import numpy as np

from signalpilot.environment import (
    SignalPhase,
    discharge_queues,
    is_green,
    opposing_green,
    sample_arrivals,
)


def test_vehicle_flow_helpers_are_seeded_and_signal_safe() -> None:
    first = sample_arrivals(np.random.default_rng(4), (0.2, 0.2, 0.2, 0.2))
    second = sample_arrivals(np.random.default_rng(4), (0.2, 0.2, 0.2, 0.2))

    assert first == second
    assert discharge_queues(SignalPhase.YELLOW, (2, 2, 2, 2), 1) == (0, 0, 0, 0)
    assert discharge_queues(SignalPhase.EW_GREEN, (2, 2, 2, 2), 1) == (0, 0, 1, 0)
    assert is_green(SignalPhase.NS_GREEN)
    assert not is_green(SignalPhase.ALL_RED)
    assert opposing_green(SignalPhase.NS_GREEN) is SignalPhase.EW_GREEN
