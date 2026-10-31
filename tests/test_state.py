import pytest

from signalpilot.environment import Approach, IntersectionState


def test_queue_for_returns_named_approach_queue() -> None:
    state = IntersectionState(0, "ns_green", (1, 2, 3, 4), None)

    assert state.queue_for(Approach.NORTH) == 1
    assert state.queue_for(Approach.WEST) == 4


def test_state_rejects_invalid_queue_shape_or_values() -> None:
    with pytest.raises(ValueError, match="one value per approach"):
        IntersectionState(0, "ns_green", (0, 0), None)

    with pytest.raises(ValueError, match="non-negative"):
        IntersectionState(0, "ns_green", (0, 0, -1, 0), None)
