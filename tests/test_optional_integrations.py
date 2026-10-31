from importlib.util import find_spec

import pytest

from signalpilot.agents import ReplayBuffer, Transition
from signalpilot.integrations.sumo import SumoUnavailableError, require_sumo


def test_replay_buffer_is_bounded_and_seeded() -> None:
    buffer = ReplayBuffer(capacity=2, seed=1)
    transition = Transition((0, 0, 0, 0), 0, 1.0, (0, 0, 0, 0), False)
    buffer.append(transition)
    buffer.append(transition)
    buffer.append(transition)

    assert len(buffer) == 2
    assert buffer.sample(1) == [transition]


def test_sumo_requirement_reports_missing_external_dependency() -> None:
    if find_spec("traci") is None:
        with pytest.raises(SumoUnavailableError, match="SUMO integration"):
            require_sumo()
    else:
        require_sumo()
