"""Typed values exchanged by the SignalPilot environment."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Approach(StrEnum):
    """The four approaches to the intersection."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class SignalPhase(StrEnum):
    """The two movement phases used by the MVP."""

    NS_GREEN = "ns_green"
    EW_GREEN = "ew_green"


class Action(IntEnum):
    """Actions understood by the Phase 1 environment contract."""

    KEEP_PHASE = 0
    SWITCH_PHASE = 1


@dataclass(frozen=True)
class IntersectionState:
    """A read-only snapshot of the simulator at one time step.

    Queue dynamics are intentionally not implemented in Phase 1, so queues are
    represented explicitly as four zero-valued approach counts. Later phases
    will update these values using the same state contract.
    """

    time_step: int
    phase: SignalPhase
    queues: tuple[int, int, int, int]
    seed: int | None

    def __post_init__(self) -> None:
        if self.time_step < 0:
            raise ValueError("time_step must be non-negative")
        if len(self.queues) != len(Approach):
            raise ValueError("queues must contain one value per approach")
        if any(queue < 0 for queue in self.queues):
            raise ValueError("queues must be non-negative")

    def queue_for(self, approach: Approach) -> int:
        """Return the queue length for one named approach."""

        return self.queues[list(Approach).index(approach)]


@dataclass(frozen=True)
class StepResult:
    """The result returned after applying one action."""

    state: IntersectionState
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, object]
