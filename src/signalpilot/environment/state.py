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
    """Signal states, including non-conflicting transition states."""

    NS_GREEN = "ns_green"
    EW_GREEN = "ew_green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


class Action(IntEnum):
    """Actions understood by the Phase 1 environment contract."""

    KEEP_PHASE = 0
    SWITCH_PHASE = 1


@dataclass(frozen=True)
class IntersectionState:
    """A read-only snapshot of the simulator at one time step."""

    time_step: int
    phase: SignalPhase
    queues: tuple[int, int, int, int]
    seed: int | None
    elapsed_green_time: int = 0
    transition_remaining: int = 0
    pending_phase: SignalPhase | None = None
    total_arrivals: int = 0
    total_served: int = 0
    total_waiting_time: int = 0

    def __post_init__(self) -> None:
        if self.time_step < 0:
            raise ValueError("time_step must be non-negative")
        if len(self.queues) != len(Approach):
            raise ValueError("queues must contain one value per approach")
        if any(queue < 0 for queue in self.queues):
            raise ValueError("queues must be non-negative")
        if self.elapsed_green_time < 0 or self.transition_remaining < 0:
            raise ValueError("time counters must be non-negative")
        if self.total_arrivals < 0 or self.total_served < 0:
            raise ValueError("vehicle counters must be non-negative")
        if self.total_waiting_time < 0:
            raise ValueError("total_waiting_time must be non-negative")
        if self.pending_phase in (SignalPhase.YELLOW, SignalPhase.ALL_RED):
            raise ValueError("pending_phase must be a green phase")

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
