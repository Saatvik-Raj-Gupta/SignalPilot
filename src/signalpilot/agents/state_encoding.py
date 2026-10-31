"""Compact discrete state encoding for tabular learning."""

from dataclasses import dataclass

from signalpilot.environment import IntersectionState, SignalPhase


@dataclass(frozen=True)
class StateEncoder:
    """Map simulator states to bounded, inspectable tuples.

    Queue boundaries are inclusive lower bounds. For example, the default
    ``(0, 1, 4, 8)`` produces buckets 0, 1-3, 4-7, and 8+.
    """

    queue_boundaries: tuple[int, ...] = (0, 1, 4, 8)
    elapsed_green_boundaries: tuple[int, ...] = (0, 1, 5, 10)

    def __post_init__(self) -> None:
        if not self.queue_boundaries or self.queue_boundaries[0] != 0:
            raise ValueError("queue_boundaries must start with zero")
        if any(
            left >= right
            for left, right in zip(self.queue_boundaries, self.queue_boundaries[1:])
        ):
            raise ValueError("queue_boundaries must be strictly increasing")
        if not self.elapsed_green_boundaries or self.elapsed_green_boundaries[0] != 0:
            raise ValueError("elapsed_green_boundaries must start with zero")

    def encode(self, state: IntersectionState) -> tuple[int, int, int, int]:
        """Return NS bucket, EW bucket, phase bucket, and elapsed bucket."""

        ns_queue = state.queues[0] + state.queues[1]
        ew_queue = state.queues[2] + state.queues[3]
        phase = {
            SignalPhase.NS_GREEN: 0,
            SignalPhase.EW_GREEN: 1,
            SignalPhase.YELLOW: 2,
            SignalPhase.ALL_RED: 3,
        }[state.phase]
        return (
            self._bucket(ns_queue, self.queue_boundaries),
            self._bucket(ew_queue, self.queue_boundaries),
            phase,
            self._bucket(state.elapsed_green_time, self.elapsed_green_boundaries),
        )

    @staticmethod
    def _bucket(value: int, boundaries: tuple[int, ...]) -> int:
        return max(
            index for index, boundary in enumerate(boundaries) if value >= boundary
        )
