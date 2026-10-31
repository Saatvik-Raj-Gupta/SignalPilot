"""Metrics collected without changing simulator behavior."""

from dataclasses import dataclass, field

import numpy as np

from signalpilot.environment import IntersectionState, StepResult


@dataclass(frozen=True)
class EpisodeMetrics:
    """Summary of one completed simulation episode."""

    mean_waiting_time: float
    p95_waiting_time: float
    max_waiting_time: int
    mean_queue_length: float
    max_queue_length: int
    throughput: int
    total_delay: int
    phase_switches: int
    episode_reward: float


@dataclass
class MetricsCollector:
    """Accumulate observable episode values and calculate a summary."""

    queue_totals: list[int] = field(default_factory=list)
    queue_maxima: list[int] = field(default_factory=list)
    waiting_times: list[int] = field(default_factory=list)
    served_waiting_times: list[int] = field(default_factory=list)
    throughput: int = 0
    total_delay: int = 0
    phase_switches: int = 0
    episode_reward: float = 0.0

    def record_state(self, state: IntersectionState) -> None:
        """Record queue and waiting values from a state snapshot."""

        queue_length = sum(state.queues)
        self.queue_totals.append(queue_length)
        self.queue_maxima.append(queue_length)
        self.waiting_times.append(state.total_waiting_time)

    def record_step(self, result: StepResult) -> None:
        """Record one environment transition."""

        self.record_state(result.state)
        self.throughput = result.state.total_served
        self.total_delay = result.state.total_waiting_time
        self.phase_switches += int(result.info.get("phase_switched", False))
        self.episode_reward += result.reward
        self.served_waiting_times.extend(
            result.info.get("served_waiting_times", ())
        )

    def finish(self) -> EpisodeMetrics:
        """Return metrics, rejecting summaries of empty episodes."""

        if not self.queue_totals:
            raise ValueError("cannot finish an empty episode")
        return EpisodeMetrics(
            mean_waiting_time=(
                sum(self.served_waiting_times) / len(self.served_waiting_times)
                if self.served_waiting_times
                else 0.0
            ),
            p95_waiting_time=(
                float(np.percentile(self.served_waiting_times, 95))
                if self.served_waiting_times
                else 0.0
            ),
            max_waiting_time=max(self.waiting_times),
            mean_queue_length=sum(self.queue_totals) / len(self.queue_totals),
            max_queue_length=max(self.queue_maxima),
            throughput=self.throughput,
            total_delay=self.total_delay,
            phase_switches=self.phase_switches,
            episode_reward=self.episode_reward,
        )
