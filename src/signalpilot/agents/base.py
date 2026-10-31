"""Common controller interface and episode execution helper."""

from typing import Protocol

from signalpilot.environment import Action, IntersectionEnvironment, IntersectionState
from signalpilot.metrics.collector import EpisodeMetrics, MetricsCollector


class Controller(Protocol):
    """Protocol shared by baseline and learning controllers."""

    def reset(self) -> None:
        """Reset controller-local episode state."""

    def action(self, state: IntersectionState) -> Action:
        """Choose an action for the supplied state."""


def run_episode(
    environment: IntersectionEnvironment,
    controller: Controller,
    *,
    seed: int | None = None,
) -> EpisodeMetrics:
    """Run one complete episode and return its measured performance."""

    state = environment.reset(seed=seed)
    controller.reset()
    collector = MetricsCollector()
    collector.record_state(state)

    while True:
        result = environment.step(controller.action(state))
        collector.record_step(result)
        state = result.state
        if result.terminated or result.truncated:
            return collector.finish()
