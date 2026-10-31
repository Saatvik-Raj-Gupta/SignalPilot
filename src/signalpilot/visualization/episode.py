"""Non-invasive recording helpers for visualization and reporting."""

from dataclasses import dataclass

from signalpilot.agents.base import Controller
from signalpilot.environment import (
    IntersectionEnvironment,
    IntersectionState,
    SignalPhase,
)
from signalpilot.metrics import EpisodeMetrics, MetricsCollector


@dataclass(frozen=True)
class EpisodeTrace:
    """Full observable trace of one completed episode."""

    states: tuple[IntersectionState, ...]
    phases: tuple[SignalPhase, ...]
    rewards: tuple[float, ...]
    waiting_times: tuple[int, ...]
    metrics: EpisodeMetrics


def record_episode(
    environment: IntersectionEnvironment,
    controller: Controller,
    *,
    seed: int | None = None,
) -> EpisodeTrace:
    """Run and retain an episode trace without changing environment logic."""

    state = environment.reset(seed=seed)
    controller.reset()
    collector = MetricsCollector()
    states = [state]
    rewards: list[float] = []
    waiting_times = [state.total_waiting_time]
    collector.record_state(state)

    while True:
        result = environment.step(controller.action(state))
        collector.record_step(result)
        states.append(result.state)
        rewards.append(result.reward)
        waiting_times.append(result.state.total_waiting_time)
        state = result.state
        if result.terminated or result.truncated:
            return EpisodeTrace(
                states=tuple(states),
                phases=tuple(item.phase for item in states),
                rewards=tuple(rewards),
                waiting_times=tuple(waiting_times),
                metrics=collector.finish(),
            )
