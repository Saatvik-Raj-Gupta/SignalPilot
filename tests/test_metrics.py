from signalpilot.environment import (
    Action,
    EnvironmentConfig,
    IntersectionEnvironment,
)
from signalpilot.metrics import MetricsCollector


def test_metrics_collector_reports_hand_calculated_values() -> None:
    environment = IntersectionEnvironment(
        EnvironmentConfig(
            episode_horizon=2,
            initial_queues=(2, 0, 0, 0),
            arrival_rates=(0.0, 0.0, 0.0, 0.0),
            saturation_flow=1,
        )
    )
    collector = MetricsCollector()
    state = environment.reset()
    collector.record_state(state)
    collector.record_step(environment.step(Action.KEEP_PHASE))
    collector.record_step(environment.step(Action.KEEP_PHASE))

    metrics = collector.finish()

    assert metrics.throughput == 2
    assert metrics.total_delay == 1
    assert metrics.max_queue_length == 2
    assert metrics.mean_queue_length == 1.0
    assert metrics.episode_reward == -1.0
