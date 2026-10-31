from signalpilot.agents import FixedTimeController, RandomController, run_episode
from signalpilot.environment import (
    Action,
    DemandScenario,
    EnvironmentConfig,
    IntersectionEnvironment,
    SignalPhase,
    scenario_config,
)


def test_fixed_time_controller_waits_for_configured_green_duration() -> None:
    controller = FixedTimeController(green_duration=3)
    state = IntersectionEnvironment(
        config=EnvironmentConfig(minimum_green_time=0)
    ).state

    assert controller.action(state) is Action.KEEP_PHASE

    state = state.__class__(
        time_step=3,
        phase=SignalPhase.NS_GREEN,
        queues=(0, 0, 0, 0),
        seed=None,
        elapsed_green_time=3,
    )
    assert controller.action(state) is Action.SWITCH_PHASE


def test_random_controller_is_reproducible_and_keeps_transition_safe() -> None:
    controller = RandomController(seed=9)
    state = IntersectionEnvironment().state
    first = [controller.action(state) for _ in range(5)]
    controller.reset()
    second = [controller.action(state) for _ in range(5)]

    assert first == second
    transition_state = state.__class__(
        time_step=1,
        phase=SignalPhase.YELLOW,
        queues=(0, 0, 0, 0),
        seed=None,
    )
    assert controller.action(transition_state) is Action.KEEP_PHASE


def test_scenario_config_exposes_named_demand_profiles() -> None:
    config = scenario_config(DemandScenario.UNBALANCED)

    assert config.arrival_rates == (0.55, 0.20, 0.10, 0.10)


def test_baseline_episode_metrics_are_repeatable_with_matched_seed() -> None:
    config = scenario_config(DemandScenario.LOW, episode_horizon=10)
    first = run_episode(
        IntersectionEnvironment(config, seed=21),
        FixedTimeController(green_duration=5),
        seed=21,
    )
    second = run_episode(
        IntersectionEnvironment(config, seed=21),
        FixedTimeController(green_duration=5),
        seed=21,
    )

    assert first == second
    assert first.throughput >= 0
    assert first.max_queue_length >= 0
