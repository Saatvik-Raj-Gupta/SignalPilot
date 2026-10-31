from signalpilot.environment import (
    Action,
    EnvironmentConfig,
    IntersectionEnvironment,
    SignalPhase,
)


def test_reset_returns_valid_initial_state() -> None:
    environment = IntersectionEnvironment(seed=7)

    state = environment.reset()

    assert state.time_step == 0
    assert state.phase is SignalPhase.NS_GREEN
    assert state.queues == (0, 0, 0, 0)
    assert state.seed == 7


def test_step_returns_contract_and_switches_phase() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(minimum_green_time=0),
        seed=7,
    )

    result = environment.step(Action.SWITCH_PHASE)

    assert result.state.time_step == 1
    assert result.state.phase is SignalPhase.YELLOW
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["phase_switched"] is True


def test_same_seed_and_actions_reproduce_transitions() -> None:
    actions = [Action.KEEP_PHASE, Action.SWITCH_PHASE, Action.KEEP_PHASE]
    config = EnvironmentConfig(episode_horizon=5)
    first = IntersectionEnvironment(seed=123, config=config)
    second = IntersectionEnvironment(seed=123, config=config)

    first_transitions = [first.step(action) for action in actions]
    second_transitions = [second.step(action) for action in actions]

    assert first_transitions == second_transitions


def test_horizon_terminates_episode() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(episode_horizon=2),
        seed=3,
    )

    environment.step(Action.KEEP_PHASE)
    result = environment.step(Action.KEEP_PHASE)

    assert result.terminated is True


def test_switch_request_before_minimum_green_is_ignored() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(minimum_green_time=2),
        seed=3,
    )

    result = environment.step(Action.SWITCH_PHASE)

    assert result.state.phase is SignalPhase.NS_GREEN
    assert result.info["phase_switched"] is False


def test_signal_transition_uses_yellow_then_all_red() -> None:
    environment = IntersectionEnvironment(
        config=EnvironmentConfig(
            minimum_green_time=0,
            yellow_duration=2,
            all_red_duration=1,
        ),
        seed=3,
    )

    phases = [
        environment.step(Action.SWITCH_PHASE).state.phase,
        environment.step(Action.KEEP_PHASE).state.phase,
        environment.step(Action.KEEP_PHASE).state.phase,
        environment.step(Action.KEEP_PHASE).state.phase,
    ]

    assert phases == [
        SignalPhase.YELLOW,
        SignalPhase.YELLOW,
        SignalPhase.ALL_RED,
        SignalPhase.EW_GREEN,
    ]
