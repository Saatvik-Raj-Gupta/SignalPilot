from signalpilot.agents import QLearningController, StateEncoder
from signalpilot.environment import (
    Action,
    EnvironmentConfig,
    IntersectionEnvironment,
    IntersectionState,
    SignalPhase,
)


def test_state_encoding_is_bounded_and_deterministic() -> None:
    encoder = StateEncoder()
    state = IntersectionState(
        time_step=4,
        phase=SignalPhase.EW_GREEN,
        queues=(10, 2, 4, 6),
        seed=1,
        elapsed_green_time=12,
    )

    assert encoder.encode(state) == (3, 3, 1, 3)
    assert encoder.encode(state) == encoder.encode(state)


def test_q_update_matches_hand_calculated_example() -> None:
    controller = QLearningController(
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.0,
    )
    state = IntersectionEnvironment().state
    next_state = IntersectionState(
        time_step=1,
        phase=SignalPhase.NS_GREEN,
        queues=(1, 0, 0, 0),
        seed=None,
        elapsed_green_time=1,
    )
    controller.q_table[controller.encoder.encode(next_state)] = [4.0, 2.0]

    updated = controller.update(
        state,
        Action.KEEP_PHASE,
        reward=2.0,
        next_state=next_state,
    )

    assert updated == 2.8


def test_terminal_update_does_not_bootstrap() -> None:
    controller = QLearningController(
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.0,
    )
    state = IntersectionEnvironment().state
    next_state = IntersectionState(
        time_step=1,
        phase=SignalPhase.EW_GREEN,
        queues=(8, 0, 0, 0),
        seed=None,
        elapsed_green_time=1,
    )
    controller.q_table[controller.encoder.encode(next_state)] = [100.0, 100.0]

    updated = controller.update(
        state,
        Action.KEEP_PHASE,
        reward=-4.0,
        next_state=next_state,
        terminated=True,
    )

    assert updated == -2.0


def test_epsilon_greedy_tie_breaks_to_keep_and_decay_has_floor() -> None:
    controller = QLearningController(
        epsilon=0.3,
        epsilon_decay=0.1,
        minimum_epsilon=0.2,
    )
    state = IntersectionEnvironment().state

    assert controller.action(state) is Action.KEEP_PHASE
    controller.end_episode()
    assert controller.epsilon == 0.2


def test_saved_and_loaded_table_has_same_greedy_policy(tmp_path) -> None:
    controller = QLearningController(epsilon=0.0, seed=4)
    state = IntersectionEnvironment(
        EnvironmentConfig(minimum_green_time=0)
    ).state
    controller.q_table[controller.encoder.encode(state)] = [1.0, 3.0]
    path = tmp_path / "q_table.json"

    controller.save(path)
    loaded = QLearningController.load(path)

    assert loaded.action(state) is Action.SWITCH_PHASE
    assert loaded.q_table == controller.q_table
