import matplotlib

matplotlib.use("Agg")

from signalpilot.agents import FixedTimeController
from signalpilot.environment import (
    DemandScenario,
    IntersectionEnvironment,
    SignalPhase,
    scenario_config,
)
from signalpilot.training import TrainingHistory
from signalpilot.visualization import (
    plot_controller_comparison,
    plot_intersection,
    plot_queue_lengths,
    plot_reward_curve,
    plot_signal_phase,
    plot_waiting_distribution,
    record_episode,
)


def test_recorded_episode_contains_full_trace() -> None:
    config = scenario_config(DemandScenario.LOW, episode_horizon=3)
    trace = record_episode(
        IntersectionEnvironment(config, seed=4),
        FixedTimeController(),
        seed=4,
    )

    assert len(trace.states) == 4
    assert len(trace.rewards) == 3
    assert trace.phases[0] is SignalPhase.NS_GREEN
    assert trace.metrics.throughput >= 0


def test_visualization_functions_return_figures() -> None:
    history = TrainingHistory(
        rewards=(-1.0, -2.0, -1.5),
        throughputs=(1, 2, 1),
        queue_lengths=(2.0, 1.0, 1.5),
        epsilons=(1.0, 0.9, 0.8),
    )
    trace = record_episode(
        IntersectionEnvironment(
            scenario_config(DemandScenario.LOW, episode_horizon=2),
            seed=5,
        ),
        FixedTimeController(),
        seed=5,
    )

    assert plot_reward_curve(history)
    assert plot_queue_lengths([state.queues for state in trace.states])
    assert plot_signal_phase(trace.phases)
    assert plot_waiting_distribution(trace.waiting_times)
    assert plot_controller_comparison(
        {"fixed": {"mean_waiting_time": 1.0}, "random": {"mean_waiting_time": 2.0}}
    )
    assert plot_intersection(trace.states[-1])
