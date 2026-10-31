from signalpilot.agents import FixedTimeController, QLearningController
from signalpilot.environment import DemandScenario, scenario_config
from signalpilot.environment.intersection import IntersectionEnvironment
from signalpilot.training import evaluate_controller, train_q_learning


def test_training_history_has_one_entry_per_episode() -> None:
    config = scenario_config(DemandScenario.LOW, episode_horizon=4)
    controller = QLearningController(seed=2)

    history = train_q_learning(
        controller,
        IntersectionEnvironment(config, seed=10),
        episodes=3,
        seed=10,
    )

    assert len(history.rewards) == 3
    assert len(history.throughputs) == 3
    assert len(history.queue_lengths) == 3
    assert len(history.epsilons) == 3
    assert len(controller.q_table) > 0


def test_training_is_repeatable_with_same_seed() -> None:
    config = scenario_config(DemandScenario.LOW, episode_horizon=4)

    def run() -> tuple[float, ...]:
        controller = QLearningController(seed=2)
        return train_q_learning(
            controller,
            IntersectionEnvironment(config, seed=10),
            episodes=3,
            seed=10,
        ).rewards

    assert run() == run()


def test_evaluation_reports_mean_and_variability_and_can_save(tmp_path) -> None:
    report = evaluate_controller(
        FixedTimeController,
        lambda scenario: scenario_config(scenario, episode_horizon=3),
        scenarios=(DemandScenario.LOW,),
        seeds=(10, 11),
    )
    path = tmp_path / "evaluation.json"
    report.save(path)

    low = report.scenarios[DemandScenario.LOW.value]
    assert low["throughput"].mean >= 0
    assert low["throughput"].standard_deviation >= 0
    assert path.exists()
