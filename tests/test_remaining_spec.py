from signalpilot.agents import FixedTimeController, RandomController
from signalpilot.config import load_environment_config
from signalpilot.environment import (
    DemandScenario,
    EnvironmentConfig,
    scenario_config,
)
from signalpilot.experiments import export_experiment
from signalpilot.training import compare_controllers


def test_yaml_config_and_experiment_export(tmp_path) -> None:
    config = load_environment_config("config/default.yaml")
    assert isinstance(config, EnvironmentConfig)
    output = export_experiment(
        tmp_path,
        configuration={"seed": 3},
    )
    assert (output / "configuration.json").exists()


def test_controller_comparison_uses_shared_scenarios_and_seeds() -> None:
    reports = compare_controllers(
        {"fixed": FixedTimeController, "random": RandomController},
        lambda scenario: scenario_config(scenario, episode_horizon=2),
        scenarios=(DemandScenario.LOW,),
        seeds=(1, 2),
    )
    assert set(reports) == {"fixed", "random"}
    assert "low" in reports["fixed"].scenarios
