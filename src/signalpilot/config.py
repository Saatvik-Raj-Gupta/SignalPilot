"""Configuration loading helpers."""

from pathlib import Path

from signalpilot.environment import EnvironmentConfig, SignalPhase


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """Load an environment configuration from YAML."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML loading requires PyYAML; install the project requirements."
        ) from error
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EnvironmentConfig(
        episode_horizon=data["episode_horizon"],
        initial_phase=SignalPhase(data["initial_phase"]),
        arrival_rates=tuple(data["arrival_rates"][name] for name in (
            "north", "south", "east", "west"
        )),
        saturation_flow=data["saturation_flow"],
        minimum_green_time=data["minimum_green_time"],
        yellow_duration=data["yellow_duration"],
        all_red_duration=data["all_red_duration"],
        initial_queues=tuple(data["initial_queues"][name] for name in (
            "north", "south", "east", "west"
        )),
    )
