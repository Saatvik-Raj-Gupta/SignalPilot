"""Named demand scenarios used for baseline comparisons."""

from enum import StrEnum

from signalpilot.environment.intersection import EnvironmentConfig


class DemandScenario(StrEnum):
    """MVP demand profiles, expressed as vehicles per second."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNBALANCED = "unbalanced"
    HELD_OUT = "held_out"


_RATES: dict[DemandScenario, tuple[float, float, float, float]] = {
    DemandScenario.LOW: (0.10, 0.10, 0.10, 0.10),
    DemandScenario.MEDIUM: (0.30, 0.30, 0.30, 0.30),
    DemandScenario.HIGH: (0.65, 0.65, 0.65, 0.65),
    DemandScenario.UNBALANCED: (0.55, 0.20, 0.10, 0.10),
    DemandScenario.HELD_OUT: (0.45, 0.15, 0.35, 0.05),
}


def scenario_config(
    scenario: DemandScenario | str,
    *,
    episode_horizon: int = 60,
    **overrides: object,
) -> EnvironmentConfig:
    """Build an environment config for a named demand profile."""

    selected = DemandScenario(scenario)
    return EnvironmentConfig(
        episode_horizon=episode_horizon,
        arrival_rates=_RATES[selected],
        **overrides,
    )
