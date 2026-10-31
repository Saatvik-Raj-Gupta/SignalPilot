"""Multi-seed and multi-scenario controller evaluation."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from signalpilot.agents.base import Controller, run_episode
from signalpilot.environment import (
    DemandScenario,
    EnvironmentConfig,
    IntersectionEnvironment,
)
from signalpilot.metrics import EpisodeMetrics


@dataclass(frozen=True)
class MetricSummary:
    """Mean and population standard deviation for one metric."""

    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results grouped by demand scenario."""

    scenarios: dict[str, dict[str, MetricSummary]]

    def save(self, path: str | Path) -> None:
        """Save the report as readable JSON."""

        payload = {
            scenario: {
                metric: asdict(summary)
                for metric, summary in metrics.items()
            }
            for scenario, metrics in self.scenarios.items()
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_controller(
    controller_factory: Callable[[], Controller],
    config_factory: Callable[[DemandScenario], EnvironmentConfig],
    *,
    scenarios: tuple[DemandScenario, ...] = tuple(DemandScenario),
    seeds: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004),
) -> EvaluationReport:
    """Evaluate a controller on matched seeds across demand scenarios."""

    if not seeds:
        raise ValueError("seeds must not be empty")
    results: dict[str, dict[str, MetricSummary]] = {}
    metric_names = tuple(EpisodeMetrics.__dataclass_fields__)

    for scenario in scenarios:
        values: dict[str, list[float]] = {name: [] for name in metric_names}
        for seed in seeds:
            controller = controller_factory()
            if hasattr(controller, "epsilon"):
                controller.epsilon = 0.0
            metrics = run_episode(
                IntersectionEnvironment(config_factory(scenario), seed=seed),
                controller,
                seed=seed,
            )
            for name in metric_names:
                values[name].append(float(getattr(metrics, name)))
        results[scenario.value] = {
            name: MetricSummary(
                mean=float(np.mean(metric_values)),
                standard_deviation=float(np.std(metric_values)),
            )
            for name, metric_values in values.items()
        }
    return EvaluationReport(scenarios=results)


def compare_controllers(
    controller_factories: dict[str, Callable[[], Controller]],
    config_factory: Callable[[DemandScenario], EnvironmentConfig],
    *,
    scenarios: tuple[DemandScenario, ...] = tuple(DemandScenario),
    seeds: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004),
) -> dict[str, EvaluationReport]:
    """Evaluate multiple controllers on identical scenario/seed combinations."""

    return {
        name: evaluate_controller(
            factory,
            config_factory,
            scenarios=scenarios,
            seeds=seeds,
        )
        for name, factory in controller_factories.items()
    }
