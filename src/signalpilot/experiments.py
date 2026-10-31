"""Export helpers for reproducible experiment artifacts."""

import json
from dataclasses import asdict
from pathlib import Path

from signalpilot.training import EvaluationReport, TrainingHistory


def export_experiment(
    directory: str | Path,
    *,
    history: TrainingHistory | None = None,
    report: EvaluationReport | None = None,
    configuration: dict[str, object] | None = None,
) -> Path:
    """Write configuration, training history, and evaluation report as JSON."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    if configuration is not None:
        (output / "configuration.json").write_text(
            json.dumps(configuration, indent=2, default=str),
            encoding="utf-8",
        )
    if history is not None:
        (output / "training_history.json").write_text(
            json.dumps(asdict(history), indent=2),
            encoding="utf-8",
        )
    if report is not None:
        report.save(output / "evaluation_report.json")
    return output
