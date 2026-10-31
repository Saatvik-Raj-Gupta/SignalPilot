"""Matplotlib plots for simulation and experiment outputs."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from signalpilot.environment import SignalPhase
from signalpilot.training.trainer import TrainingHistory


def _axis(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is None:
        figure, axis = plt.subplots()
        return figure, axis
    return ax.figure, ax


def plot_reward_curve(
    history: TrainingHistory,
    *,
    rolling_window: int = 10,
    ax: Axes | None = None,
) -> Figure:
    """Plot episode rewards and their rolling mean."""

    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")
    figure, axis = _axis(ax)
    rewards = np.asarray(history.rewards, dtype=float)
    axis.plot(rewards, alpha=0.35, label="episode reward")
    if rewards.size:
        window = min(rolling_window, rewards.size)
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axis.plot(
            np.arange(window - 1, rewards.size),
            rolling,
            label=f"rolling mean ({window})",
        )
    axis.set(title="Training reward", xlabel="Episode", ylabel="Reward")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_queue_lengths(
    queues: Sequence[Sequence[int]],
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot individual approach queues over time."""

    figure, axis = _axis(ax)
    values = np.asarray(queues)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("queues must have shape (time, 4)")
    for index, name in enumerate(("North", "South", "East", "West")):
        axis.plot(values[:, index], label=name)
    axis.set(title="Queues over time", xlabel="Time step", ylabel="Vehicles")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_signal_phase(
    phases: Sequence[SignalPhase],
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot signal states as a categorical timeline."""

    figure, axis = _axis(ax)
    labels = list(SignalPhase)
    codes = [labels.index(phase) for phase in phases]
    axis.step(range(len(codes)), codes, where="post")
    axis.set(
        title="Signal phase over time",
        xlabel="Time step",
        ylabel="Phase",
        yticks=range(len(labels)),
        yticklabels=[phase.value for phase in labels],
    )
    figure.tight_layout()
    return figure


def plot_waiting_distribution(
    waiting_times: Sequence[int],
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot the distribution of observed waiting-time values."""

    figure, axis = _axis(ax)
    axis.hist(waiting_times, bins="auto")
    axis.set(title="Waiting-time distribution", xlabel="Waiting time", ylabel="Count")
    figure.tight_layout()
    return figure


def plot_controller_comparison(
    summaries: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "mean_waiting_time",
    ax: Axes | None = None,
) -> Figure:
    """Compare controllers using one already-computed metric."""

    if not summaries:
        raise ValueError("summaries must not be empty")
    missing = [name for name, values in summaries.items() if metric not in values]
    if missing:
        raise ValueError(f"metric {metric!r} missing for: {', '.join(missing)}")
    figure, axis = _axis(ax)
    names = list(summaries)
    axis.bar(names, [summaries[name][metric] for name in names])
    axis.set(title=f"Controller comparison: {metric}", ylabel=metric)
    figure.tight_layout()
    return figure
