"""Top-down rendering of one intersection state."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from signalpilot.environment import IntersectionState, SignalPhase


def plot_intersection(
    state: IntersectionState,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Render roads, signal lamps, queue markers, and state metadata."""

    if ax is None:
        figure, axis = plt.subplots(figsize=(7, 7))
    else:
        figure, axis = ax.figure, ax
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.set_aspect("equal")
    axis.axis("off")

    axis.add_patch(Rectangle((-1.4, -5), 2.8, 10, color="#555555"))
    axis.add_patch(Rectangle((-5, -1.4), 10, 2.8, color="#555555"))
    axis.add_patch(Rectangle((-1.4, -1.4), 2.8, 2.8, color="#333333"))

    positions = [(-2.2, 3.8), (-1.0, 3.8), (-3.8, 2.2), (-3.8, 1.0)]
    labels = ("N", "S", "E", "W")
    for (x, y), label, queue in zip(positions, labels, state.queues, strict=True):
        axis.text(x, y, f"{label}: {queue}", ha="center", va="center", color="white")

    ns_color = "#35c759" if state.phase is SignalPhase.NS_GREEN else "#d62728"
    ew_color = "#35c759" if state.phase is SignalPhase.EW_GREEN else "#d62728"
    axis.scatter([-2.0, 2.0], [0, 0], c=ew_color, s=180, edgecolors="black")
    axis.scatter([0, 0], [-2.0, 2.0], c=ns_color, s=180, edgecolors="black")
    axis.set_title(
        f"SignalPilot | phase: {state.phase.value} | "
        f"time: {state.time_step} | total queue: {sum(state.queues)}"
    )
    figure.tight_layout()
    return figure
