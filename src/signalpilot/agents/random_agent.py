"""Random valid-action baseline controller."""

from signalpilot.environment import Action, IntersectionState, SignalPhase
from signalpilot.utils.seeding import create_rng


class RandomController:
    """Randomly choose keep or switch while respecting transition states."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = create_rng(seed)

    def reset(self) -> None:
        """Reset the random stream so repeated seeded episodes match."""

        self._rng = create_rng(self._seed)

    def action(self, state: IntersectionState) -> Action:
        """Choose a random action only when the signal is green."""

        if state.phase not in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            return Action.KEEP_PHASE
        return Action(int(self._rng.integers(0, 2)))
