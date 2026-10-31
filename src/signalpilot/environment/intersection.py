"""Phase 1 environment contract for the SignalPilot intersection."""

from dataclasses import dataclass

import numpy as np

from signalpilot.environment.state import (
    Action,
    IntersectionState,
    SignalPhase,
    StepResult,
)
from signalpilot.utils.seeding import create_rng


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration needed by the minimal environment contract."""

    episode_horizon: int = 60
    initial_phase: SignalPhase = SignalPhase.NS_GREEN

    def __post_init__(self) -> None:
        if self.episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")


class IntersectionEnvironment:
    """A deterministic skeleton for one four-way intersection.

    Phase 1 models only time, phase selection, and the episode boundary. It
    establishes the contract that later traffic mechanics will implement:
    ``reset`` returns an initial state and ``step`` returns state, reward,
    termination flags, and diagnostic information.
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or EnvironmentConfig()
        self._seed = seed
        self._rng: np.random.Generator
        self._state: IntersectionState | None = None
        self.reset()

    @property
    def state(self) -> IntersectionState:
        """Return the current state, requiring reset before use."""

        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    def reset(self, *, seed: int | None = None) -> IntersectionState:
        """Start a new episode and return its initial state.

        Passing a seed makes the episode reproducible. Calling ``reset``
        without a seed reuses the environment's configured seed, which makes
        repeated seeded episodes identical.
        """

        if seed is not None:
            self._seed = seed
        self._rng = create_rng(self._seed)
        self._state = IntersectionState(
            time_step=0,
            phase=self.config.initial_phase,
            queues=(0, 0, 0, 0),
            seed=self._seed,
        )
        return self._state

    def step(self, action: Action | int) -> StepResult:
        """Apply one action and advance the episode by one second.

        Phase 1 has no arrivals or discharge yet. ``SWITCH_PHASE`` therefore
        toggles the active phase immediately; safe transition states and
        minimum-green enforcement are added in Phase 2.
        """

        if self._state is None:
            raise RuntimeError("call reset before step")
        if self._state.time_step >= self.config.episode_horizon:
            raise RuntimeError("cannot step a terminated episode; call reset")

        try:
            selected_action = Action(action)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported action: {action!r}") from error

        phase = self._state.phase
        switched = selected_action is Action.SWITCH_PHASE
        if switched:
            phase = (
                SignalPhase.EW_GREEN
                if phase is SignalPhase.NS_GREEN
                else SignalPhase.NS_GREEN
            )

        next_state = IntersectionState(
            time_step=self._state.time_step + 1,
            phase=phase,
            queues=self._state.queues,
            seed=self._state.seed,
        )
        self._state = next_state
        return StepResult(
            state=next_state,
            reward=0.0,
            terminated=next_state.time_step >= self.config.episode_horizon,
            truncated=False,
            info={"action": selected_action, "phase_switched": switched},
        )
