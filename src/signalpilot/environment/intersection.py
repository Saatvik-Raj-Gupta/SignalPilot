"""Four-way intersection environment with seeded traffic mechanics."""

from dataclasses import dataclass

import numpy as np

from signalpilot.environment.signals import opposing_green
from signalpilot.environment.state import (
    Action,
    Approach,
    IntersectionState,
    SignalPhase,
    StepResult,
)
from signalpilot.environment.vehicle_flow import discharge_queues, sample_arrivals
from signalpilot.utils.seeding import create_rng


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configurable traffic and signal mechanics for one intersection."""

    episode_horizon: int = 60
    initial_phase: SignalPhase = SignalPhase.NS_GREEN
    arrival_rates: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    saturation_flow: int = 1
    minimum_green_time: int = 5
    yellow_duration: int = 2
    all_red_duration: int = 1
    initial_queues: tuple[int, int, int, int] = (0, 0, 0, 0)
    reward_alpha: float = 1.0
    reward_beta: float = 0.0
    reward_switch_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")
        if self.initial_phase not in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            raise ValueError("initial_phase must be a green phase")
        if len(self.arrival_rates) != len(Approach):
            raise ValueError("arrival_rates must contain one value per approach")
        if any(rate < 0 for rate in self.arrival_rates):
            raise ValueError("arrival rates must be non-negative")
        if len(self.initial_queues) != len(Approach):
            raise ValueError("initial_queues must contain one value per approach")
        if any(queue < 0 for queue in self.initial_queues):
            raise ValueError("initial queues must be non-negative")
        if self.saturation_flow <= 0:
            raise ValueError("saturation_flow must be positive")
        if self.minimum_green_time < 0:
            raise ValueError("minimum_green_time must be non-negative")
        if self.yellow_duration <= 0 or self.all_red_duration <= 0:
            raise ValueError("transition durations must be positive")
        if (
            self.reward_alpha < 0
            or self.reward_beta < 0
            or self.reward_switch_cost < 0
        ):
            raise ValueError("reward weights must be non-negative")


class IntersectionEnvironment:
    """A seeded, discrete-time simulator for one four-way intersection."""

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
        self._vehicle_waits: list[list[int]] = []
        self._last_served_waits: tuple[int, ...] = ()
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
        self._vehicle_waits = [
            [0] * queue for queue in self.config.initial_queues
        ]
        self._last_served_waits = ()
        self._state = IntersectionState(
            time_step=0,
            phase=self.config.initial_phase,
            queues=self.config.initial_queues,
            seed=self._seed,
        )
        return self._state

    def step(self, action: Action | int) -> StepResult:
        """Apply one action and advance the simulation by one second.

        Each step updates transitions, samples and adds arrivals, discharges
        eligible vehicles, accounts for vehicles still waiting, and records
        cumulative counts in the returned state and info mapping.
        """

        if self._state is None:
            raise RuntimeError("call reset before step")
        if self._state.time_step >= self.config.episode_horizon:
            raise RuntimeError("cannot step a terminated episode; call reset")

        try:
            selected_action = Action(action)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported action: {action!r}") from error

        previous_state = self.state
        phase = self._advance_signal(selected_action)
        phase_switched = (
            previous_state.phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN)
            and phase is SignalPhase.YELLOW
        )
        arrivals = sample_arrivals(self._rng, self.config.arrival_rates)
        for index, count in enumerate(arrivals):
            self._vehicle_waits[index].extend([0] * count)
        queues = tuple(
            queue + arrival
            for queue, arrival in zip(self._state.queues, arrivals, strict=True)
        )
        served = discharge_queues(phase, queues, self.config.saturation_flow)
        served_waits: list[int] = []
        for index, count in enumerate(served):
            if count:
                served_waits.extend(self._vehicle_waits[index][:count])
                del self._vehicle_waits[index][:count]
        for waits in self._vehicle_waits:
            for vehicle_index in range(len(waits)):
                waits[vehicle_index] += 1
        self._last_served_waits = tuple(served_waits)
        queues = tuple(
            max(queue - served_count, 0)
            for queue, served_count in zip(queues, served, strict=True)
        )
        waiting_increment = sum(queues)
        total_arrivals = self._state.total_arrivals + sum(arrivals)
        total_served = self._state.total_served + sum(served)

        next_state = IntersectionState(
            time_step=self._state.time_step + 1,
            phase=phase,
            queues=queues,
            seed=self._state.seed,
            elapsed_green_time=(
                previous_state.elapsed_green_time + 1
                if phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN)
                and previous_state.phase is phase
                else 1 if phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN) else 0
            ),
            transition_remaining=self._transition_remaining_after_step(phase),
            pending_phase=self._pending_phase_after_step(phase),
            total_arrivals=total_arrivals,
            total_served=total_served,
            total_waiting_time=self._state.total_waiting_time + waiting_increment,
        )
        self._state = next_state
        reward = (
            -self.config.reward_alpha * sum(queues)
            - self.config.reward_beta * waiting_increment
            - self.config.reward_switch_cost * int(phase_switched)
        )
        return StepResult(
            state=next_state,
            reward=reward,
            terminated=next_state.time_step >= self.config.episode_horizon,
            truncated=False,
            info={
                "action": selected_action,
                "arrivals": arrivals,
                "served": served,
                "waiting_increment": waiting_increment,
                "served_waiting_times": self._last_served_waits,
                "phase_switched": phase_switched,
            },
        )

    def _advance_signal(self, action: Action) -> SignalPhase:
        state = self.state
        if state.phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            if (
                action is Action.SWITCH_PHASE
                and state.elapsed_green_time >= self.config.minimum_green_time
            ):
                return SignalPhase.YELLOW
            return state.phase
        if state.phase is SignalPhase.YELLOW:
            return (
                SignalPhase.ALL_RED
                if state.transition_remaining <= 1
                else SignalPhase.YELLOW
            )
        return state.pending_phase or SignalPhase.NS_GREEN

    def _transition_remaining_after_step(self, phase: SignalPhase) -> int:
        if phase is SignalPhase.YELLOW:
            if self.state.phase is SignalPhase.YELLOW:
                return max(self.state.transition_remaining - 1, 0)
            return self.config.yellow_duration
        if phase is SignalPhase.ALL_RED:
            if self.state.phase is SignalPhase.ALL_RED:
                return max(self.state.transition_remaining - 1, 0)
            return self.config.all_red_duration
        return 0

    def _pending_phase_after_step(
        self,
        phase: SignalPhase,
    ) -> SignalPhase | None:
        if phase is SignalPhase.YELLOW:
            if self.state.phase is SignalPhase.YELLOW:
                return self.state.pending_phase
            return opposing_green(self.state.phase)
        if phase is SignalPhase.ALL_RED:
            return self.state.pending_phase
        return None
