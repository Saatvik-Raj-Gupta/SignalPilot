"""Small, transparent tabular Q-Learning controller."""

import json
from pathlib import Path

from signalpilot.agents.state_encoding import StateEncoder
from signalpilot.environment import Action, IntersectionState, SignalPhase
from signalpilot.utils.seeding import create_rng


class QLearningController:
    """Epsilon-greedy Q-Learning controller with a dictionary Q-table."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        minimum_epsilon: float = 0.05,
        seed: int | None = None,
        encoder: StateEncoder | None = None,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0 <= epsilon <= 1 or not 0 <= minimum_epsilon <= 1:
            raise ValueError("epsilon values must be in [0, 1]")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self._seed = seed
        self._rng = create_rng(seed)
        self.encoder = encoder or StateEncoder()
        self.q_table: dict[tuple[int, int, int, int], list[float]] = {}

    def reset(self) -> None:
        """Reset episode-local randomness while preserving learned values."""

        self._rng = create_rng(self._seed)

    def action(self, state: IntersectionState) -> Action:
        """Choose an action using epsilon-greedy exploration."""

        if state.phase not in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            return Action.KEEP_PHASE
        values = self._values(self.encoder.encode(state))
        if self._rng.random() < self.epsilon:
            return Action(int(self._rng.integers(0, 2)))
        return Action(self._best_action(values))

    def update(
        self,
        state: IntersectionState,
        action: Action | int,
        reward: float,
        next_state: IntersectionState,
        *,
        terminated: bool = False,
    ) -> float:
        """Apply one Bellman Q-Learning update and return the new value."""

        selected_action = Action(action)
        current = self._values(self.encoder.encode(state))
        next_values = self._values(self.encoder.encode(next_state))
        target = float(reward)
        if not terminated:
            target += self.discount_factor * max(next_values)
        current[selected_action.value] += self.learning_rate * (
            target - current[selected_action.value]
        )
        return current[selected_action.value]

    def end_episode(self) -> None:
        """Decay exploration after one training episode."""

        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path: str | Path) -> None:
        """Save Q-values and hyperparameter metadata as JSON."""

        payload = {
            "metadata": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "minimum_epsilon": self.minimum_epsilon,
                "seed": self._seed,
                "queue_boundaries": list(self.encoder.queue_boundaries),
                "elapsed_green_boundaries": list(
                    self.encoder.elapsed_green_boundaries
                ),
            },
            "q_table": [
                {"state": list(state), "values": values}
                for state, values in self.q_table.items()
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningController":
        """Load a controller from the JSON format produced by ``save``."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        controller = cls(
            learning_rate=metadata["learning_rate"],
            discount_factor=metadata["discount_factor"],
            epsilon=metadata["epsilon"],
            epsilon_decay=metadata["epsilon_decay"],
            minimum_epsilon=metadata["minimum_epsilon"],
            seed=metadata["seed"],
            encoder=StateEncoder(
                tuple(metadata["queue_boundaries"]),
                tuple(metadata["elapsed_green_boundaries"]),
            ),
        )
        controller.q_table = {
            tuple(item["state"]): [float(value) for value in item["values"]]
            for item in payload["q_table"]
        }
        return controller

    def _values(self, state: tuple[int, int, int, int]) -> list[float]:
        return self.q_table.setdefault(state, [0.0, 0.0])

    @staticmethod
    def _best_action(values: list[float]) -> int:
        return max(range(len(values)), key=lambda index: (values[index], -index))
