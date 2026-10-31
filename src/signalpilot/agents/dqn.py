"""Optional PyTorch DQN controller.

The MVP remains tabular. This module imports PyTorch only when the controller
is constructed, so users can use the simulator without installing PyTorch.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from signalpilot.agents.state_encoding import StateEncoder
from signalpilot.environment import Action, IntersectionState, SignalPhase


@dataclass(frozen=True)
class Transition:
    """One replay-buffer transition."""

    state: tuple[int, int, int, int]
    action: int
    reward: float
    next_state: tuple[int, int, int, int]
    terminated: bool


class ReplayBuffer:
    """Bounded replay memory independent of PyTorch."""

    def __init__(self, capacity: int = 10_000, *, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, size: int) -> list[Transition]:
        if not 0 < size <= len(self._items):
            raise ValueError("sample size must be positive and fit in the buffer")
        return self._random.sample(list(self._items), size)

    def __len__(self) -> int:
        return len(self._items)


class DQNController:
    """Minimal replay-buffer DQN controller for the optional later phase."""

    def __init__(
        self,
        *,
        encoder: StateEncoder | None = None,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        minimum_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        hidden_size: int = 64,
        replay_capacity: int = 10_000,
        batch_size: int = 32,
        target_update_interval: int = 100,
        seed: int | None = None,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError(
                "DQNController requires optional PyTorch; install "
                'with `python -m pip install -e ".[dqn]"`'
            ) from error
        if hidden_size <= 0 or batch_size <= 0:
            raise ValueError("hidden_size and batch_size must be positive")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if not 0 <= minimum_epsilon <= epsilon <= 1:
            raise ValueError("epsilon values must satisfy 0 <= minimum <= epsilon <= 1")
        self._torch = torch
        self._seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder or StateEncoder()
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self._updates = 0
        self._random = np.random.default_rng(seed)
        self.replay = ReplayBuffer(replay_capacity, seed=seed)
        self.network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        ).to(self.device)
        self.target_network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss()

    def reset(self) -> None:
        """Reset episode-local exploration state."""

    def action(self, state: IntersectionState) -> Action:
        """Choose an epsilon-greedy action using the online network."""

        if state.phase not in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            return Action.KEEP_PHASE
        if self._random.random() < self.epsilon:
            return Action(int(self._random.integers(0, 2)))
        with self._torch.no_grad():
            values = self.network(self._tensor(self.encoder.encode(state)))
        return Action(int(values.argmax().item()))

    def update(
        self,
        state: IntersectionState,
        action: Action | int,
        reward: float,
        next_state: IntersectionState,
        *,
        terminated: bool = False,
    ) -> float | None:
        """Store a transition and train when enough replay data exists."""

        self.replay.append(
            Transition(
                self.encoder.encode(state),
                Action(action).value,
                float(reward),
                self.encoder.encode(next_state),
                terminated,
            )
        )
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = self._torch.cat([self._tensor(item.state) for item in batch])
        next_states = self._torch.cat(
            [self._tensor(item.next_state) for item in batch]
        )
        actions = self._torch.tensor(
            [item.action for item in batch],
            dtype=self._torch.long,
            device=self.device,
        )
        rewards = self._torch.tensor(
            [item.reward for item in batch],
            dtype=self._torch.float32,
            device=self.device,
        )
        terminals = self._torch.tensor(
            [item.terminated for item in batch],
            dtype=self._torch.bool,
            device=self.device,
        )
        current = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with self._torch.no_grad():
            future = self.target_network(next_states).max(dim=1).values
            target = rewards + self.discount_factor * future * (~terminals)
        value = self.loss(current, target)
        self.optimizer.zero_grad()
        value.backward()
        self.optimizer.step()
        self._updates += 1
        if self._updates % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return float(value.item())

    def end_episode(self) -> None:
        """Decay exploration after one training episode."""

        self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay)

    def _tensor(self, state: tuple[int, int, int, int]) -> Any:
        return self._torch.tensor(
            [state],
            dtype=self._torch.float32,
            device=self.device,
        )
