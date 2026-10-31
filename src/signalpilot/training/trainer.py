"""Training loop for the tabular Q-Learning controller."""

from dataclasses import dataclass

from signalpilot.agents.q_learning import QLearningController
from signalpilot.environment import IntersectionEnvironment


@dataclass(frozen=True)
class TrainingHistory:
    """Episode-level values recorded during training."""

    rewards: tuple[float, ...]
    throughputs: tuple[int, ...]
    queue_lengths: tuple[float, ...]
    epsilons: tuple[float, ...]


def train_q_learning(
    controller: QLearningController,
    environment: IntersectionEnvironment,
    *,
    episodes: int,
    seed: int | None = None,
) -> TrainingHistory:
    """Train one controller for a fixed number of seeded episodes.

    The seed is used as the first episode seed and incremented for subsequent
    episodes, keeping training randomness separate from evaluation seeds.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rewards: list[float] = []
    throughputs: list[int] = []
    queue_lengths: list[float] = []
    epsilons: list[float] = []

    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        state = environment.reset(seed=episode_seed)
        controller.reset()
        episode_reward = 0.0
        episode_queue_total = 0

        while True:
            action = controller.action(state)
            result = environment.step(action)
            controller.update(
                state,
                action,
                result.reward,
                result.state,
                terminated=result.terminated or result.truncated,
            )
            episode_reward += result.reward
            episode_queue_total += sum(result.state.queues)
            state = result.state
            if result.terminated or result.truncated:
                break

        rewards.append(episode_reward)
        throughputs.append(state.total_served)
        queue_lengths.append(
            episode_queue_total / environment.config.episode_horizon
        )
        controller.end_episode()
        epsilons.append(controller.epsilon)

    return TrainingHistory(
        rewards=tuple(rewards),
        throughputs=tuple(throughputs),
        queue_lengths=tuple(queue_lengths),
        epsilons=tuple(epsilons),
    )
