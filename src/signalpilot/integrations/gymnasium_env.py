"""Optional Gymnasium-compatible wrapper around the native environment."""

from signalpilot.environment import Action, EnvironmentConfig, IntersectionEnvironment


class GymnasiumIntersectionEnv:
    """Adapter using the Gymnasium API without coupling the core simulator."""

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        try:
            from gymnasium import spaces
        except ImportError as error:
            raise RuntimeError(
                "Gymnasium adapter requires optional Gymnasium; install "
                'with `python -m pip install -e ".[gymnasium]"`'
            ) from error
        self._spaces = spaces
        self.environment = IntersectionEnvironment(config, seed=seed)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiDiscrete([4, 4, 4, 4])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        state = self.environment.reset(seed=seed)
        return self._observation(state), {"seed": state.seed}

    def step(self, action: int):
        result = self.environment.step(Action(action))
        return (
            self._observation(result.state),
            result.reward,
            result.terminated,
            result.truncated,
            dict(result.info),
        )

    @staticmethod
    def _observation(state):
        return tuple(min(queue, 3) for queue in state.queues)
