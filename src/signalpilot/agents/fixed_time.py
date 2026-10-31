"""Fixed-time alternating signal controller."""

from signalpilot.environment import Action, IntersectionState, SignalPhase


class FixedTimeController:
    """Request a phase change after a fixed green duration."""

    def __init__(self, green_duration: int = 10) -> None:
        if green_duration <= 0:
            raise ValueError("green_duration must be positive")
        self.green_duration = green_duration

    def reset(self) -> None:
        """Reset the controller; it has no state beyond the environment."""

    def action(self, state: IntersectionState) -> Action:
        """Switch after the configured green duration, otherwise keep."""

        if state.phase not in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN):
            return Action.KEEP_PHASE
        return (
            Action.SWITCH_PHASE
            if state.elapsed_green_time >= self.green_duration
            else Action.KEEP_PHASE
        )
