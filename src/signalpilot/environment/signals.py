"""Signal transition helpers shared by the intersection environment."""

from signalpilot.environment.state import SignalPhase


def is_green(phase: SignalPhase) -> bool:
    """Return whether a phase permits vehicle discharge."""

    return phase in (SignalPhase.NS_GREEN, SignalPhase.EW_GREEN)


def opposing_green(phase: SignalPhase) -> SignalPhase:
    """Return the green phase opposing a current green phase."""

    if phase is SignalPhase.NS_GREEN:
        return SignalPhase.EW_GREEN
    if phase is SignalPhase.EW_GREEN:
        return SignalPhase.NS_GREEN
    raise ValueError("opposing_green requires a green phase")
