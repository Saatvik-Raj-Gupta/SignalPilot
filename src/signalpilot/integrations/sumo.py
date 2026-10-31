"""Optional SUMO dependency guard.

The educational MVP remains independent of SUMO. This helper gives callers a
clear installation error before attempting to build an external adapter.
"""


class SumoUnavailableError(RuntimeError):
    """Raised when SUMO's TraCI Python client is unavailable."""


def require_sumo() -> None:
    """Require the optional TraCI dependency for a future SUMO adapter."""

    try:
        import traci  # noqa: F401
    except ImportError as error:
        raise SumoUnavailableError(
            "SUMO integration requires TraCI and a local SUMO installation; "
            "install the optional Python dependency and configure SUMO_HOME."
        ) from error
