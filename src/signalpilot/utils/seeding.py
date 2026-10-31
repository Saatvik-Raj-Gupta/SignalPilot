"""Explicit random-number handling for reproducible simulations."""

import numpy as np


def create_rng(seed: int | None = None) -> np.random.Generator:
    """Create an independent NumPy generator without changing global state."""

    return np.random.default_rng(seed)
