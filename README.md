# SignalPilot

SignalPilot is an educational reinforcement-learning traffic signal
simulation. The project starts with a small, explainable four-way intersection
before adding traffic dynamics and learning.

## Development setup

The project is developed with Python 3.11 or newer. With the Miniconda
environment created for this repository:

```bash
conda activate signalpilot
python -m pip install -e ".[dev]"
```

Run the Phase 1 tests with:

```bash
pytest
```

## Current scope

Phase 1 provides a typed environment contract with deterministic reset and
step behavior. It models the active North-South or East-West phase and the
episode horizon, but does not yet include vehicle arrivals, queue discharge,
safe yellow/all-red transitions, rewards, controllers, or Q-learning. Those
features are intentionally staged for later implementation phases.
