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

Alternatively, install the pinned project requirement groups directly:

```bash
python -m pip install -r requirements.txt
```

## Run tests on CPU

The simulator, tabular Q-Learning implementation, baselines, training code,
plots, and Streamlit support are CPU-compatible. Run the complete test suite
and lint checks with:

```bash
python -m pytest
python -m ruff check .
```

To run a focused test file:

```bash
python -m pytest tests/test_environment.py
```

Training is not started by the test suite. When you are ready to train the
tabular agent, use the existing `train_q_learning` function from
`signalpilot.training`; it is CPU-oriented and does not require PyTorch.

## Optional GPU/DQN setup

GPU acceleration is relevant only to the optional PyTorch DQN controller.
Tabular Q-Learning uses Python dictionaries and scalar updates, so a GPU will
not make it faster. Install the optional DQN dependency:

```bash
conda activate signalpilot
python -m pip install -e ".[dqn]"
```

Verify that PyTorch can see CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The current DQN controller is intentionally optional and automatically uses
CUDA when `torch.cuda.is_available()` is true; otherwise it uses the CPU. A
CUDA-enabled PyTorch installation is required for GPU execution; installing
the `dqn` extra alone does not guarantee that the machine has a compatible
NVIDIA driver or CUDA runtime.
Run the normal tests after installing DQN as well:

```bash
python -m pytest
python -m ruff check .
```

The tests do not start DQN training or any long-running experiment.

## Streamlit application

Install the optional UI dependency and open the batch-run simulator:

```bash
conda activate signalpilot
python -m pip install -e ".[ui]"
python -m streamlit run app.py
```

This opens the local Streamlit address shown in the terminal, normally
`http://localhost:8501`. The UI runs fixed-time, random, and tabular Q-Learning episodes, or compares
the three controllers on matched seeds. It renders the final intersection
state, queue/phase/waiting plots, and metrics. It reuses the same environment
and controllers as the Python API.

## Current scope

Phase 2 now includes seeded Poisson arrivals, four approach queues, saturation
flow discharge, waiting-time accounting, and safe yellow/all-red transitions.
Phase 3 adds fixed-time and random baseline controllers, named low/medium/high,
unbalanced, and held-out demand scenarios, plus repeatable episode metrics.
Phase 4 adds compact state encoding and transparent tabular Q-learning with
epsilon-greedy exploration, terminal-aware updates, and JSON persistence.
Phase 5 adds a separated training loop and multi-seed, multi-scenario
evaluation report generation. Training is intentionally not run automatically.
The visualization layer now provides episode traces, reward/queue/phase/waiting
plots, controller comparisons, and a top-down intersection view. A batch
Streamlit interface is available through `app.py`.

Optional integrations are now available behind extras:

```bash
python -m pip install -e ".[dqn]"
python -m pip install -e ".[gymnasium]"
python -m pip install -e ".[sumo]"
```

The DQN implementation uses PyTorch and can use a CUDA device after PyTorch
installation, but it is intentionally not imported by the core simulator.
The Gymnasium adapter wraps the existing environment without duplicating
simulation logic. SUMO remains an external integration: a local SUMO
installation and `SUMO_HOME`/TraCI configuration are required.

## Project limitations

- The simulator aggregates North+South and East+West into two straight-through
  movements while retaining four displayed approach queues.
- There are no turns, pedestrians, multiple intersections, or validated
  traffic-engineering assumptions.
- SUMO requires a separate local installation and scenario files.
- DQN, Gymnasium, and SUMO are optional integrations; the tested MVP does not
  require them.
- No performance claim should be made for Q-Learning until it has been trained
  and evaluated against matched-seed baselines.

## Repository structure

The implementation follows the separation described in the specification:

- `src/signalpilot/environment/`: intersection state, signal safety, traffic
  flow, and demand scenarios
- `src/signalpilot/agents/`: baseline, tabular Q-Learning, and optional DQN
- `src/signalpilot/training/`: training and evaluation workflows
- `src/signalpilot/metrics/`: episode metric collection
- `src/signalpilot/visualization/`: plots, traces, and intersection rendering
- `src/signalpilot/utils/`: reproducibility helpers
- `src/signalpilot/integrations/`: optional Gymnasium and SUMO adapters
- `config/`: default simulator configuration
- `experiments/`: reserved for user-generated experiment outputs
- `models/`: reserved for saved Q-tables and model artifacts
- `docs/`: reserved for project documentation
