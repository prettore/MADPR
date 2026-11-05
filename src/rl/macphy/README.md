# HAPPO + MAC/PHY model path

This directory contains the second MADPR implementation supplied for UAV backhaul topology generation. It is kept separate from the original RLlib/PPO baseline in `src/rl/`. This path uses a native PyTorch HAPPO trainer and evaluates links/topologies with the included modified Abreu-style MAC/physical solver.

The source snapshot identifies this as Phase II of *Adaptive UAV positioning and routing* (L. H. C. Reis et al., submitted to Elsevier Computer Communications) and reports source commit `4015f7c` dated 2026-09-08. This repository integrates the supplied implementation; it does not independently validate the model or reproduce paper results.

## Files

| File | Responsibility |
| --- | --- |
| `5GMLMA_MA.py` | Native PyTorch HAPPO trainer: rollouts, sequential heterogeneous actor updates, centralized critic, evaluation, checkpointing, and TensorBoard-compatible logging. No Ray/RLlib dependency. |
| `drone_env_multi_abreu.py` | UAV environment, action masks, and the wrappers used by the trainer. |
| `abreu_modified.py` | Propagation/link success model and multi-path MAC/PHY evaluator (queueing, contention/backoff, interference, retries, and goodput/reliability metrics). |
| `start.sh` | Automatic curriculum or manual-block launcher. It anchors default script, state, log, and result paths to this directory. |
| `requirements.txt` | Pinned dependency snapshot for the HAPPO/MAC-PHY path. |
| `legacy/optuna_2stages_outside.py` | Preserved Optuna script from the supplied bundle; it expects an older Ray/RLlib interface and is not compatible with this native HAPPO trainer. |

## Installation

From the repository root, use this model's own environment rather than the root RLlib environment:

```bash
python3 -m venv src/rl/macphy/.venv
src/rl/macphy/.venv/bin/python -m pip install --upgrade pip
src/rl/macphy/.venv/bin/python -m pip install -r src/rl/macphy/requirements.txt
```

Python 3.11 or 3.12 is recommended to match the supplied snapshot. TensorBoard is used when its optional runtime dependency is installed; otherwise training metrics are still written to JSON/console, but TensorBoard event visualization may be unavailable.

## Run

Inspect the native trainer's options without starting a training job:

```bash
src/rl/macphy/.venv/bin/python src/rl/macphy/5GMLMA_MA.py --help
```

The full automatic curriculum is the launcher's default and detaches into the background:

```bash
bash src/rl/macphy/start.sh
```

To run a single manual block in the foreground:

```bash
CURRICULUM=0 BLOCK_FOREGROUND=1 bash src/rl/macphy/start.sh
```

The automatic curriculum and default trainer settings are intended for substantial compute. Set the documented environment variables or use the trainer CLI to define an experiment's timestep budget, worker count, evaluation protocol, and output directories. `start.sh` prints the launch configuration; it can also be run with `RUN_BACKGROUND=0` to keep the curriculum attached to the terminal.

## Outputs

By default, this path writes run data below `results/`, best-model files below `ppo_best_model/`, and curriculum state/log files in this directory. The script supports custom locations through `PROJECT_ROOT`, `RESULTS_ROOT`, `LATEST_METRICS_PATH`, and `CURRICULUM_STATE_FILE`. The repository `.gitignore` excludes these generated artifacts.

The native PyTorch checkpoints are not compatible with the RLlib checkpoints produced by the original baseline. Keep run folders and checkpoint directories separate when doing comparative experiments.

## Propagation and evaluation model

The supplied `LinkModelParams` uses a 2.6 path-loss exponent and Rician fading for A2A link success. The environment applies episode/link shadowing and evaluates routes with a solver that includes multiple paths/copies, interference, 802.11 MAC contention/backoff, and finite M/M/1/K queues. The formulas, defaults, and remaining heuristic parts are in `abreu_modified.py` and the environment configuration. Reproducible scientific use should record those settings and calibrate them for the scenario; this code integration makes no independent accuracy or performance claim.

## Optuna status

`legacy/optuna_2stages_outside.py` is retained only as supplied-source history. It builds the old RLlib/PPO command line and searches Ray/Tune `progress.csv` files, while this folder's trainer is native HAPPO and writes its own JSON/checkpoint/TensorBoard artifacts. The legacy file is therefore not runnable as an HPO utility for this model. Do not use it unless adapting and validating it for the native trainer first.
