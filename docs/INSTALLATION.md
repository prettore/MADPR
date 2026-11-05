# Installation

## Scope

This document describes the environment required to install the MADPR implementation. The project contains Python source files and does not require compilation of native project code.
The original baseline trainer at `src/rl/5GMLMA.py` imports Ray RLlib, PyTorch, Gymnasium, SciPy, NumPy, TensorBoard, and TensorBoardX. The added HAPPO/MAC-PHY path has its own dependency manifest and is documented separately below.

## Supported environment

Python 3.10 or newer is recommended. The source file `5GMLMA.py` uses modern type-annotation syntax, while the dependency versions are resolved by `pip` from `requirements.txt`. GPU execution is optional; use `--num-gpus 0` for a CPU-only run.

| Component | Requirement | Role |
| --- | --- | --- |
| Python | 3.10+ recommended | Runtime for the trainer and environment. |
| NumPy | Declared in `requirements.txt` | Array operations and state representation. |
| SciPy | Declared in `requirements.txt` | Graph and numerical operations used by the environment. |
| PyTorch | Declared in `requirements.txt` | Policy and value networks. |
| Gymnasium | Declared in `requirements.txt` | Environment and space interfaces. |
| Ray RLlib | Declared as `ray[rllib]` | Multi-agent training, evaluation, and checkpointing. |
| TensorBoard/TensorBoardX | Declared in `requirements.txt` | Training-metric logging and visualization. |

## Installation from a clean checkout

```bash
git clone https://github.com/prettore/MADPR.git
cd MADPR
python3 -m venv .venv-free-space
source .venv-free-space/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the baseline environment with `.venv-free-space\\Scripts\\activate` instead of `source .venv-free-space/bin/activate`.

## Installation check

The following commands validate syntax and imports without starting a Ray training job:

```bash
python -m py_compile src/rl/5GMLMA.py src/rl/drone_env_multi.py src/rl/calculaProbLink_otimizado.py
python -c "import gymnasium, numpy, scipy, torch, ray; print('MADPR dependencies imported successfully')"
```

The syntax check does not verify the behavior of a complete training run. A short execution test is described in [`USAGE.md`](USAGE.md).

## Hardware and resource notes

Ray can start local workers and temporary directories during training. For a first validation, use a single process, disable GPU allocation, and set a small number of timesteps. Larger experiments should record the CPU count, GPU model, driver/runtime versions, worker count, and the complete command line.

The environment variable `DRONE5G_WORKDIR` can redirect runtime artifacts:

```bash
export DRONE5G_WORKDIR=/path/to/madpr_runs/exp01
```

The default runtime directories are ignored by Git through the root `.gitignore` file.

## References

[1]: https://docs.ray.io/en/latest/rllib/ "Ray RLlib documentation"
[2]: https://gymnasium.farama.org/ "Gymnasium documentation"
[3]: https://pytorch.org/get-started/locally/ "PyTorch installation guidance"


## HAPPO + MAC/PHY path

The second trainer and environment are under `src/rl/macphy/`. Create a separate venv from the repository root:

```bash
python3 -m venv src/rl/macphy/.venv
src/rl/macphy/.venv/bin/python -m pip install --upgrade pip
src/rl/macphy/.venv/bin/python -m pip install -r src/rl/macphy/requirements.txt
```

Python 3.11 or 3.12 is recommended for the supplied package snapshot. These pinned versions are separate from the root RLlib requirements; do not assume that combining both manifests in one venv is compatible. This trainer is native PyTorch and does not require Ray/RLlib. `pandas`/`optuna` are in the supplied manifest for the archived Optuna artifact; that utility is not compatible with the native trainer. Install `tensorboard` into this venv if TensorBoard event-file visualization is required.

The native trainer can be inspected without starting a training job:

```bash
src/rl/macphy/.venv/bin/python src/rl/macphy/5GMLMA_MA.py --help
```

Check the shell launcher syntax with `bash -n src/rl/macphy/start.sh`. Its default automatic curriculum is long-running and detached; see the [HAPPO/MAC-PHY usage section](USAGE.md#happo-macphy-path) before launching it.
