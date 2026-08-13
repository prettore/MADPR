# Installation

## Scope

This document describes the environment required to install the MADPR implementation. The project contains Python source files and does not require compilation of native project code. The main trainer imports Ray RLlib, PyTorch, Gymnasium, SciPy, NumPy, TensorBoard, and TensorBoardX.

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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the virtual environment with `.venv\\Scripts\\activate` instead of `source .venv/bin/activate`.

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
