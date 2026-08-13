# MADPR — Multi-Agent Drone Positioning and Routing

![Workflow of the multi-agent learning process](img/workflow.png)

*Figure 1 — Workflow supplied with the project. The source PDF is preserved in [`img/source/WorkFlowMilcom2.pdf`](img/source/WorkFlowMilcom2.pdf).*

![Multi-agent operational scenario](img/multi-agent-scenario1.png)

*Figure 2 — Multi-agent scenario supplied with the project. The source PDF is preserved in [`img/source/Multi-agent-Scenario1.pdf`](img/source/Multi-agent-Scenario1.pdf).*

## Abstract

This repository contains the implementation of a multi-agent reinforcement-learning environment for drone positioning and routing. The supplied implementation models a source anchor, intermediate relay drones, forbidden zones, network links, link reliability, action masking, and a centralized state representation for training. The trainer uses Ray RLlib and PyTorch [1] [4], with two policy groups: `anchor` for the source anchor and `inter` for intermediate drones. The code supports evaluation, checkpointing, TensorBoard logging, automatic recovery from the best checkpoint, and configurable environment and optimization parameters.

The repository documents the executable implementation rather than reporting experimental performance. Numerical results, comparisons with baselines, and claims of algorithmic superiority must be added only after the corresponding experiments and statistical analysis are made available.

## Methodology and design

The environment represents a network in which the first drone acts as the source anchor and the last drone acts as the sink. Intermediate drones can move and modify local links. The multi-agent wrapper exposes agent identifiers with the prefixes `anchor_` and `inter_`; this enables policy sharing within each group while retaining different policy configurations between groups.

The observation design contains local observations, a centralized state vector, and an action mask. The custom Torch model uses the local observation to produce policy logits and the centralized state to estimate the value function, following the custom-model interface exposed by RLlib [2]. Invalid actions are suppressed before sampling. The environment evaluates connectivity and reliability using the optimized link-reliability module.

> **Implementation note.** The supplied workflow labels the learning loop as HAPPO, whereas the executable trainer constructs an RLlib `PPOConfig` and runs the `PPO` algorithm. This repository therefore describes the current code as **PPO-based multi-agent reinforcement learning**. A strict HAPPO claim requires reconciling the algorithm label with the implementation before publication.

## Repository structure

```text
MADPR/
├── src/
│   └── rl/
│       ├── 5GMLMA.py                       # Main RLlib trainer
│       ├── drone_env_multi.py               # Single-agent environment and MARL wrapper
│       └── calculaProbLink_otimizado.py     # Link-reliability calculations
├── img/
│   ├── workflow.png                         # Rasterized workflow figure
│   ├── multi-agent-scenario1.png            # Rasterized scenario figure
│   └── source/                              # Original one-page PDF figures
├── docs/
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── ARCHITECTURE.md
│   ├── COMPARISON.md
│   └── SOURCE_AUDIT.md
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Installation

The supplied trainer uses Python features compatible with a modern Python 3 environment and imports Ray RLlib, PyTorch, Gymnasium, SciPy, NumPy, TensorBoard, and TensorBoardX. Python 3.10 or newer is recommended because the source uses modern type annotations. A virtual environment should be used to isolate the dependencies.

```bash
git clone https://github.com/prettore/MADPR.git
cd MADPR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For additional details and installation checks, see [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Running the trainer

The main entry point is `src/rl/5GMLMA.py`. Run it from the module directory so that the local environment module is resolved directly; the environment follows the Gymnasium interface [3]:

```bash
cd src/rl
python 5GMLMA.py --timesteps 100000 --num-gpus 0
```

The default configuration is intended for a full training run and may require substantial computational resources. For a CPU-only smoke run, explicitly set `--num-gpus 0` and use a small value for `--timesteps`. A longer run can configure the worker and evaluation settings as follows:

```bash
python 5GMLMA.py \
  --timesteps 1000000 \
  --num-gpus 0 \
  --num-workers 2 \
  --envs-per-worker 1 \
  --run-name madpr_experiment \
  --instance-tag exp01
```

The trainer creates runtime artifacts under `src/rl/marl_workdir/` by default. The best checkpoint is written to `src/rl/ppo_best_model/` unless `--best-model-dir` is provided. Set `DRONE5G_WORKDIR` to place work files elsewhere.

Training metrics can be inspected with TensorBoard:

```bash
tensorboard --logdir src/rl/marl_workdir/tb
```

The complete command reference, resume procedure, and output layout are described in [`docs/USAGE.md`](docs/USAGE.md).

## Main configuration groups

| Group | Representative options | Purpose |
| --- | --- | --- |
| Training | `--timesteps`, `--n-steps`, `--batch-size`, `--n-epochs` | Controls the PPO optimization schedule. |
| Parallelism | `--num-workers`, `--envs-per-worker`, `--num-gpus` | Controls Ray workers and accelerator allocation. |
| Policies | `--policy-arch`, `--activation`, `--policy-arch-anchor`, `--policy-arch-inter` | Defines the neural-network architecture for each policy group. |
| Evaluation | `--eval-freq`, `--n-eval-episodes`, `--best-metric`, `--best-mode` | Controls periodic evaluation and best-checkpoint selection. |
| Environment | `--min-drones`, `--max-drones`, `--variable-drones`, `--episode-horizon` | Defines the fleet and episode configuration. |
| Zones and observations | `--zonas-count`, `--zona-w`, `--zona-h`, `--obs-zone-map`, `--zone-map-scale` | Configures forbidden zones and their representation in observations. |
| Reliability and reward | `--rel-step`, `--conn-quality-coef`, `--hops-weight`, `--layout-mix` | Adjusts reliability, connectivity, routing, and layout terms. |

## Documentation

The repository documentation is organized as follows:

| Document | Scope |
| --- | --- |
| [`INSTALLATION.md`](docs/INSTALLATION.md) | Environment creation, dependencies, and installation checks. |
| [`USAGE.md`](docs/USAGE.md) | Training commands, configuration, evaluation, resume, and TensorBoard. |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Modules, agent groups, observations, actions, rewards, and checkpoint flow. |
| [`COMPARISON.md`](docs/COMPARISON.md) | Scope differences between iDPR and MADPR, without unsupported performance claims. |
| [`SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md) | Required files, dependency notes, and compatibility decisions from the supplied audit. |

## Reproducibility and limitations

A reproducible experiment requires the exact command line, software versions, random seeds, hardware configuration, checkpoint identifier, and evaluation protocol to be recorded with the results. The repository currently provides the source code and configuration interface; it does not include trained checkpoints, fixed experiment manifests, or benchmark result tables.

The code creates runtime directories and may use Ray workers, TensorBoard writers, and RLlib checkpoints. These artifacts are intentionally excluded from version control. Source figures are preserved as PDFs and displayed in the repository as PNG files generated at 300 DPI.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## References

[1]: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html "Ray RLlib multi-agent environments"
[2]: https://docs.ray.io/en/latest/rllib/algorithms/ppo/ "Ray RLlib PPO"
[3]: https://gymnasium.farama.org/api/env/ "Gymnasium environment API"
[4]: https://pytorch.org/docs/stable/ "PyTorch documentation"
