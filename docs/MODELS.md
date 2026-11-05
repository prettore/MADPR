# Model paths

MADPR contains two separate multi-agent pipelines. Choose the path based on the experiment's propagation/link assumptions and training algorithm; do not treat them as interchangeable implementations of one checkpoint.

## At a glance

| | Original baseline — Free Space path | HAPPO + MAC/PHY path |
| --- | --- | --- |
| Directory | [`src/rl/`](../src/rl/) | [`src/rl/macphy/`](../src/rl/macphy/) |
| Trainer | [`5GMLMA.py`](../src/rl/5GMLMA.py) | [`5GMLMA_MA.py`](../src/rl/macphy/5GMLMA_MA.py) |
| Environment | [`drone_env_multi.py`](../src/rl/drone_env_multi.py) | [`drone_env_multi_abreu.py`](../src/rl/macphy/drone_env_multi_abreu.py) |
| Link evaluator | [`calculaProbLink_otimizado.py`](../src/rl/calculaProbLink_otimizado.py) | [`abreu_modified.py`](../src/rl/macphy/abreu_modified.py) |
| Trainer implementation | Ray RLlib PPO; shared `anchor` and `inter` policy groups | Native PyTorch HAPPO; sequential heterogeneous actor updates and centralized critic |
| Dependencies | Root [`requirements.txt`](../requirements.txt) | [`src/rl/macphy/requirements.txt`](../src/rl/macphy/requirements.txt) |
| Outputs | `src/rl/marl_workdir/`, `src/rl/ppo_best_model/` (by default) | `src/rl/macphy/results/`, `src/rl/macphy/ppo_best_model/` (by default) |

## Original baseline — Free Space path

This is the implementation that was already in MADPR. Its trainer is an RLlib `PPOConfig`/PPO run; the two policy groups are named `anchor` and `inter`. The environment, trainer, and link-reliability helper remain at their original paths, which preserves existing commands and source references.

The supplied link helper, `calculaProbLink_otimizado.py`, defines a Friis-based reference loss, a distance-dependent path-loss exponent (`n=2.3` by default), and log-normal shadowing (`sigma_db=4.0` dB by default). Thus, while this path is called the Free Space baseline in the project context, its checked-in defaults are not a pure Friis-only model. Use and report the actual parameters from the code/configuration when describing an experiment.

See [the baseline architecture](ARCHITECTURE.md#original-baseline-free-space-path) and [baseline usage](USAGE.md#original-baseline-free-space-path).

## HAPPO + MAC/PHY path

The added path keeps the supplied second implementation together under `src/rl/macphy/`:

| File | Role |
| --- | --- |
| [`5GMLMA_MA.py`](../src/rl/macphy/5GMLMA_MA.py) | Native PyTorch HAPPO trainer; rollout, sequential actor updates, centralized critic, evaluation, and PyTorch checkpoints. It does not use Ray/RLlib. |
| [`drone_env_multi_abreu.py`](../src/rl/macphy/drone_env_multi_abreu.py) | Environment, action masking, and multi-agent wrappers used by the native trainer. |
| [`abreu_modified.py`](../src/rl/macphy/abreu_modified.py) | Link model and modified Abreu-style evaluator: Rician A2A success probability, multiple paths/copies, interference, 802.11 contention/backoff, and M/M/1/K queues. |
| [`start.sh`](../src/rl/macphy/start.sh) | Automatic curriculum and manual-block launcher; paths default to this model directory. |
| [`requirements.txt`](../src/rl/macphy/requirements.txt) | Isolated dependency set for this path. |
| [`README.md`](../src/rl/macphy/README.md) | Model-specific file map and run instructions. |

The link model defines a 2.6 path-loss exponent and a Rician fading parameter in `LinkModelParams`; the environment also samples per-link shadowing and uses the MAC/PHY evaluator for its configured graph metrics. MAC parameters such as carrier-sense range, contention window, retries, and offered traffic affect the computed performance. These are more detailed assumptions than a distance-only link calculation, but their empirical validity for a study's scenario still needs calibration against an appropriate measurement or reference model.

See [the HAPPO/MAC-PHY architecture](ARCHITECTURE.md#happo-macphy-path) and [its usage instructions](USAGE.md#happo-macphy-path).

### Archived Optuna script

The attachment also contained `optuna_2stages_outside.py`. It assumes the older RLlib/PPO CLI, Ray/Tune `progress.csv` outputs, and flags that the native HAPPO trainer does not expose. It is preserved at [`src/rl/macphy/legacy/optuna_2stages_outside.py`](../src/rl/macphy/legacy/optuna_2stages_outside.py) for provenance only; it is **not a working HPO entry point for this HAPPO path**. Do not launch it as part of a HAPPO experiment. A compatible HPO utility would need to be implemented against the native trainer's CLI and JSON/TensorBoard outputs.

## Experimental comparison

The environments differ in more than propagation: their action/observation dimensions, reward terms, policies, metric definitions, and training algorithms also differ. A direct checkpoint transfer is unsupported. To compare results scientifically, define a common scenario and evaluation protocol, align traffic and radio assumptions where meaningful, use repeatable seeds, and report both each model's native metrics and common end-to-end metrics. More complex model assumptions are not, on their own, evidence of greater accuracy or better learned policies.
