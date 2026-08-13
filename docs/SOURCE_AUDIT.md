# Source audit

## Required source files

The supplied archive identifies the following files as the complete local source needed for a fresh training run:

| File | Repository location | Function |
| --- | --- | --- |
| `5GMLMA.py` | `src/rl/5GMLMA.py` | Main RLlib trainer and policy configuration. |
| `drone_env_multi.py` | `src/rl/drone_env_multi.py` | Environment and multi-agent wrapper. |
| `calculaProbLink_otimizado.py` | `src/rl/calculaProbLink_otimizado.py` | Optimized link-reliability implementation. |

No additional non-package project file is required for a new run. Work directories, TensorBoard logs, checkpoints, manifests, and recovery bundles are created at runtime.

## Redundant or optional compatibility files

The archive audit reports that `calculaProbLink.py` is a final fallback used only when the optimized module and a CamelCase intermediate fallback are unavailable. The optimized module is present in this repository, so a duplicate `calculaProbLink.py` was not added.

The filename `calculaProbLinkOtimizado.py` is also reported as an intermediate CamelCase fallback. It is not required while `calculaProbLink_otimizado.py` is available, and no duplicate was added. Keeping a single canonical filename avoids ambiguous imports and divergent implementations.

The audit also notes that `networkx` is used only opportunistically by the fallback module, while the optimized reliability module does not require it. It is therefore not included in `requirements.txt`. Likewise, `gym` is only a legacy fallback in the environment; the trainer imports Gymnasium directly, so `gymnasium` is the declared interface dependency.

## Python dependencies

```text
numpy
scipy
torch
gymnasium
ray[rllib]
tensorboard
tensorboardX
```

The file `requirements.txt` in the repository is copied from the supplied archive. Dependency versions are intentionally not pinned because no lockfile or tested version matrix was included. Reproducible experiments should record the resolved package versions after installation.

## Algorithm-label consistency

The source archive is named `code_MultiAgent_HAPPO.zip`, and the workflow figure contains the label HAPPO. The executable trainer imports `PPOConfig` and invokes the RLlib `PPO` algorithm. This audit records the discrepancy rather than silently changing the algorithm or renaming the source. The README and architecture documents identify the current executable path as PPO-based multi-agent RL and mark the HAPPO label for author review.

## Validation boundary

This audit verifies file presence, imports visible in the source, declared dependencies, and the absence of redundant compatibility files. It does not establish convergence, physical-model validity, statistical significance, or superiority over a baseline. Those properties require controlled experiments and are outside the scope of a source-file integration.

## References

[1]: https://docs.ray.io/en/latest/rllib/algorithms/ppo/ "Ray RLlib PPO"
[2]: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html "Ray RLlib multi-agent environments"
[3]: https://gymnasium.farama.org/api/env/ "Gymnasium environment API"
