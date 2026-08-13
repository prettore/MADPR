# Architecture

## Overview

MADPR is organized as a compact Python reinforcement-learning implementation. The design separates the physical/network environment, the optimized reliability calculation, and the RLlib trainer. The separation follows the same `src/`, `img/`, and `docs/` organization used by iDPR, but the executable scope is multi-agent rather than a combined RL and genetic-algorithm repository.

```text
src/rl/
├── 5GMLMA.py
├── drone_env_multi.py
└── calculaProbLink_otimizado.py
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `5GMLMA.py` | Configures Ray, registers the environment, defines the action-masked centralized-critic Torch model, maps agents to policies, launches PPO, evaluates policies, and manages checkpoints and TensorBoard callbacks. |
| `drone_env_multi.py` | Implements the underlying `DroneEnv` and the turn-based `Drone5GMAEnv` wrapper. It defines drone placement, forbidden zones, actions, observations, transitions, reward terms, masks, and terminal metrics. |
| `calculaProbLink_otimizado.py` | Computes distance-dependent link reliability and iterative network reliability using radio parameters, position matrices, adjacency information, and a topological-order fallback. |

## Agent model

The wrapper exposes two policy groups:

| Agent identifier | Policy group | Role |
| --- | --- | --- |
| `anchor_0` | `anchor` | Source-side anchor responsible for local outgoing-link actions. |
| `inter_1` through `inter_N-2` | `inter` | Intermediate relay drones responsible for movement and local outgoing-link actions. |
| Final drone `N-1` | Sink representation | Destination endpoint; it does not manage outgoing links. |

The policy mapping function uses the identifier prefix. All intermediate agents share the `inter` policy, while the source anchor uses the `anchor` policy. The trainer can assign different network architectures and activation functions to the two groups.

## Environment lifecycle

The environment is turn-based. At each decision, the active agent receives an observation and selects an action. The transition updates either a local link configuration, the position of an intermediate drone, or both, depending on the agent group and the action encoding. The environment then recalculates the relevant topology and returns the next active-agent observation.

The wrapper publishes agent-specific observation and action spaces. The observations are dictionary spaces containing at least the following conceptual components:

| Component | Function |
| --- | --- |
| `obs` | Local information used by the policy network to choose an action. |
| `state` | Centralized information used by the value function under the centralized-training/decentralized-execution design. |
| `action_mask` | Feasible-action indicator used to suppress invalid actions. |

The exact vector lengths are derived at runtime from the configured number of drones, zones, and observation flags. Consumers should obtain the spaces from the environment rather than hard-code dimensions.

## Model and training loop

`TorchActionMaskCCModel` supports both dictionary and flattened observations. The policy network receives the local observation. The value network receives the centralized state when it is available. Before action sampling, invalid logits receive a large negative mask value. This avoids selecting actions that the environment has marked as infeasible.

The trainer builds an RLlib `PPOConfig`, registers two policy specifications, maps agent IDs to policy groups, and enables evaluation. Callback code collects terminal metrics, writes custom TensorBoard scalars, and selects a best checkpoint according to a configurable evaluation metric. The implementation also mirrors the selected checkpoint to the best-model directory and writes recovery sidecars.

```text
reset environment
      │
      ▼
observation + state + action mask
      │
      ▼
policy mapping: anchor / inter
      │
      ▼
action-masked policy logits
      │
      ▼
position or link transition
      │
      ▼
topology, reliability, reward, terminal metrics
      │
      ├── PPO update through RLlib
      ├── evaluation episodes
      └── checkpoint and TensorBoard output
```

## Reliability and reward interface

The optimized link module first evaluates distance-dependent link reliability from radio parameters. The environment uses these values when evaluating the current adjacency/topology and assembling the reliability-related reward terms. Additional terms in the environment account for connectivity quality, hop count, layout, line alignment, disjoint-path conditions, forbidden-zone behavior, and invalid actions. Their weights are configurable through command-line options.

The documentation intentionally does not assign a single physical interpretation to every reward coefficient beyond the names exposed by the source code. Experimental papers should define the complete mathematical objective, units, normalization, and coefficient-selection protocol separately.

## Checkpoint and logging flow

The trainer creates a work directory relative to the script location unless `DRONE5G_WORKDIR` is set. Ray/Tune artifacts and custom TensorBoard files use separate subdirectories. Evaluation callbacks select the best model according to `--best-metric` and `--best-mode`; the resulting checkpoint is copied, rather than symlinked, to the best-model directory by the trainer's current runtime policy.

## References

[1]: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html "Ray RLlib multi-agent environments"
[2]: https://docs.ray.io/en/latest/rllib/rllib-models.html "Ray RLlib custom models"
[3]: https://docs.ray.io/en/latest/rllib/algorithms/ppo/ "Ray RLlib PPO"
[4]: https://gymnasium.farama.org/api/env/ "Gymnasium environment API"
