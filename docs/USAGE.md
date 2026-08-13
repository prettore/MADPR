# Usage

## Entry point

The trainer is `src/rl/5GMLMA.py`. It registers a turn-based multi-agent environment, creates two policy groups, configures an action-masked centralized-critic model, and launches an RLlib PPO experiment.

Run commands from `src/rl` because the trainer imports `drone_env_multi` as a local module:

```bash
cd src/rl
python 5GMLMA.py --timesteps 100000 --num-gpus 0
```

## Short validation run

A short run is useful after installation. It is a functional check, not a scientific experiment:

```bash
python 5GMLMA.py \
  --timesteps 2000 \
  --num-gpus 0 \
  --num-workers 0 \
  --envs-per-worker 1 \
  --eval-freq 1000 \
  --n-eval-episodes 1 \
  --run-name madpr_smoke \
  --instance-tag smoke
```

The exact number of collected samples can depend on RLlib batching and evaluation configuration. The run should be treated as successful only when the process initializes the environment, executes training iterations, and writes its output directories without an exception.

## Main training configuration

The default parser exposes a large set of options. The following table groups the parameters that are most relevant for reproducible experiments.

| Group | Options | Description |
| --- | --- | --- |
| Training horizon | `--timesteps`, `--n-steps`, `--batch-size`, `--minibatch-size`, `--n-epochs` | Controls the number of timesteps and PPO update sizes. |
| Optimizer | `--lr-start`, `--clip-start`, `--ent-coef`, `--gamma`, `--gae-lambda`, `--target-kl` | Configures learning-rate, clipping, entropy, discounting, GAE, and KL control. |
| Workers | `--num-workers`, `--envs-per-worker`, `--n-envs`, `--num-gpus`, `--max-speed` | Controls Ray parallelism and resource allocation. |
| Policy networks | `--policy-arch`, `--activation`, `--policy-arch-anchor`, `--policy-arch-inter`, `--activation-anchor`, `--activation-inter` | Defines shared or group-specific policy-network settings. |
| Fleet | `--min-drones`, `--max-drones`, `--variable-drones`, `--episode-horizon` | Defines the number of drones and the episode horizon. |
| Forbidden zones | `--zonas-count`, `--zona-w`, `--zona-h`, `--zona-single-fixed`, `--p-sem-zona` | Controls zone generation and placement. |
| Observations | `--obs-zone-map`, `--zone-map-scale`, `--obs-edge-cross`, `--mask-actions` | Controls map, edge, and action-mask information. |
| Reward terms | `--rel-step`, `--conn-quality-coef`, `--hops-weight`, `--layout-mix`, `--line-align-w` | Adjusts reliability, connectivity, hop, layout, and alignment terms. |
| Evaluation | `--eval-freq`, `--n-eval-episodes`, `--best-metric`, `--best-mode`, `--best-min-delta` | Controls evaluation and best-checkpoint selection. |
| Checkpoints | `--checkpoint-freq-steps`, `--best-model-dir`, `--best-copy-mode`, `--run-dir` | Controls checkpoint frequency and storage. |

For a larger CPU experiment:

```bash
python 5GMLMA.py \
  --timesteps 1000000 \
  --num-gpus 0 \
  --num-workers 2 \
  --envs-per-worker 1 \
  --n-eval-episodes 5 \
  --run-name madpr_cpu \
  --instance-tag exp01
```

## Policies and agent groups

The policy mapping function assigns `anchor_*` agents to the `anchor` policy and all `inter_*` agents to the shared `inter` policy. The source anchor controls its local link actions. Intermediate agents use a combined action space for movement and local link toggling. The sink is represented by the final drone but does not receive an outgoing-link management action.

The architecture can be customized independently for the two groups:

```bash
python 5GMLMA.py \
  --policy-arch-anchor 256,256 \
  --policy-arch-inter 384,384,384 \
  --activation-anchor tanh \
  --activation-inter relu \
  --timesteps 100000 \
  --num-gpus 0
```

## Resume and best checkpoint

By default, `--resume-from-best 1` allows resume mode to locate a valid best checkpoint under the configured best-model directory. To request a resume:

```bash
python 5GMLMA.py \
  --resume 1 \
  --run-name madpr_cpu \
  --best-model-dir /path/to/ppo_best_model \
  --num-gpus 0
```

A specific checkpoint can be provided explicitly:

```bash
python 5GMLMA.py \
  --resume 1 \
  --restore-path /path/to/checkpoint \
  --run-name madpr_cpu \
  --num-gpus 0
```

The legacy value `--use-model continue` is accepted as an alias for resume mode. Do not use `--overwrite 1` together with a directory whose previous logs must be retained; that option removes the existing experiment directory when resume is disabled.

## Runtime outputs

The trainer writes output under the work directory derived from `DRONE5G_WORKDIR`. If the variable is not set, the base is `src/rl/marl_workdir/`. The main runtime categories are summarized below.

| Location | Content |
| --- | --- |
| `<workdir>/tb/` | TensorBoard-related Ray/Tune output. |
| `<workdir>/tb_custom/` | Custom scalar logs emitted by callbacks. |
| `<workdir>/saves/` | Experiment directories and checkpoints when the default run directory is used. |
| `--best-model-dir` | Copied best checkpoint and recovery sidecars. |
| `/tmp/ray_marl_<tag>` | Short-path link or fallback for Ray temporary files. |

Use a distinct `--instance-tag` when launching independent experiments concurrently. This prevents the Ray temporary directory and experiment names from colliding.

## TensorBoard

From the repository root, inspect both the Ray/Tune and custom logs as needed:

```bash
tensorboard --logdir src/rl/marl_workdir/tb
```

If custom metrics are required:

```bash
tensorboard --logdir src/rl/marl_workdir/tb_custom
```

The callback exposes terminal evaluation metrics under names such as `eval_terminal/reliability_terminal_mean`, which is also the default best-model metric. The exact scalar set depends on the evaluation lifecycle and the RLlib version.

## Reproducibility record

For each experiment, record the repository commit, Python version, dependency installation state, hardware, random-seed configuration, complete command line, work directory, best metric, checkpoint path, and evaluation episode count. A checkpoint alone is not sufficient to reconstruct an experiment if these conditions are omitted.

## References

[1]: https://docs.ray.io/en/latest/rllib/algorithms/ppo/ "Ray RLlib PPO"
[2]: https://docs.ray.io/en/latest/tune/ "Ray Tune documentation"
[3]: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html "Ray RLlib multi-agent environments"
