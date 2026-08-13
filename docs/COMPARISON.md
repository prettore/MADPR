# Comparison with iDPR

## Purpose

MADPR was organized using the repository conventions visible in [iDPR][1]: a `src/` directory for implementation, an `img/` directory for figures, a `docs/` directory for technical documentation, a root dependency file, and a README that explains installation and execution. The comparison below addresses scope and organization. It does not compare performance because the supplied MADPR package contains no benchmark table, trained checkpoint, or experiment manifest.

## Scope comparison

| Dimension | iDPR | MADPR |
| --- | --- | --- |
| Primary problem | Intelligent drone positioning and routing. | Multi-agent drone positioning and routing. |
| Main implementation split | Reinforcement learning in Python and genetic algorithm components in C++. | Multi-agent reinforcement learning in Python. |
| RL entry point | `src/rl/5GML.py`. | `src/rl/5GMLMA.py`. |
| Environment interface | Single-agent Gymnasium environment for positioning. | Turn-based RLlib multi-agent wrapper over `DroneEnv`. |
| Policy organization | One PPO-oriented RL implementation described in the iDPR README. | `anchor` and shared `inter` policy groups mapped from agent identifiers. |
| Observation design | Project-specific state representation. | Local observation, centralized state, and action mask exposed by dictionary spaces. |
| Link-reliability module | `calculaProbLink_otimizado.py`. | The same module name is retained and supplied in the MADPR archive. |
| Figures | Workflow figure in `img/`. | Workflow and multi-agent scenario figures in `img/`, with original PDFs preserved under `img/source/`. |
| Experimental claims | Claims are described in the iDPR README and should be interpreted with its own experimental context. | This repository makes no new performance claim; it documents the supplied implementation. |

## Methodological distinction

The central architectural difference is the transition from a single policy/environment presentation to a multi-agent interface. MADPR exposes agent IDs and policy groups, supports turn-based interaction, and supplies a centralized state to the value function while retaining agent-local policy inputs. This is a change in the learning interface and training configuration; it should not be interpreted as evidence of improved performance without a controlled evaluation.

The supplied workflow labels the learning loop as HAPPO. The current trainer, however, imports `PPOConfig` from RLlib and calls `tune.run("PPO", ...)`. The implementation and the figure therefore use different algorithm labels. For scientific writing, the algorithm name should be standardized after the authors decide whether the intended contribution is an RLlib PPO configuration or a HAPPO implementation with the corresponding update rule.

## What can be compared experimentally

A valid comparison between iDPR and MADPR should keep the scenario, radio model, grid, forbidden zones, reliability calculation, initialization protocol, computational budget, and evaluation seeds controlled. At minimum, the study should report the number of drones, reliability, end-to-end latency, throughput or connectivity metrics when defined, constraint violations, training cost, inference cost, and dispersion across repeated runs.

Until these controls and measurements are available, this repository limits itself to a transparent implementation comparison. The absence of a result table is intentional and prevents the documentation from implying conclusions not supported by the supplied materials.

## References

[1]: https://github.com/prettore/iDPR "prettore/iDPR repository"
[2]: https://github.com/prettore/MADPR "prettore/MADPR repository"
[3]: https://docs.ray.io/en/latest/rllib/algorithms/ppo/ "Ray RLlib PPO"
[4]: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html "Ray RLlib multi-agent environments"
