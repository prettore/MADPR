# Comparison with iDPR

## Purpose

MADPR follows repository conventions visible in [iDPR][1]: implementation under `src/`, figures under `img/`, technical documentation under `docs/`, dependency manifests, and a README describing installation and execution. This document compares scope and organization, not performance. The supplied files do not include a shared benchmark table, trained checkpoints, or experiment manifest for an apples-to-apples comparison.

## Scope comparison

| Dimension | iDPR | MADPR |
| --- | --- | --- |
| Primary problem | Intelligent drone positioning and routing. | Multi-agent drone positioning and routing. |
| Main implementation split | Reinforcement learning in Python and genetic-algorithm components in C++. | Two isolated Python multi-agent paths: original RLlib/PPO baseline and native HAPPO with MAC/PHY evaluation. |
| Original RL entry point | `src/rl/5GML.py`. | `src/rl/5GMLMA.py`. |
| Added HAPPO entry point | — | `src/rl/macphy/5GMLMA_MA.py`. |
| Environment interface | Project-specific single-agent positioning interface. | Original turn-based RLlib wrapper and a separate HAPPO/MAC-PHY environment/wrapper. |
| Link evaluation | Project-specific reliability implementation. | Original `calculaProbLink_otimizado.py`; added path uses `abreu_modified.py` for Rician links, MAC/PHY, queues, and multipath metrics. |
| Experimental claims | Interpret claims in the context of iDPR's own experiments. | No new performance claim is made by this source integration. |

## Methodological distinction

The two MADPR paths differ from each other as well as from iDPR. The original path is RLlib PPO even though its supplied workflow figure labels HAPPO; it remains unchanged as the baseline. The added path provides a native PyTorch HAPPO trainer together with a more detailed link/MAC/PHY evaluator. Neither the HAPPO label nor additional physical-layer detail by itself establishes a performance or propagation-accuracy improvement.

## What can be compared experimentally

A valid comparison among iDPR and the MADPR paths should control or explicitly report scenario geometry, radio and traffic assumptions, grid, forbidden zones, initialization, training/evaluation seeds, compute budget, and metric definitions. At minimum, define common end-to-end metrics where possible and report reliability, delivery ratio, latency, throughput/goodput, constraint violations, training/inference cost, and run-to-run variation. Model-specific metrics should remain clearly identified rather than treated as identical quantities.

Until such controls and measurements are provided, the repository documents implementation scope and source organization only. The absence of a results table is intentional and avoids implying conclusions unsupported by the supplied materials.

## References

[1]: https://github.com/prettore/iDPR "prettore/iDPR repository"
[2]: https://github.com/prettore/MADPR "prettore/MADPR repository"
