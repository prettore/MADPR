# Legacy HPO artifact

`optuna_2stages_outside.py` is preserved from the supplied source bundle for provenance. It targets an older Ray/RLlib PPO trainer, expects Ray/Tune `progress.csv`, and passes command-line options absent from the native HAPPO trainer in the parent directory. It is **not supported or runnable as the HAPPO/MAC-PHY hyperparameter search**. A future compatible search tool must be adapted to the native CLI and its JSON/TensorBoard result schema, then validated separately.
