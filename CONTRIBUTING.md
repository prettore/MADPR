# Contributing to MADPR

Contributions should preserve the reproducibility and scientific scope of the project. Changes to the environment, reward function, radio model, policy mapping, or checkpoint logic should be described in the pull request and reflected in the relevant documentation.

## Development setup

Create the project environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional development tools may be installed separately when needed:

```bash
pip install pytest black flake8
```

## Source organization

Python source belongs under `src/rl/`. Documentation belongs under `docs/`, and repository figures belong under `img/`. Original figure sources should be preserved under `img/source/` when redistribution permits it. Runtime logs, checkpoints, Ray temporary files, and TensorBoard artifacts must not be committed.

## Validation before a pull request

At minimum, run the syntax check:

```bash
python -m py_compile src/rl/5GMLMA.py src/rl/drone_env_multi.py src/rl/calculaProbLink_otimizado.py
```

When dependencies are available, run a short CPU-only smoke test with a unique run name and instance tag. If a change affects action spaces, observation spaces, reward terms, or policy mapping, the pull request should include the command used and the observed output directory.

## Scientific changes

Changes that alter the objective, reliability computation, topology constraints, or evaluation metrics should include the mathematical definition, units, normalization, and rationale in the documentation. Do not report performance improvements without repeated controlled experiments and an explicit baseline.

If the implementation is changed from RLlib PPO to another algorithm, update the README, architecture description, source audit, dependency file, and figure captions so that the algorithm name is consistent across the repository.

## Commit and pull-request conventions

Use concise imperative commit subjects, for example:

```text
Document multi-agent observation spaces
```

Pull requests should state the motivation, summarize the code and documentation changes, list validation commands, and identify any limitations or unresolved compatibility issues.

## License

By contributing to MADPR, you agree that your contribution is distributed under the repository's MIT License.

## References

[1]: https://peps.python.org/pep-0008/ "PEP 8 style guide"
[2]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests "GitHub pull request documentation"
