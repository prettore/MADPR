# HAPPO + MAC/PHY

Este diretório contém o segundo caminho de execução do MADPR, separado do baseline PPO/RLlib em `src/rl/`.

- `5GMLMA_MA.py`: treinador HAPPO nativo em PyTorch.
- `drone_env_multi_abreu.py`: ambiente e wrappers multiagente.
- `abreu_modified.py`: modelo de enlace Rician e avaliação de caminhos, interferência, contenção IEEE 802.11 e filas M/M/1/K.
- `start.sh`: currículo automático ou execução de bloco manual.
- `requirements.txt`: dependências próprias do caminho HAPPO/MAC-PHY.
- `legacy/optuna_2stages_outside.py`: arquivo Optuna legado, incompatível com o treinador nativo; preservado apenas como referência.

Instale em venv separado a partir da raiz do repositório:

```bash
python3 -m venv src/rl/macphy/.venv
src/rl/macphy/.venv/bin/python -m pip install -r src/rl/macphy/requirements.txt
src/rl/macphy/.venv/bin/python src/rl/macphy/5GMLMA_MA.py --help
```

O launcher padrão pode iniciar currículo longo em segundo plano; revise as configurações antes de executar:

```bash
bash src/rl/macphy/start.sh
```

Mais detalhes e a comparação com o baseline estão em [`docs/MODELOS.md`](../../../docs/MODELOS.md) e [`docs/USAGE.md`](../../../docs/USAGE.md).
