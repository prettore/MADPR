# MADPR — Posicionamento e roteamento multiagente de drones

O repositório mantém agora dois caminhos de pesquisa independentes: o baseline original PPO/RLlib e o modelo HAPPO com avaliação MAC/PHY. Os códigos, ambientes, dependências, métricas e checkpoints ficam separados. Para comparar os pressupostos, consulte o [guia detalhado dos modelos](docs/MODELOS.md).

## Escolha do modelo

| Caminho | Entrada principal | Propagação/avaliação | Treinador |
| --- | --- | --- | --- |
| Baseline original (Free Space) | [`src/rl/5GMLMA.py`](src/rl/5GMLMA.py) | [`calculaProbLink_otimizado.py`](src/rl/calculaProbLink_otimizado.py) | PPO via Ray RLlib |
| HAPPO + MAC/PHY | [`src/rl/macphy/5GMLMA_MA.py`](src/rl/macphy/5GMLMA_MA.py) | [`abreu_modified.py`](src/rl/macphy/abreu_modified.py) e [`drone_env_multi_abreu.py`](src/rl/macphy/drone_env_multi_abreu.py) | HAPPO nativo em PyTorch |

O código original permanece nos caminhos `src/rl/`. O modelo novo foi adicionado em `src/rl/macphy/`, sem substituir o baseline. As duas implementações têm espaços de observação/ação, métricas, dependências e formatos de checkpoint diferentes; não transfira checkpoints entre elas.

> **Nota sobre a propagação.** O baseline é chamado de Free Space no contexto do projeto, mas o código entregue usa perda de referência baseada em Friis, expoente de perda `n=2.3` e sombreamento log-normal `σ=4 dB` por padrão. Portanto, não é uma fórmula Friis pura. O caminho novo inclui desvanecimento Rician, perda de percurso com expoente 2.6 e avaliação MAC/PHY com filas, contenção e caminhos múltiplos. Esses detalhes ampliam o modelo de enlace, mas não substituem calibração nem demonstram, por si só, maior precisão.

## Instalação isolada

Use um ambiente virtual para cada caminho.

**Baseline PPO/RLlib**, a partir da raiz do repositório:

```bash
python3 -m venv .venv-free-space
source .venv-free-space/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**HAPPO + MAC/PHY**, também a partir da raiz:

```bash
python3 -m venv src/rl/macphy/.venv
src/rl/macphy/.venv/bin/python -m pip install --upgrade pip
src/rl/macphy/.venv/bin/python -m pip install -r src/rl/macphy/requirements.txt
```

Python 3.11 ou 3.12 é recomendado para o snapshot HAPPO. Não instale os dois manifestos no mesmo ambiente sem validar a compatibilidade do Ray/RLlib com as versões fixadas pelo modelo novo.

## Execução

Ative `.venv-free-space` e execute o baseline a partir de `src/rl`:

```bash
cd src/rl
python 5GMLMA.py --timesteps 100000 --num-gpus 0
```

Para consultar as opções do treinador HAPPO, sem iniciar treino:

```bash
src/rl/macphy/.venv/bin/python src/rl/macphy/5GMLMA_MA.py --help
```

O launcher HAPPO inicia por padrão um currículo longo em segundo plano e agora ancora os caminhos no próprio diretório do modelo:

```bash
bash src/rl/macphy/start.sh
```

Um bloco manual em primeiro plano pode ser iniciado com:

```bash
CURRICULUM=0 BLOCK_FOREGROUND=1 bash src/rl/macphy/start.sh
```

Revise `src/rl/macphy/start.sh` antes de executar uma experiência completa: os valores padrão podem solicitar muitos workers e milhões de passos.

## Arquivos principais

```text
src/rl/
├── 5GMLMA.py                       # PPO/RLlib do baseline
├── drone_env_multi.py              # ambiente original
└── calculaProbLink_otimizado.py    # confiabilidade do enlace original

src/rl/macphy/
├── 5GMLMA_MA.py                    # treinador HAPPO nativo em PyTorch
├── drone_env_multi_abreu.py        # ambiente e wrappers MAC/PHY
├── abreu_modified.py               # propagação, MAC/PHY, filas e múltiplos caminhos
├── start.sh                        # currículo e blocos de treino
├── requirements.txt                # dependências isoladas
└── legacy/                          # ferramenta Optuna antiga, não compatível
```

O [guia de modelos](docs/MODELOS.md) detalha as diferenças, dependências, outputs e limitações. A integração não inclui resultados experimentais ou checkpoints e não afirma superioridade de um modelo. Registre parâmetros físicos, tráfego, seeds, versões, hardware e protocolo de avaliação ao publicar resultados.
