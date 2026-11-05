# Modelos disponíveis no MADPR

O MADPR disponibiliza dois caminhos multiagente independentes para posicionamento e roteamento de UAVs. Escolha o caminho conforme o algoritmo e os pressupostos de propagação/MAC necessários ao experimento.

## Comparação rápida

| Item | Baseline original — Free Space | HAPPO + MAC/PHY |
| --- | --- | --- |
| Diretório | [`src/rl/`](../src/rl/) | [`src/rl/macphy/`](../src/rl/macphy/) |
| Treinador | [`5GMLMA.py`](../src/rl/5GMLMA.py) | [`5GMLMA_MA.py`](../src/rl/macphy/5GMLMA_MA.py) |
| Ambiente | [`drone_env_multi.py`](../src/rl/drone_env_multi.py) | [`drone_env_multi_abreu.py`](../src/rl/macphy/drone_env_multi_abreu.py) |
| Avaliação de enlace | [`calculaProbLink_otimizado.py`](../src/rl/calculaProbLink_otimizado.py) | [`abreu_modified.py`](../src/rl/macphy/abreu_modified.py) |
| Treinamento | PPO via Ray RLlib; políticas compartilhadas `anchor` e `inter` | HAPPO nativo em PyTorch; atualização sequencial de atores heterogêneos e crítico centralizado |
| Dependências | [`requirements.txt`](../requirements.txt) | [`src/rl/macphy/requirements.txt`](../src/rl/macphy/requirements.txt) |
| Saídas padrão | `src/rl/marl_workdir/`, `src/rl/ppo_best_model/` | Com `start.sh`: `src/rl/macphy/results/`, `src/rl/macphy/ppo_best_model/` |

## Baseline original — caminho Free Space

Este é o código que já existia no MADPR. O treinador executável `src/rl/5GMLMA.py` configura o algoritmo PPO do RLlib; apesar do rótulo HAPPO usado na figura de workflow antiga, esse treinador específico não implementa as atualizações HAPPO.

O módulo `calculaProbLink_otimizado.py` usa perda de referência baseada em Friis, expoente de perda de percurso `n=2.3` e desvio-padrão de sombreamento log-normal de `4 dB` nos valores padrão. Por isso, embora o projeto chame este baseline de Free Space, a implementação não é somente uma equação Friis pura. Descreva os parâmetros reais do código ao documentar seus experimentos.

### Arquivos do baseline

- `src/rl/5GMLMA.py`: treinador RLlib PPO, mapeamento de políticas, avaliação e checkpoints.
- `src/rl/drone_env_multi.py`: ambiente e wrapper multiagente do baseline.
- `src/rl/calculaProbLink_otimizado.py`: cálculo de probabilidade/confiabilidade do enlace.
- `requirements.txt`: dependências do caminho RLlib/PPO.

## HAPPO + MAC/PHY

O segundo caminho é adicionado em `src/rl/macphy/` sem substituir os fontes do baseline. O treinador é nativo em PyTorch, sem dependência de Ray/RLlib. O avaliador usa parâmetros Rician A2A e expoente de perda de percurso 2.6 nos defaults fornecidos, além de considerar caminhos múltiplos/cópias, interferência, contenção/backoff IEEE 802.11, tentativas de retransmissão e filas finitas M/M/1/K.

O conjunto tem mais componentes de rádio/MAC que uma avaliação baseada apenas em distância. Isso é a motivação da implementação anexada, mas **não comprova validade física ou superioridade**. Calibre as hipóteses ao cenário e reporte os valores usados.

### Arquivos HAPPO/MAC-PHY

| Arquivo | Papel |
| --- | --- |
| [`5GMLMA_MA.py`](../src/rl/macphy/5GMLMA_MA.py) | Treinador HAPPO nativo, rollouts, atualizações, avaliação e checkpoints `.pt`. |
| [`drone_env_multi_abreu.py`](../src/rl/macphy/drone_env_multi_abreu.py) | Ambiente, action masks e wrappers usados pelo treinador. |
| [`abreu_modified.py`](../src/rl/macphy/abreu_modified.py) | Modelo de enlace e avaliação MAC/PHY multi-caminho. |
| [`start.sh`](../src/rl/macphy/start.sh) | Launcher de currículo automático ou bloco manual. |
| [`requirements.txt`](../src/rl/macphy/requirements.txt) | Manifesto independente de dependências fixadas. |
| [`README.md`](../src/rl/macphy/README.md) | Guia do diretório HAPPO/MAC-PHY. |

### Instalação e execução do HAPPO

Na raiz do repositório, instale as dependências em ambiente dedicado:

```bash
python3 -m venv src/rl/macphy/.venv
src/rl/macphy/.venv/bin/python -m pip install -r src/rl/macphy/requirements.txt
```

Consulte os argumentos sem iniciar treino:

```bash
src/rl/macphy/.venv/bin/python src/rl/macphy/5GMLMA_MA.py --help
```

O currículo automático do launcher é longo e roda em segundo plano por padrão:

```bash
bash src/rl/macphy/start.sh
```

Para executar um bloco manual em primeiro plano:

```bash
CURRICULUM=0 BLOCK_FOREGROUND=1 bash src/rl/macphy/start.sh
```

O launcher define defaults de workers, timesteps, avaliação, zonas, rádio e MAC. Registre qualquer variável de ambiente que os altere. Na chamada direta ao treinador, configure explicitamente `--results-root` e `--best-model-dir` para evitar espalhar outputs; o default da CLI não é o mesmo do launcher.

## Ferramenta Optuna preservada

O anexo incluiu `optuna_2stages_outside.py`, originalmente voltado a uma API antiga de RLlib/PPO, opções que não existem no HAPPO nativo e arquivos de saída Ray/Tune `progress.csv`. Por isso ela foi guardada em [`src/rl/macphy/legacy/`](../src/rl/macphy/legacy/) e **não é compatível com o treinador HAPPO incluído**. Não a use para HPO deste modelo sem reescrever e validar a integração com a CLI e as métricas nativas em JSON/TensorBoard.

## Comparação experimental e checkpoints

Os dois ambientes diferem em espaços de ação/observação, recompensas, métricas, algoritmo e pressupostos físicos. Não são checkpoints compatíveis nem permitem transferência direta de pesos. Para comparação, controle ou reporte cenário, geometria, tráfego, configurações de rádio/MAC, zonas, inicialização, seeds, orçamento computacional e protocolo de avaliação. Apresente métricas comuns e também métricas nativas separadamente.
