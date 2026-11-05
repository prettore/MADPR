#!/bin/bash
set -euo pipefail

# ============================================================
# start_fases_integrado.sh
#
# Modos de execução:
#   1) Currículo automático, padrão:
#      ./start.sh
#
#   2) Bloco/manual sem currículo:
#      CURRICULUM=0 ./start.sh
#
#   3) Bloco/manual em primeiro plano:
#      CURRICULUM=0 BLOCK_FOREGROUND=1 ./start.sh
#
# O currículo chama este mesmo script internamente com CURRICULUM=0.
# Assim, não é necessário usar curriculum_manager.py separado.
# ============================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"

ulimit -n 1048576 || true

export PYTHONFAULTHANDLER=1
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}

export COMPILE_MODELS=${COMPILE_MODELS:-false}
export AMP=${AMP:-auto}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

SCRIPT="${SCRIPT:-$PROJECT_ROOT/5GMLMA_MA.py}"
CURRICULUM="${CURRICULUM:-1}"
BLOCK_FOREGROUND="${BLOCK_FOREGROUND:-0}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
LATEST_METRICS_PATH="${LATEST_METRICS_PATH:-$RESULTS_ROOT/latest_metrics.json}"
CURRICULUM_STATE_FILE="${CURRICULUM_STATE_FILE:-$PROJECT_ROOT/curriculum_state.json}"

# ============================================================
# MODO CURRÍCULO AUTOMÁTICO
# ============================================================
if [ "$CURRICULUM" = "1" ]; then
  # --- Execução em segundo plano (sobrevive a fechar o terminal/SSH) ---
  # Por padrão (RUN_BACKGROUND=1), re-lança o currículo inteiro destacado da
  # sessão via setsid+nohup, gravando log e PID. Para rodar em primeiro plano
  # (ver o output ao vivo), use RUN_BACKGROUND=0 ./start.sh.
  # A variável-guarda _CURRICULUM_DETACHED evita recursão infinita.
  CURRICULUM_LOG_FILE="${CURRICULUM_LOG_FILE:-./curriculum.log}"
  CURRICULUM_PID_FILE="${CURRICULUM_PID_FILE:-./curriculum.pid}"
  if [ "${RUN_BACKGROUND:-1}" = "1" ] && [ "${_CURRICULUM_DETACHED:-0}" != "1" ]; then
    # impede dois currículos simultâneos
    if [ -f "$CURRICULUM_PID_FILE" ]; then
      OLD_PID="$(cat "$CURRICULUM_PID_FILE" 2>/dev/null || true)"
      if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
        echo "Currículo já está em execução (PID: $OLD_PID). Abortei."
        exit 1
      else
        rm -f "$CURRICULUM_PID_FILE"
      fi
    fi
    echo "[START] Iniciando currículo em segundo plano (setsid+nohup)."
    _CURRICULUM_DETACHED=1 setsid nohup bash "$START_SCRIPT_PATH" > "$CURRICULUM_LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "$CURRICULUM_PID_FILE"
    echo "Currículo iniciado. PID: $BG_PID"
    echo "Log: $CURRICULUM_LOG_FILE"
    echo "PID salvo em: $CURRICULUM_PID_FILE"
    echo "Acompanhe com: tail -f $CURRICULUM_LOG_FILE"
    echo "Pare com: kill \$(cat $CURRICULUM_PID_FILE)"
    exit 0
  fi

  export PROJECT_ROOT PYTHON_BIN RESULTS_ROOT LATEST_METRICS_PATH CURRICULUM_STATE_FILE
  export SEED_MODEL="${USE_MODEL:-}"
  export START_SCRIPT="$START_SCRIPT_PATH"

  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Phase:
    name: str
    reliability_target: float
    delivery_ratio_target: float
    prune_delivery_min: float
    prune_rel_min: float
    carrier_sense_range_m: float
    advance_path_exists: float
    advance_delivery_ratio: float
    advance_reliability_block: float
    max_latency_ms: float = 300.0
    max_blocks: int = 30
    patience_to_advance: int = 5      # nº de avaliações (sobre média móvel) p/ avançar
    advance_window: int = 5           # janela da média móvel do critério de avanço
    qos_frac_of_baseline: float = 0.0 # meta de qos = min(abs, frac*baseline_pré-zona); auto-calibra
    min_qos_target_met: float = 0.0
    max_bad_deactivation: float = 0.10
    # ---- ZONAS PROIBIDAS (currículo) ----
    # p_sem_zona = fração de episódios SEM zona. Começa alto para preservar a
    # competência de KPI já aprendida (antídoto ao esquecimento catastrófico) e
    # cai gradualmente. Zonas são sorteadas por episódio (single_fixed=0), o que
    # força a política a LER o mapa de zonas em vez de decorar uma rota.
    zonas_count: int = 0
    zona_w: int = 5
    zona_h: int = 5
    p_sem_zona: float = 1.0
    zona_single_fixed: int = 0
    p_zone_violation_seed: float = 0.0   # frac. de resets que JÁ NASCEM violando (evacuação)
    min_zone_clear: float = 0.0          # fração mínima de episódios com zona EVACUADA p/ avançar
    zone_success_suppression: float = 1.0  # supressao dos bonus de sucesso enquanto violar (1=off)
    zone_violation_abs_penalty: float = 0.0  # custo ABSOLUTO por passo de violar zona (fora do potencial).
                                            # Cresce ao longo do curriculo: e o unico termo capaz de
                                            # tornar "atravessar com QoS" pior que "contornar".


PHASES: List[Phase] = [
    # Currículo evolutivo de confiabilidade REMOVIDO.
    # Antes havia 4 fases subindo reliability_target 0.70 -> 0.85 -> 0.95 -> 0.995.
    # Agora o reset parte de uma topologia JÁ VIÁVEL (warm start), então não há
    # motivo para começar com alvos baixos: o treino usa os KPIs finais desde o
    # início e o agente foca em manter/aprimorar os KPIs e MINIMIZAR drones.
    # ================= CURRÍCULO DE ZONAS PROIBIDAS =================
    # Parte de um modelo que JÁ atinge os KPIs sem zonas (use USE_MODEL apontando
    # para o best do run sem-zona). A dificuldade cresce pela FREQUÊNCIA e pelo
    # TAMANHO das zonas — nunca por "ligar/desligar" de repente, o que causaria
    # mudança abrupta de distribuição e colapso do crítico.
    #
    # O aprendizado real aqui é de POSICIONAMENTO: o hard mask já IMPEDE arestas
    # que cruzam zona, então a política não precisa aprender a evitá-las — ela
    # precisa aprender a posicionar drones de modo que EXISTA rota contornando.
    Phase(
        name="zona_introducao",      # 30% dos episódios com 1 zona pequena
        reliability_target=0.995,
        delivery_ratio_target=0.999,
        prune_delivery_min=0.90,
        prune_rel_min=0.80,
        carrier_sense_range_m=200.0,
        advance_path_exists=0.98,
        advance_delivery_ratio=0.999,
        advance_reliability_block=0.995,
        min_qos_target_met=0.35,     # meta menor: com zona o problema é mais duro
        max_blocks=15,
        zonas_count=1, zona_w=8, zona_h=8, p_sem_zona=0.7, p_zone_violation_seed=0.3, min_zone_clear=0.50,
        patience_to_advance=2, advance_window=5, qos_frac_of_baseline=0.60,
        zone_success_suppression=0.70, zone_violation_abs_penalty=2.0,
    ),
    Phase(
        name="zona_regime",          # 70% dos episódios com 1 zona média
        reliability_target=0.995,
        delivery_ratio_target=0.999,
        prune_delivery_min=0.90,
        prune_rel_min=0.80,
        carrier_sense_range_m=200.0,
        advance_path_exists=0.98,
        advance_delivery_ratio=0.999,
        advance_reliability_block=0.995,
        min_qos_target_met=0.40,
        max_blocks=20,
        zonas_count=1, zona_w=12, zona_h=12, p_sem_zona=0.3, p_zone_violation_seed=0.5, min_zone_clear=0.65,
        patience_to_advance=2, advance_window=5, qos_frac_of_baseline=0.70,
        zone_success_suppression=0.50, zone_violation_abs_penalty=8.0,
    ),
    Phase(
        name="zona_dificil",         # 90% com 2 zonas grandes (regime alvo)
        reliability_target=0.995,
        delivery_ratio_target=0.999,
        prune_delivery_min=0.90,
        prune_rel_min=0.80,
        carrier_sense_range_m=200.0,
        advance_path_exists=0.98,
        advance_delivery_ratio=0.999,
        advance_reliability_block=0.995,
        min_qos_target_met=0.45,
        max_blocks=40,
        zonas_count=2, zona_w=15, zona_h=15, p_sem_zona=0.1, p_zone_violation_seed=0.6,
        min_zone_clear=0.80, patience_to_advance=3, advance_window=6, qos_frac_of_baseline=0.80,
        zone_success_suppression=0.30, zone_violation_abs_penalty=12.0,
    ),
    # Fase legada (sem zonas). Mantida para reproduzir o run baseline:
    # rode com CURRICULUM_START_PHASE=3 se quiser só ela.
    Phase(
        name="alvo_final_unico",
        reliability_target=0.995,
        delivery_ratio_target=0.999,
        # Viabilidade (campo fixo 2km, física real ~2.4GHz, MAX_DRONES=64):
        # caso lado-a-lado fecha com ~36 drones; diagonal ~2.8km fica no limite
        # de 64. Se estagnar na diagonal, relaxe para 0.98/0.98 ou restrinja A/B.
        prune_delivery_min=0.90,
        prune_rel_min=0.80,
        # CS afeta SOMENTE contenção/colisão no Abreu (o alcance de comunicação
        # vem da física do rádio em _prob_link).
        carrier_sense_range_m=120.0,
        advance_path_exists=0.98,
        advance_delivery_ratio=0.999,
        advance_reliability_block=0.995,
        min_qos_target_met=0.50,
        max_blocks=200,
        patience_to_advance=12,
    ),
]


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).expanduser().resolve()
START_SCRIPT = Path(os.environ.get("START_SCRIPT", "./start.sh")).expanduser().resolve()
STATE_FILE = Path(os.environ.get("CURRICULUM_STATE_FILE", PROJECT_ROOT / "curriculum_state.json")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "results")).expanduser().resolve()
LATEST_METRICS_PATH = Path(os.environ.get("LATEST_METRICS_PATH", RESULTS_ROOT / "latest_metrics.json")).expanduser().resolve()

BLOCK_TIMESTEPS = env_int("BLOCK_TIMESTEPS", 2_000_000)
# Nº de blocos de estagnação consecutivos antes de PARAR o treino para ajuste
# manual. Configurável via env. Defina 0 (ou negativo) para DESABILITAR a parada
# por estagnação — o treino segue indefinidamente (útil para deixar mudanças
# recentes maturarem além do ponto em que o currículo normalmente pararia).
STALL_PATIENCE_BLOCKS = env_int("STALL_PATIENCE_BLOCKS", 12)
MIN_IMPROVEMENT_DELTA = env_float("MIN_IMPROVEMENT_DELTA", 1e-3)


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "phase_idx": 0,
        "global_block_idx": 0,
        "phase_block_idx": 0,
        "consecutive_success": 0,
        "resume": 0,
        "best_phase_score": -1e18,
        "stall_blocks": 0,
        "baseline_qos": None,   # QoS pré-zona (anti-esquecimento)
        "history": [],
        "completed": False,
    }


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def read_metrics() -> Dict[str, Any]:
    if not LATEST_METRICS_PATH.exists():
        return {}
    for _ in range(5):
        try:
            return json.loads(LATEST_METRICS_PATH.read_text())
        except Exception:
            time.sleep(0.5)
    return {}


def metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def phase_score(metrics: Dict[str, Any], phase: Phase) -> float:
    # NOTA: delivery/rel entram como PROGRESSO (0..1) rumo ao alvo, não como
    # gate — as médias nunca chegam a 0.995/0.999 (distribuição bimodal).
    path = min(metric(metrics, "path_exists_mean") / max(1e-12, phase.advance_path_exists), 1.0)
    delivery = min(metric(metrics, "delivery_ratio_mean") / max(1e-12, phase.advance_delivery_ratio), 1.0)
    rel = min(metric(metrics, "reliability_block_mean") / max(1e-12, phase.advance_reliability_block), 1.0)
    zone_bonus = 0.0
    if phase.zonas_count > 0:
        # progresso de evacuação entra no score (detecta estagnação na zona)
        zone_bonus = 3.0 * metric(metrics, "zone_clear_mean", 0.0)
    if phase.min_qos_target_met > 0:
        qos = min(metric(metrics, "qos_target_met_mean") / max(1e-12, phase.min_qos_target_met), 1.0)
    else:
        qos = metric(metrics, "qos_target_met_mean")
    latency = 1.0
    lat_ms = metric(metrics, "latency_ms_mean", 0.0)
    if lat_ms > 0.0:
        latency = max(0.0, min(1.0, 1.0 - lat_ms / max(1e-12, phase.max_latency_ms)))
    active_pen = 0.002 * max(0.0, metric(metrics, "active_drones_mean", 0.0))
    return 3.0 * path + 4.0 * delivery + 4.0 * rel + 2.0 * qos + 1.0 * latency + zone_bonus - active_pen


def _smoothed_metric(state: Dict[str, Any], key: str, window: int, default: float = 0.0) -> float:
    """Média dos últimos `window` blocos para `key` (métrica de avanço).

    Motivo: qos_target_met_mean é uma FRAÇÃO binária sobre ~100 episódios; perto
    de p=0.35 seu erro-padrão entre avaliações é ~0.05 e a distribuição bimodal
    (sombra) amplifica a serrilha. Exigir N blocos CONSECUTIVOS acima do limiar
    (esquema antigo) raramente fecha o streak — a oscilação o reinicia antes.
    Comparar a MÉDIA MÓVEL contra o limiar é robusto: a fase 'zona_introducao'
    atingiu competência (média móvel de qos cruzou 0.35) mas nunca teve 3
    avaliações seguidas acima, e esgotou max_blocks sem avançar."""
    hist = state.get("history", [])
    vals = []
    for ev in reversed(hist):
        m = ev.get("metrics", {})
        if key in m:
            vals.append(float(m[key]))
        if len(vals) >= window:
            break
    if not vals:
        return default
    return sum(vals) / len(vals)


def should_advance(metrics: Dict[str, Any], phase: Phase, state: Dict[str, Any] | None = None) -> tuple[bool, Dict[str, bool]]:
    """Critério de avanço.

    ATENÇÃO — correção importante: o critério ORIGINAL exigia
    reliability_block_mean >= 0.995 e delivery_ratio_mean >= 0.999, ou seja
    MÉDIAS sobre episódios. Isso é IMPOSSÍVEL neste ambiente: a distribuição é
    BIMODAL (sombreamento log-normal faz ~metade dos episódios nascer em fade
    profundo), e mesmo o melhor run tem rel MÉDIA ~0.55 embora atinja os KPIs
    nos episódios viáveis. Com aquele critério o currículo NUNCA avançaria de
    fase — ficaria preso até esgotar max_blocks.

    O critério correto usa qos_target_met_mean: a FRAÇÃO de episódios que fecham
    os três KPIs simultaneamente (é a métrica que já integra rel+vazão+latência
    por episódio). Somam-se as metas de ZONA quando a fase tem zonas.
    """
    win = int(getattr(phase, "advance_window", 5))

    # Meta de QoS efetiva: máx(meta absoluta da fase, fração do baseline pré-zona).
    # Se o baseline sem-zona era qos~0.45 e a fase pede 60% disso, a meta vira
    # 0.27 — alcançável — em vez de um 0.35 fixo que a fase talvez nunca atinja
    # porque a introdução de zonas naturalmente reduz o QoS médio.
    qos_meta = _qos_meta_for(phase, state)

    def M(key, default=0.0):
        # média móvel dos últimos `win` blocos (robusta à serrilha); cai para o
        # valor instantâneo se não houver histórico (1º bloco após retomada).
        if state is not None:
            return _smoothed_metric(state, key, win, metric(metrics, key, default))
        return metric(metrics, key, default)

    checks = {
        "path_exists": M("path_exists_mean") >= phase.advance_path_exists,
        "qos_target_met": M("qos_target_met_mean") >= qos_meta,
        "bad_deactivation": M("bad_deactivation_mean", 0.0) <= phase.max_bad_deactivation,
    }
    # Metas de ZONA: só entram nas fases que têm zonas. Exigem que a política
    # esteja de fato evacuando/contornando, não só mantendo KPI ignorando-as.
    if phase.zonas_count > 0 and phase.min_zone_clear > 0.0:
        checks["zone_clear"] = M("zone_clear_mean", 0.0) >= phase.min_zone_clear
    return all(checks.values()), checks


# Piso de competência: o QoS medido ANTES de introduzir zonas (fase 0). Serve de
# referência para detectar ESQUECIMENTO CATASTRÓFICO — a política aprendendo a
# contornar zonas ao custo de desaprender a atingir os KPIs (competência que
# custou dezenas de milhões de steps). Gravado no state no 1º bloco.
FORGET_TOLERANCE = env_float("FORGET_TOLERANCE", 0.15)  # queda relativa tolerada
# Blocos de carência após ENTRAR numa fase, antes de a guarda voltar a valer.
# A política precisa de tempo para se adaptar à nova mistura antes de ser
# julgada por ela.
FORGET_GRACE_BLOCKS = int(env_float("FORGET_GRACE_BLOCKS", 2))


def _smoothed_metric_phase(state: Dict[str, Any], key: str, window: int,
                           phase_idx: int, default: float = 0.0) -> float:
    """Como _smoothed_metric, mas só considera blocos DA MESMA FASE.

    O history é compartilhado entre fases; misturar blocos de fases com
    misturas de zona diferentes produz uma média sem significado.
    """
    vals = []
    for ev in reversed(state.get("history", [])):
        if int(ev.get("phase_idx", -1)) != int(phase_idx):
            continue
        m = ev.get("metrics", {})
        if key in m:
            vals.append(float(m[key]))
        if len(vals) >= window:
            break
    if not vals:
        return default
    return sum(vals) / len(vals)


def _qos_meta_for(phase: "Phase", state: Dict[str, Any] | None) -> float:
    """Meta de QoS EFETIVA da fase: máx(meta absoluta, fração do baseline).

    Extraída de should_advance para que a guarda de esquecimento use exatamente
    o mesmo número que o gate de avanço — se divergirem, a guarda pode parar um
    run que o gate deixaria passar (foi o que aconteceu: guarda parou com
    qos=0.580 enquanto o gate exigia apenas 0.400).
    """
    qos_meta = float(phase.min_qos_target_met)
    if state is not None:
        base = float(state.get("baseline_qos", 0.0) or 0.0)
        frac = float(getattr(phase, "qos_frac_of_baseline", 0.0))
        if base > 0.0 and frac > 0.0:
            qos_meta = min(qos_meta, base * frac) if qos_meta > 0 else base * frac
    return float(qos_meta)


def check_forgetting(metrics: Dict[str, Any], state: Dict[str, Any],
                     phase: "Phase") -> tuple[bool, str]:
    """Retorna (esqueceu?, mensagem).

    Detecta ESQUECIMENTO: a política perdendo competência de KPI DENTRO de uma
    fase. Três correções sobre a versão anterior, que deu dois falsos positivos
    seguidos e parou o currículo à toa:

    1) BASELINE POR FASE. Antes, `baseline_qos` era gravado uma única vez, no
       1º bloco do currículo, e comparado com todas as fases seguintes. Mas
       cada fase muda a MISTURA (p_sem_zona 0.7 -> 0.3 -> 0.1) e o tamanho da
       zona (8x8 -> 12x12 -> 15x15). Uma política de competência IDÊNTICA
       pontua muito menos em zona_regime que em zona_introducao — só pela
       mistura, 0.530 vira ~0.24. Comparar entre fases é comparar coisas
       diferentes, e disparava esquecimento onde não havia.
    2) MÉDIA MÓVEL, não valor instantâneo. qos_target_met_mean é fração
       binária sobre ~100 episódios (erro-padrão ~0.05, bimodal). O
       should_advance já usa M(); a GUARDA usava o valor cru — ou seja, a
       trava de segurança era mais ruidosa que o critério que ela protege.
    3) CARÊNCIA após trocar de fase: a política precisa de blocos para se
       adaptar à nova mistura antes de ser julgada por ela.

    A guarda continua detectando o caso real que motivou sua criação: queda
    sustentada de qos DENTRO de uma mesma fase, medida contra o baseline
    daquela fase.
    """
    if FORGET_TOLERANCE <= 0.0:
        return False, ""   # guarda desabilitada
    phase_idx = int(state.get("phase_idx", 0))
    bases = state.get("baseline_qos_by_phase", {}) or {}
    base = float(bases.get(str(phase_idx), -1.0))
    if base <= 0.0:
        return False, ""   # fase ainda sem baseline registrado

    # Carência: conta blocos JÁ CONCLUÍDOS nesta fase.
    blocks_in_phase = sum(1 for ev in state.get("history", [])
                          if int(ev.get("phase_idx", -1)) == phase_idx)
    if blocks_in_phase <= FORGET_GRACE_BLOCKS:
        return False, ""

    win = max(1, int(getattr(phase, "advance_window", 5)))
    cur = _smoothed_metric_phase(state, "qos_target_met_mean", win, phase_idx,
                                 metric(metrics, "qos_target_met_mean", 0.0))
    # PISO ANCORADO NO GATE, não no pico histórico.
    # O baseline da fase é registrado no bloco 0 — quando a política está no
    # PICO de KPI e com ZERO competência de zona. Qualquer progresso rumo ao
    # objetivo real (limpar zona mantendo KPI) troca um pouco de qos por
    # zone_clear e, para um piso puramente relativo, isso é indistinguível de
    # esquecimento. Foi o que ocorreu: a guarda parou o run com qos=0.580
    # (média de 5 blocos) enquanto o gate de avanço exigia apenas 0.400 — 45%
    # de folga — e o único obstáculo real era zone_clear.
    #
    # Esquecimento só IMPORTA se ameaçar o avanço. Logo o piso nunca desce
    # abaixo de qos_meta, mas também nunca sobe acima dele por causa de um
    # baseline otimista: usamos o MENOR entre os dois critérios.
    rel_floor = base * (1.0 - FORGET_TOLERANCE)
    meta = _qos_meta_for(phase, state)
    floor = min(rel_floor, meta) if meta > 0.0 else rel_floor
    if cur < floor:
        return True, (
            f"qos(média móvel {win} blocos da fase)={cur:.3f} < piso={floor:.3f} "
            f"(baseline da fase '{phase.name}'={base:.3f}, "
            f"tolerância={FORGET_TOLERANCE:.0%}, qos_meta do gate={meta:.3f})"
        )
    return False, ""


def build_env_for_phase(phase: Phase, state: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env["CURRICULUM"] = "0"
    env["BLOCK_FOREGROUND"] = "1"
    env["RUN_BACKGROUND"] = "0"
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["RESULTS_ROOT"] = str(RESULTS_ROOT)
    env["LATEST_METRICS_PATH"] = str(LATEST_METRICS_PATH)
    env["TIMESTEPS"] = str(BLOCK_TIMESTEPS)
    env["TIMESTEPS_MODE"] = "incremental"
    env["RESUME"] = str(int(state.get("resume", 0)))
    if int(state.get("resume", 0)):
        env["USE_MODEL"] = "continue"   # blocos seguintes: retoma o checkpoint do currículo
        env["RESTORE_MODEL_PATH"] = ""
    else:
        # 1º bloco: parte do MODELO SEMENTE passado na linha de comando
        # (ex.: ./ppo_best_model_sem_zona/best_model.pt). Sem isto, o currículo
        # de zonas começaria com pesos ALEATÓRIOS, desperdiçando o modelo
        # sem-zona e a fase construtiva de ~15M steps.
        seed_model = str(os.environ.get("SEED_MODEL", "") or "")
        # IMPORTANTE: o modelo semente vai em RESTORE_MODEL_PATH (o trainer
        # ignora caminhos em USE_MODEL — só aceita "continue"/"resume").
        env["USE_MODEL"] = ""
        env["RESTORE_MODEL_PATH"] = seed_model
        if seed_model:
            print(f"[CURRICULUM] 1º bloco parte do modelo semente: {seed_model}", flush=True)
        else:
            print("[CURRICULUM][AVISO] Nenhum USE_MODEL definido — currículo começa do ZERO "
                  "(passe USE_MODEL=<best sem-zona> para fine-tuning).", flush=True)

    env["RELIABILITY_TARGET"] = str(phase.reliability_target)
    env["DELIVERY_RATIO_TARGET"] = str(phase.delivery_ratio_target)
    env["PRUNE_DELIVERY_MIN"] = str(phase.prune_delivery_min)
    env["PRUNE_REL_MIN"] = str(phase.prune_rel_min)
    env["CARRIER_SENSE_RANGE_M"] = str(phase.carrier_sense_range_m)
    env["MAX_DELAY_MS"] = str(phase.max_latency_ms)

    # ---- ZONAS PROIBIDAS (por fase) ----
    env["ZONAS_COUNT"] = str(int(phase.zonas_count))
    env["ZONA_W"] = str(int(phase.zona_w))
    env["ZONA_H"] = str(int(phase.zona_h))
    env["P_SEM_ZONA"] = str(float(phase.p_sem_zona))
    env["ZONA_SINGLE_FIXED"] = str(int(phase.zona_single_fixed))
    env["P_ZONE_VIOLATION_SEED"] = str(float(phase.p_zone_violation_seed))
    env["ZONE_SUCCESS_SUPPRESSION"] = str(float(phase.zone_success_suppression))
    env["ZONE_VIOLATION_ABS_PENALTY"] = str(float(phase.zone_violation_abs_penalty))

    # A receita do run que CONVERGIU (não alterar junto com o currículo de zonas:
    # uma variável por experimento). Só define se o usuário não sobrescreveu.
    env.setdefault("LATENCY_HEADROOM_PENALTY", "0")
    env.setdefault("CONVERGE_ON_STATE", "0")
    env.setdefault("CONVERGE_PATIENCE", "120")

    env["CURRICULUM_PHASE_NAME"] = phase.name
    env["CURRICULUM_PHASE_IDX"] = str(int(state["phase_idx"]))
    env["CURRICULUM_BLOCK_IDX"] = str(int(state["global_block_idx"]))

    env.setdefault("MAX_DRONES", "30")
    env.setdefault("MAX_OUT_CONNECTIONS", "4")
    env.setdefault("ABREU_MAX_PATHS", "8")
    env.setdefault("RUN_NAME", "connectivity_curriculum_pruning")
    env.setdefault("INSTANCE_TAG", "A")
    env.setdefault("DEVICE", "auto")
    return env


def run_block(phase: Phase, state: Dict[str, Any]) -> None:
    if not START_SCRIPT.exists():
        raise FileNotFoundError(f"start.sh integrado não encontrado: {START_SCRIPT}")
    env = build_env_for_phase(phase, state)
    print(
        f"[CURRICULUM] Rodando {phase.name} | bloco_global={state['global_block_idx']} "
        f"bloco_fase={state['phase_block_idx']} | resume={env['RESUME']} | steps={BLOCK_TIMESTEPS}",
        flush=True,
    )
    subprocess.run(["bash", str(START_SCRIPT)], cwd=str(PROJECT_ROOT), env=env, check=True)


def main() -> None:
    state = load_state()
    if bool(state.get("completed", False)):
        print("[CURRICULUM] Currículo já concluído. Apague curriculum_state.json para reiniciar.", flush=True)
        return

    # RETOMADA COM CRITÉRIO NOVO: se a fase corrente esgotou max_blocks (ou está
    # perto), reavalia AGORA com a média móvel do histórico salvo. Isso destrava
    # fases que tinham competência mas travaram no esquema consecutivo antigo
    # (ex.: zona_introducao esgotou 15 blocos com qos médio ~0.35 mas sem 3
    # avaliações seguidas acima). Assim a retomada não desperdiça blocos.
    if state.get("history") and int(state["phase_idx"]) < len(PHASES):
        _ph = PHASES[int(state["phase_idx"])]
        _m = state["history"][-1].get("metrics", {})
        _ok, _chk = should_advance(_m, _ph, state)
        _exhausted = int(state.get("phase_block_idx", 0)) >= int(_ph.max_blocks)
        if _ok and (_exhausted or int(state.get("phase_block_idx", 0)) >= _ph.patience_to_advance):
            print(f"[CURRICULUM][RETOMADA] Fase '{_ph.name}' JÁ satisfaz o critério novo "
                  f"(média móvel): {_chk}. Avançando sem reprocessar blocos.", flush=True)
            state["phase_idx"] = int(state["phase_idx"]) + 1
            state["phase_block_idx"] = 0
            state["consecutive_success"] = 0
            state["stall_blocks"] = 0
            state["best_phase_score"] = -1e18
            save_state(state)
        elif _exhausted:
            # esgotou mas AINDA não satisfaz: reabre a fase para mais blocos com
            # o critério novo (em vez de ficar presa "concluída por desistência").
            print(f"[CURRICULUM][RETOMADA] Fase '{_ph.name}' esgotou max_blocks sem satisfazer "
                  f"o critério novo: {_chk}. Reabrindo p/ +{_ph.max_blocks} blocos.", flush=True)
            state["phase_block_idx"] = 0
            state["stall_blocks"] = 0
            save_state(state)

    while int(state["phase_idx"]) < len(PHASES):
        phase = PHASES[int(state["phase_idx"])]
        if int(state.get("phase_block_idx", 0)) == 0:
            state["consecutive_success"] = 0
            state["best_phase_score"] = -1e18
            state["stall_blocks"] = 0
            save_state(state)

        while int(state.get("phase_block_idx", 0)) < int(phase.max_blocks):
            run_block(phase, state)
            state["resume"] = 1
            state["global_block_idx"] = int(state.get("global_block_idx", 0)) + 1
            state["phase_block_idx"] = int(state.get("phase_block_idx", 0)) + 1

            metrics = read_metrics()
            score = phase_score(metrics, phase)
            ok, checks = should_advance(metrics, phase, state)

            # Baseline de competência POR FASE: registra o QoS do primeiro bloco
            # DESTA fase. Cada fase tem mistura própria (p_sem_zona, tamanho da
            # zona), então cada uma precisa da sua própria referência — um
            # baseline global comparava regimes incomparáveis.
            _bases = state.setdefault("baseline_qos_by_phase", {})
            _pk = str(int(state["phase_idx"]))
            if _pk not in _bases:
                _bases[_pk] = float(metric(metrics, "qos_target_met_mean", 0.0))
                print(f"[CURRICULUM] baseline_qos da fase '{phase.name}' = {_bases[_pk]:.3f}", flush=True)
            # compat: mantém a chave antiga (só informativa) para logs/análises
            if state.get("baseline_qos") is None:
                state["baseline_qos"] = float(_bases[_pk])

            forgot, why = check_forgetting(metrics, state, phase)
            if forgot:
                print(
                    f"[CURRICULUM][ESQUECIMENTO] {why}\n"
                    f"  A política está perdendo a competência de KPI ao aprender zonas.\n"
                    f"  Ação sugerida: AUMENTAR p_sem_zona da fase '{phase.name}' "
                    f"(mais episódios sem zona preservam o aprendizado anterior)\n"
                    f"  ou reduzir LR. Currículo PARADO para ajuste manual.",
                    flush=True,
                )
                save_state(state)
                return

            if score > float(state.get("best_phase_score", -1e18)) + MIN_IMPROVEMENT_DELTA:
                state["best_phase_score"] = float(score)
                state["stall_blocks"] = 0
            else:
                state["stall_blocks"] = int(state.get("stall_blocks", 0)) + 1

            if ok:
                state["consecutive_success"] = int(state.get("consecutive_success", 0)) + 1
            else:
                state["consecutive_success"] = 0

            event = {
                "phase_idx": int(state["phase_idx"]),
                "phase_name": phase.name,
                "global_block_idx": int(state["global_block_idx"]),
                "phase_block_idx": int(state["phase_block_idx"]),
                "score": float(score),
                "checks": checks,
                "consecutive_success": int(state["consecutive_success"]),
                "stall_blocks": int(state["stall_blocks"]),
                "metrics": metrics,
                "phase": asdict(phase),
            }
            state.setdefault("history", []).append(event)
            state["history"] = state["history"][-500:]
            save_state(state)

            print(
                "[CURRICULUM] Métricas: "
                f"path={metric(metrics, 'path_exists_mean'):.3f} "
                f"delivery={metric(metrics, 'delivery_ratio_mean'):.4f} "
                f"rel_block={metric(metrics, 'reliability_block_mean'):.4f} "
                f"qos={metric(metrics, 'qos_target_met_mean'):.3f} "
                f"active={metric(metrics, 'active_drones_mean'):.2f} "
                f"bad_deact={metric(metrics, 'bad_deactivation_mean'):.3f} "
                f"score={score:.4f} "
                f"ok={ok} patience={state['consecutive_success']}/{phase.patience_to_advance} "
                f"stall={state['stall_blocks']}/{STALL_PATIENCE_BLOCKS if STALL_PATIENCE_BLOCKS > 0 else 'desabilitado'}",
                flush=True,
            )

            if int(state["consecutive_success"]) >= int(phase.patience_to_advance):
                print(f"[CURRICULUM] Avançando fase: {phase.name}", flush=True)
                state["phase_idx"] = int(state["phase_idx"]) + 1
                state["phase_block_idx"] = 0
                state["consecutive_success"] = 0
                state["best_phase_score"] = -1e18
                state["stall_blocks"] = 0
                save_state(state)
                break

            # Parada por estagnação: só ocorre se STALL_PATIENCE_BLOCKS > 0.
            # Defina STALL_PATIENCE_BLOCKS=0 (ou negativo) para DESABILITAR a
            # parada automática — nesse modo o treino NUNCA para por estagnação
            # e segue rodando blocos até o limite de timesteps/manual.
            if STALL_PATIENCE_BLOCKS > 0 and int(state["stall_blocks"]) >= STALL_PATIENCE_BLOCKS:
                print(
                    f"[CURRICULUM] Estagnação detectada na fase {phase.name}. "
                    "Interrompendo para ajuste manual dos parâmetros.",
                    flush=True,
                )
                save_state(state)
                return
        else:
            print(f"[CURRICULUM] Limite de blocos atingido na fase {phase.name}.", flush=True)
            save_state(state)
            return

    state["completed"] = True
    save_state(state)
    print("[CURRICULUM] Todas as fases foram concluídas.", flush=True)


if __name__ == "__main__":
    main()
PY
  exit $?
fi

# ============================================================
# MODO BLOCO / MANUAL
# ============================================================
NUM_WORKERS="${NUM_WORKERS:-128}"
# AJUSTE DE MEMÓRIA (importante):
# A observação cresce com MAX_DRONES^2 (duas matrizes N×N: adjacência e
# edge-cross). Com MAX_DRONES=64, o buffer de rollout do PPO é
#   ~ rollout_steps × num_workers × max_agents × (obs+mask) × 4 bytes.
# Com 128 workers × 1024 rollout isso passa de ~580 GB e estoura a RAM.
# 128 × 256 fica em ~73 GB, com folga confortável em 512 GB.
# Se quiser rollouts mais longos, reduza num_workers na mesma proporção
# (o produto workers×rollout é o que governa a memória).
ROLLOUT_STEPS="${ROLLOUT_STEPS:-512}"
TIMESTEPS="${TIMESTEPS:-2000000}"
TIMESTEPS_MODE="${TIMESTEPS_MODE:-incremental}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-8192}"
N_EPOCHS="${N_EPOCHS:-3}"
RUN_NAME="${RUN_NAME:-connectivity_curriculum_pruning}"
INSTANCE_TAG="${INSTANCE_TAG:-A}"
USE_MODEL="${USE_MODEL:-}"
# Caminho de um checkpoint para SEMEAR os pesos (fine-tuning). O trainer só
# entende USE_MODEL="continue"/"resume" (retoma o PRÓPRIO run); um CAMINHO em
# USE_MODEL era silenciosamente ignorado -> o treino começava do ZERO. O
# parâmetro correto do trainer é --restore-model-path.
RESTORE_MODEL_PATH="${RESTORE_MODEL_PATH:-}"
RESUME="${RESUME:-0}"
BEST_MODEL_DIR="${BEST_MODEL_DIR:-./ppo_best_model}"
# Métrica que define o "melhor" modelo salvo. Critério LEXICOGRÁFICO:
# QoS (eval_qos_target_met_mean) é primário e a recompensa é o desempate.
# Salva se o QoS melhora, OU se o QoS se mantém E a recompensa melhora.
# NUNCA salva se o QoS piora (bloqueia reward hacking: recompensa subindo com
# KPIs caindo não é capturada). Combina alinhamento ao objetivo (QoS) com
# progresso contínuo (recompensa de desempate quando o QoS empata).
BEST_METRIC="${BEST_METRIC:-lex_qos_reward}"
BEST_MODE="${BEST_MODE:-max}"
# delta mínimo para a recompensa (desempate) ser considerada melhora real.
BEST_MIN_DELTA="${BEST_MIN_DELTA:-0.5}"
CHECKPOINT_TARGET_TIMESTEPS="${CHECKPOINT_TARGET_TIMESTEPS:-2000000}"
EVAL_INTERVAL_UPDATES="${EVAL_INTERVAL_UPDATES:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_ENVS="${EVAL_ENVS:-25}"

LR_START="${LR_START:-7.253599353774804e-05}"
CRITIC_LR_START="${CRITIC_LR_START:-5.0e-05}"
CLIP_START="${CLIP_START:-0.1508396902041272}"
ENT_COEF="${ENT_COEF:-0.02}"
GAMMA="${GAMMA:-0.9958880495065485}"
GAE_LAMBDA="${GAE_LAMBDA:-0.9704784345025532}"
GRAD_CLIP="${GRAD_CLIP:-10.0}"
ALLOW_TF32="${ALLOW_TF32:-true}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
POLICY_ARCH="${POLICY_ARCH:-1024,512,256}"

EPISODE_HORIZON="${EPISODE_HORIZON:-1000}"
MASK_ACTIONS="${MASK_ACTIONS:-1}"
MAX_DRONES="${MAX_DRONES:-30}"
MAX_OUT_CONNECTIONS="${MAX_OUT_CONNECTIONS:-4}"
ZONAS_COUNT="${ZONAS_COUNT:-0}"
ZONA_W="${ZONA_W:-5}"
ZONA_H="${ZONA_H:-5}"
ZONA_SINGLE_FIXED="${ZONA_SINGLE_FIXED:-0}"
P_SEM_ZONA="${P_SEM_ZONA:-1.0}"
OBS_ZONE_MAP="${OBS_ZONE_MAP:-1}"
ZONE_MAP_SCALE="${ZONE_MAP_SCALE:-4}"
OBS_EDGE_CROSS="${OBS_EDGE_CROSS:-1}"
SEED_INITIAL_PATH="${SEED_INITIAL_PATH:-true}"
# Intensidade da aleatoriedade da semeadura (0=topologia fixa; >0 varia
# A/B, numero de caminhos e jitter por episodio, de forma reprodutivel e
# mantendo os KPIs viaveis). Comece em 0 e aumente gradualmente conforme o
# agente domina o caso base.
# Cenário inicial FIXO (seed_randomness=0): o critério de sucesso é atingir os
# KPIs de forma consistente, e o cenário inicial não precisa ser aleatório.
# Treinar com cenários muito variados (0.3) estava causando alta variância de
# desempenho (rel terminal de 0.40 a 0.99 entre episódios) — a política não
# generalizava sobre cenários. Com cenário fixo, ela só precisa resolver UM
# caso, o que é muito mais tratável e alinhado com o requisito real.
# FASE 2: variação moderada do cenário inicial (seed_randomness=0.15 por padrão). Varia
# A/B, número de caminhos e posições por episódio (reprodutível por seed). O
# cenário fixo (Fase 1) ficou trivial — o warm-start já atingia todos os KPIs,
# sem nada para a política aprender. Com variação, o warm-start deixa de ser
# perfeito e a política precisa aprender a ajustar a rede. Suba gradualmente
# (0.15 -> 0.3 -> 0.5 -> ...) conforme a política domina cada nível.
SEED_RANDOMNESS="${SEED_RANDOMNESS:-0.05}"
# Raio máximo (m) para criar enlace. Restringe o espaço de ação efetivo aos
# vizinhos plausíveis (~8 drones em média, em vez de 63), reduzindo ~87% as
# opções de criar-link. 400m dá margem sobre o alcance confiável (~300m) e os
# saltos da semeadura (~250-284m). 0 = sem restrição.
LINK_NEIGHBOR_RADIUS_M="${LINK_NEIGHBOR_RADIUS_M:-400.0}"
# FASE 2 — dificuldade mais leve: 2 caminhos, saltos-alvo ~140 m,
# com reparo geométrico moderado (0.25) para não deixar o warm-start perfeito.
# Ajuste para mudar a dificuldade:
#   - SEED_NUM_PATHS menor ou SEED_TARGET_HOP_M maior = mais difícil
#   - SEED_VIABILITY_SLACK controla o quanto a semeadura preserva saltos longos
SEED_NUM_PATHS="${SEED_NUM_PATHS:-2}"
SEED_TARGET_HOP_M="${SEED_TARGET_HOP_M:-140.0}"
SEED_VIABILITY_SLACK="${SEED_VIABILITY_SLACK:-1.15}"
# MODO LINHA MÍNIMA (warm-start construtivo) — LIGADO POR PADRÃO:
#   Semeia a MENOR linha reta A->B que chega PRÓXIMO de atender os KPIs
#   (analítico via Abreu: n_atende-1 hops; ex. 2km/phy100 -> 6 drones,
#   rel~0.74, lat~271ms). Demais drones DESATIVADOS e espalhados. A política
#   aprende a ATIVAR os necessários e criar rotas (redundantes ou não).
#   Para desligar e voltar à semeadura multi-caminho tradicional: SEED_LINE_MIN=0.
SEED_LINE_MIN="${SEED_LINE_MIN:-1}"
# Custo por drone ativo no potencial (ativar custa; só compensa se fechar KPI)
DRONE_POTENTIAL_COST="${DRONE_POTENTIAL_COST:-0.15}"
# Nascimento inteligente na ativação (reposiciona perto da linha média + 1 link)
ACTIVATE_SPAWN_NEAR_LINE="${ACTIVATE_SPAWN_NEAR_LINE:-1}"

ABREU_ENABLED="${ABREU_ENABLED:-1}"
ABREU_MAX_PATHS="${ABREU_MAX_PATHS:-8}"
TRAFFIC_RATE_MBPS="${TRAFFIC_RATE_MBPS:-10}"
FRAME_SIZE_BYTES="${FRAME_SIZE_BYTES:-1500}"
BUFFER_SIZE_PACKETS="${BUFFER_SIZE_PACKETS:-20}"
PHY_RATE_MBPS="${PHY_RATE_MBPS:-100}"
SLOT_TIME_US="${SLOT_TIME_US:-9}"
DIFS_US="${DIFS_US:-28}"
SIFS_US="${SIFS_US:-10}"
ACK_US="${ACK_US:-24}"
ACK_TIMEOUT_US="${ACK_TIMEOUT_US:-34}"
CW_MIN="${CW_MIN:-15}"
CW_MAX="${CW_MAX:-1023}"
MAX_RETRIES="${MAX_RETRIES:-6}"
CARRIER_SENSE_RANGE_M="${CARRIER_SENSE_RANGE_M:-200}"

# Alvos REAIS da tese (antes estavam em 0.70/0.90 — resquício do currículo de
# 4 fases já removido, que fazia a política parar em ~0.82 por já superar o
# alvo relaxado). Os KPIs da tese são confiabilidade >= 0.995 e entrega >= 0.999.
RELIABILITY_TARGET="${RELIABILITY_TARGET:-0.995}"
DELIVERY_RATIO_TARGET="${DELIVERY_RATIO_TARGET:-0.999}"
PRUNE_DELIVERY_MIN="${PRUNE_DELIVERY_MIN:-0.90}"
PRUNE_REL_MIN="${PRUNE_REL_MIN:-0.80}"
DRONE_ACTIVE_COST_BEFORE_QOS="${DRONE_ACTIVE_COST_BEFORE_QOS:-0.02}"
BAD_DEACTIVATION_PENALTY="${BAD_DEACTIVATION_PENALTY:-0.80}"
MAX_DELAY_MS="${MAX_DELAY_MS:-300.0}"
THROUGHPUT_MAX_MBPS="${THROUGHPUT_MAX_MBPS:-9600.0}"

# Recompensa híbrida / terminação por convergência. Mantido explícito no
# start.sh para que o experimento seja reprodutível, mesmo que o parser tenha
# defaults equivalentes.
USE_HYBRID_REWARD="${USE_HYBRID_REWARD:-true}"
QOS_MAINTAIN_BONUS="${QOS_MAINTAIN_BONUS:-6.0}"
CONVERGE_PATIENCE="${CONVERGE_PATIENCE:-120}"
CONVERGE_EPS="${CONVERGE_EPS:-0.01}"
# Receita do run que CONVERGIU (ver histórico: anti-farming por estado e a
# penalidade de folga de latência causaram regressão; patience=120 preserva o
# benefício sem destruir o currículo implícito de QoS).
LATENCY_HEADROOM_PENALTY="${LATENCY_HEADROOM_PENALTY:-0}"
ZONE_VIOLATION_PENALTY="${ZONE_VIOLATION_PENALTY:-6.0}"
ZONE_LINK_WEIGHT_MULT="${ZONE_LINK_WEIGHT_MULT:-3.5}"
ZONE_SUCCESS_SUPPRESSION="${ZONE_SUCCESS_SUPPRESSION:-1.0}"  # sobrescrito por fase abaixo
ZONE_VIOLATION_ABS_PENALTY="${ZONE_VIOLATION_ABS_PENALTY:-0.0}"  # sobrescrito por fase abaixo
LATENCY_FROM_LONGEST_ROUTE="${LATENCY_FROM_LONGEST_ROUTE:-0}"  # 1 = semantica legada (espaco livre)
P_ZONE_VIOLATION_SEED="${P_ZONE_VIOLATION_SEED:-0.0}"
CONVERGE_ON_STATE="${CONVERGE_ON_STATE:-0}"

CURRICULUM_PHASE_NAME="${CURRICULUM_PHASE_NAME:-manual_block}"
CURRICULUM_PHASE_IDX="${CURRICULUM_PHASE_IDX:--1}"
CURRICULUM_BLOCK_IDX="${CURRICULUM_BLOCK_IDX:--1}"
DEVICE="${DEVICE:-auto}"

PID_FILE="${PID_FILE:-./meu_programa.pid}"
LOG_FILE="${LOG_FILE:-./meu_programa.log}"

CMD=(
  "$PYTHON_BIN" -u "$SCRIPT"
  --num-workers "$NUM_WORKERS"
  --rollout-steps "$ROLLOUT_STEPS"
  --timesteps "$TIMESTEPS"
  --timesteps-mode "$TIMESTEPS_MODE"
  --eval-interval-updates "$EVAL_INTERVAL_UPDATES"
  --eval-episodes "$EVAL_EPISODES"
  --eval-envs "$EVAL_ENVS"
  --checkpoint-target-timesteps "$CHECKPOINT_TARGET_TIMESTEPS"
  --checkpoint-freq-updates 2
  --run-name "$RUN_NAME"
  --instance-tag "$INSTANCE_TAG"
  --results-root "$RESULTS_ROOT"
  --best-model-dir "$BEST_MODEL_DIR"
  --best-metric "$BEST_METRIC"
  --best-mode "$BEST_MODE"
  --best-min-delta "$BEST_MIN_DELTA"
  --resume "$RESUME"
  --use-model "$USE_MODEL"
  --restore-model-path "$RESTORE_MODEL_PATH"
  --policy-arch "$POLICY_ARCH"
  --lr-start "$LR_START"
  --critic-lr-start "$CRITIC_LR_START"
  --clip-start "$CLIP_START"
  --ent-coef "$ENT_COEF"
  --gamma "$GAMMA"
  --gae-lambda "$GAE_LAMBDA"
  --grad-clip "$GRAD_CLIP"
  --n-epochs "$N_EPOCHS"
  --minibatch-size "$MINIBATCH_SIZE"
  --amp "$AMP"
  --amp-dtype "$AMP_DTYPE"
  --compile-models "$COMPILE_MODELS"
  --allow-tf32 "$ALLOW_TF32"
  --episode-horizon "$EPISODE_HORIZON"
  --mask-actions "$MASK_ACTIONS"
  --max-drones "$MAX_DRONES"
  --max-out-connections "$MAX_OUT_CONNECTIONS"
  --zonas-count "$ZONAS_COUNT"
  --zona-w "$ZONA_W"
  --zona-h "$ZONA_H"
  --zona-single-fixed "$ZONA_SINGLE_FIXED"
  --p-sem-zona "$P_SEM_ZONA"
  --obs-zone-map "$OBS_ZONE_MAP"
  --zone-map-scale "$ZONE_MAP_SCALE"
  --obs-edge-cross "$OBS_EDGE_CROSS"
  --seed-initial-path "$SEED_INITIAL_PATH"
  --seed-randomness "$SEED_RANDOMNESS"
  --link-neighbor-radius-m "$LINK_NEIGHBOR_RADIUS_M"
  --seed-num-paths "$SEED_NUM_PATHS"
  --seed-target-hop-m "$SEED_TARGET_HOP_M"
  --seed-viability-slack "$SEED_VIABILITY_SLACK"
  --seed-line-min "$SEED_LINE_MIN"
  --drone-potential-cost "$DRONE_POTENTIAL_COST"
  --activate-spawn-near-line "$ACTIVATE_SPAWN_NEAR_LINE"
  --abreu-enabled "$ABREU_ENABLED"
  --abreu-max-paths "$ABREU_MAX_PATHS"
  --traffic-rate-mbps "$TRAFFIC_RATE_MBPS"
  --frame-size-bytes "$FRAME_SIZE_BYTES"
  --buffer-size-packets "$BUFFER_SIZE_PACKETS"
  --phy-rate-mbps "$PHY_RATE_MBPS"
  --slot-time-us "$SLOT_TIME_US"
  --difs-us "$DIFS_US"
  --sifs-us "$SIFS_US"
  --ack-us "$ACK_US"
  --ack-timeout-us "$ACK_TIMEOUT_US"
  --cw-min "$CW_MIN"
  --cw-max "$CW_MAX"
  --max-retries "$MAX_RETRIES"
  --carrier-sense-range-m "$CARRIER_SENSE_RANGE_M"
  --abreu-solver-gpu false
  --reliability-target "$RELIABILITY_TARGET"
  --delivery-ratio-target "$DELIVERY_RATIO_TARGET"
  --prune-delivery-min "$PRUNE_DELIVERY_MIN"
  --prune-rel-min "$PRUNE_REL_MIN"
  --drone-active-cost-before-qos "$DRONE_ACTIVE_COST_BEFORE_QOS"
  --bad-deactivation-penalty "$BAD_DEACTIVATION_PENALTY"
  --max-delay-ms "$MAX_DELAY_MS"
  --throughput-max-mbps "$THROUGHPUT_MAX_MBPS"
  --use-hybrid-reward "$USE_HYBRID_REWARD"
  --qos-maintain-bonus "$QOS_MAINTAIN_BONUS"
  --converge-patience "$CONVERGE_PATIENCE"
  --latency-headroom-penalty "$LATENCY_HEADROOM_PENALTY"
  --zone-violation-penalty "$ZONE_VIOLATION_PENALTY"
  --zone-link-weight-mult "$ZONE_LINK_WEIGHT_MULT"
  --zone-success-suppression "$ZONE_SUCCESS_SUPPRESSION"
  --zone-violation-abs-penalty "$ZONE_VIOLATION_ABS_PENALTY"
  --latency-from-longest-route "$LATENCY_FROM_LONGEST_ROUTE"
  --p-zone-violation-seed "$P_ZONE_VIOLATION_SEED"
  --converge-on-state "$CONVERGE_ON_STATE"
  --converge-eps "$CONVERGE_EPS"
  --curriculum-phase-name "$CURRICULUM_PHASE_NAME"
  --curriculum-phase-idx "$CURRICULUM_PHASE_IDX"
  --curriculum-block-idx "$CURRICULUM_BLOCK_IDX"
  --latest-metrics-path "$LATEST_METRICS_PATH"
  --device "$DEVICE"
)

echo "[START] Script: $SCRIPT"
echo "[START] Curriculum: $CURRICULUM | foreground=$BLOCK_FOREGROUND | background=$RUN_BACKGROUND"
echo "[START] Phase: $CURRICULUM_PHASE_IDX/$CURRICULUM_PHASE_NAME | block=$CURRICULUM_BLOCK_IDX"
echo "[START] Timesteps: $TIMESTEPS ($TIMESTEPS_MODE) | workers=$NUM_WORKERS rollout_steps=$ROLLOUT_STEPS"
echo "[START] Targets: rel=$RELIABILITY_TARGET delivery=$DELIVERY_RATIO_TARGET prune_delivery=$PRUNE_DELIVERY_MIN prune_rel=$PRUNE_REL_MIN cs=$CARRIER_SENSE_RANGE_M"
echo "[START] latest_metrics: $LATEST_METRICS_PATH"

if [ "$BLOCK_FOREGROUND" = "1" ] || [ "$RUN_BACKGROUND" = "0" ]; then
  exec "${CMD[@]}"
fi

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" || true)
  if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "O programa já está em execução com o PID: $PID."
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
NOHUP_PID=$!
echo "$NOHUP_PID" > "$PID_FILE"
echo "Programa iniciado em segundo plano."
echo "PID: $NOHUP_PID"
echo "Log: $LOG_FILE"
echo "PID salvo em: $PID_FILE"
