#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drone_env_multi.py (standalone)

AUTO-CONTIDO:
- Embute o DroneEnv (monoagente) dentro deste mesmo arquivo.
- Fornece o wrapper multiagente (Drone5GMAEnv) com action masking real (toggle + movimento)
  e higienização de métricas *_terminal em steps não-terminais.

Sem dependência de um arquivo externo drone_env.py.
"""


# =========================
# Embedded single-agent env
# =========================

# -*- coding: utf-8 -*-
"""
DroneEnv completo (PPO):
- Zonas parametrizáveis (count, w/h, single_fixed, p_sem_zona)
- Movimento macro+micro com níveis {0,1,2,4,8,16} células (step_code 0..5) com 5 direções (parado, +x, -x, +y, -y)
- Atualização de links com validações (max_out, sem self-loop, evita left-link opcional, evita cruzar zona)
- Cache de matriz de distâncias
- Recompensa com confiabilidade/latência/throughput/hops/estrutura
- Termos de espalhamento: NN-spacing, repulsão curta, uniformidade de ocupação (bins)
"""

import math
import inspect
import zlib
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYMNASIUM = True
except Exception:
    import gym
    from gym import spaces

    _GYMNASIUM = False
from collections import deque
import torch


try:
    from .abreu_modified import (
        solve_modified_abreu_multiflow, solve_modified_abreu_redundant,
        enumerate_simple_paths, _link_ber_from_success_prob, FlowRequest,
        _link_success_prob_from_distance, LinkModelParams,
    )
except ImportError:
    try:
        from abreu_modified import (
            solve_modified_abreu_multiflow, solve_modified_abreu_redundant,
            enumerate_simple_paths, _link_ber_from_success_prob, FlowRequest,
            _link_success_prob_from_distance, LinkModelParams,
        )
    except Exception:
        solve_modified_abreu_multiflow = None
        solve_modified_abreu_redundant = None
        enumerate_simple_paths = None
        _link_ber_from_success_prob = None
        FlowRequest = None
        _link_success_prob_from_distance = None
        LinkModelParams = None

def _torch_to_numpy_cpu_view(t: "torch.Tensor") -> np.ndarray:
    """Converte tensor torch -> numpy com o menor overhead possível.

    - CPU: retorna um *view* via .detach().numpy() (sem cópia).
    - Não-CPU: faz .detach().cpu().numpy() (cópia inevitável).

    Observação: este projeto roda em CPU por padrão (DEVICE=cpu), então
    isso elimina o custo do .cpu() e reduz overhead em hot-paths.
    """
    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    if t.device.type == "cpu":
        return t.detach().numpy()
    return t.detach().cpu().numpy()

# =========================
# Constantes/Parâmetros
# =========================

DEVICE = torch.device("cpu")

MAX_DRONES = 64
GRID_SIZE_DEFAULT = 100
SQUARE_SIZE_M_DEFAULT = 20.0

# Movimento: código discreto (0..3) -> passo em células {1,2,3,4}.
# Observação: dir_code==0 significa 'não mover' (o passo é ignorado).
# Tamanhos de passo de movimento (em células). Reduzido de [1,2,3,4,8,16] para
# [1,4,16]: corta o espaço de ação por agente intermediário ~48% (de 6425 para
# 3341), o que tende a acelerar a convergência (espaço de ação menor = menos
# experiência necessária para estimar o valor de cada ação). O passo de 1 célula
# preserva o ajuste fino de posição; 16 cobre deslocamentos grandes. Combinações
# ao longo de turnos sucessivos alcançam qualquer deslocamento.
STEP_CELLS_CHOICES = [1, 4, 16]
STEP_CELLS_N = len(STEP_CELLS_CHOICES)

# Metas/limiares
RELIABILITY_TARGET = 0.995
RELIABILITY_BASE = 0.01
THROUGHPUT_MAX_MBPS = 9600.0
# ----------------------------
# Use exclusivamente FIT_WEIGHTS_DEFAULT para ajustar pesos do fitness.

# Latência “física” simplificada
MAX_DELAY_MS_DEFAULT = 300.0
# Alias mantido para compatibilidade com versões anteriores do arquivo.
LATENCY_TARGET_MS_DEFAULT = MAX_DELAY_MS_DEFAULT

# =========================
# Pesos EFETIVOS do fitness (o que realmente entra no fit)
# Edite aqui (ou sobrescreva via DroneEnv(fit_weights={...}))
#
# O fitness (fit) usa estes pesos abaixo.
# =========================
# -------------------------------------------------------------------
# Pesos rebalanceados para faixa compacta e estável (PPO).
#
# A magnitude bruta anterior (milhares) saturava o clipping do PPO e
# desestabilizava a value function. Aqui a recompensa por step fica
# aproximadamente em [-3, +5], deixando a PRIORIDADE (estrutura
# lexicográfica em _calculate_fit), e não a magnitude, fazer a ordenação
# dos objetivos: caminho > confiabilidade/entrega > vazão/latência > nº de drones.
#
# Recomenda-se ainda normalizar a recompensa no trainer (VecNormalize com
# norm_reward=True) e usar gamma alto (>=0.99). Veja nota no final do arquivo.
# -------------------------------------------------------------------
FIT_WEIGHTS_DEFAULT = {
    "PATH_MISSING": 2.0,
    "LEFT_LINK_COEF": 0.5,
    "CYCLE_PENALTY": 1.0,
    # THROUGHPUT_PENALTY_MAX=2.0 (aumentado de 1.0): fortalece o empurrão da
    # vazão até o alvo. Relevante agora que a vazão é medida corretamente pelo
    # Abreu (goodput agregado real) e não mais pela aproximação que subestimava.
    "THROUGHPUT_PENALTY_MAX": 2.0,
    "LATENCY_OVER_LIMIT_W": 1.0,
    "TERM_REL": 3.0,
    "INVALID_ACTION_PENALTY": 0.05,
    "SUCCESS_BONUS": 5.0,
    # DRONE_ACTIVATION_PENALTY removida: redundante com o custo de estado
    # (DRONE_ACTIVE_COST_*). Minimização de drones fica a cargo do custo de estado.
    # DRONE_DEACTIVATION_BONUS e PRUNE_KEEP_PATH_BONUS removidos: eram bônus de
    # EVENTO por desativação (farmáveis via oscilação ativar->desativar). A
    # minimização de drones agora vem do CUSTO DE ESTADO (DRONE_ACTIVE_COST_*),
    # não-farmável. Ver _calculate_fit.
    "DRONE_ACTIVE_COST_AFTER_QOS": 0.20,
    "DRONE_ACTIVE_COST_BEFORE_QOS": 0.02,
    "BAD_DEACTIVATION_PENALTY": 0.80,
}


class DroneEnv(gym.Env):
    """
    Ambiente de drones para PPO (Gymnasium). Observação composta por:
      - posições normalizadas (x,y) para cada drone,
      - matriz de adjacência achatada,
      - máscara de drones ativos (aqui sempre 1.0 para N drones).
    Ações (MultiDiscrete):
      [ move_idx (0..MAX_DRONES-1), step_code (0..3)->{1,2,3,4}, dir_code (0..4), toggle_op (0..(1+2*E-1)) ]
"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            grid_size=GRID_SIZE_DEFAULT,
            num_drones=13,
            max_out_connections=4,
            square_size_m=SQUARE_SIZE_M_DEFAULT,
            zonas_proibidas=None,
            p_sem_zona=0.0,
            max_drones=None,
            # zonas parametrizáveis
            zonas_count=None,
            zona_size=None,  # (w, h)
            zona_single_fixed=False,
            # pesos do fit (efetivos) / logging detalhado do fit
            fit_weights=None,
            log_fit_terms=False,
            # ---- Observação extra (opcional) ----
            obs_zone_map=0,          # acrescenta um mapa (downsample) de zona proibida ao obs
            zone_map_scale=4,        # fator de downsample (ex.: 4 => 100x100 -> 25x25)
            obs_edge_cross=0,        # acrescenta matriz NxN indicando se a aresta (i,j) cruza zona
            obs_link_quality=1,      # acrescenta matriz NxN com prob. de sucesso por par (dist+sombra)
            zone_violation_penalty=6.0,   # peso da EVACUAÇÃO de zonas no potencial (0 = desligado)
            zone_link_weight_mult=3.5,    # links pesam Nx mais que drones (desvio de link é mais caro em KPI)
            zone_success_viol_ref=3.0,    # nº de violações p/ atingir o piso de supressão (decai gradual até lá)
            zone_abs_penalty_scale=13.0,  # divisor fixo do custo absoluto de zona. Fixo de
                                          # propósito: qualquer divisor que dependa do nº de
                                          # drones ativos reabre a brecha de diluição.
            latency_from_longest_route=False,  # False = usa o atraso agregado do modelo de Abreu
                                          # (worst_delay_ms; multi-caminho agrega por MÍNIMO, a 1ª
                                          # cópia que chega entrega). True = LEGADO: atraso do caminho
                                          # com mais saltos, herdado dos modelos de espaço livre, em
                                          # que atraso ~ nº de saltos. Mantido só para reproduzir
                                          # resultados anteriores.
            zone_violation_abs_penalty=0.0,  # custo ABSOLUTO por passo de violar zona (fora do Φ).
                                          # O Φ telescopa (paga a transição, não a permanência) e a
                                          # supressão só escala bônus positivos -> violar nunca custa
                                          # menos que zero. Este termo é o que torna "atravessar com
                                          # QoS" pior que "contornar". 0 = desligado (comportamento
                                          # antigo). Ver notas em _compute_reward.
            zone_success_suppression=1.0, # fator MULTIPLICATIVO dos bônus de sucesso/QoS quando há
                                          # violação de zona. 1.0=sem supressão; 0.1=bônus ~zerado
                                          # enquanto violar. Escalado pelo currículo (suave->duro).
                                          # É a forma de tornar violar NÃO-COMPENSÁVEL por KPI: um
                                          # termo ADITIVO (peso) é ruído frente ao reward ~800; a
                                          # supressão MULTIPLICATIVA tira uma fração do KPI, enorme.
                                          # CALIBRADO: com 2.0, evacuar rendia +0.67 mas o desvio
                                          # custava -0.6..-1.5 de KPI => FICAR na zona compensava.
                                          # Com 6.0 o ganho (+2.0) supera a perda típica do desvio.
            zone_anchor_margin=3,         # margem (células) que as zonas devem manter das âncoras
            p_zone_violation_seed=0.0,    # prob. de o reset SEMEAR violação (zona sobre a linha)
            min_line_hops_precomputed=None,   # calibração pré-computada pelo pai (evita varredura por worker)
            min_qos_drones_precomputed=None,  # idem para o baseline de drones
            calibration_shadow_k=21,          # nº de sorteios de sombra na calibração analítica
            calibration_min_drones_cap=14,   # TETO do baseline de drones (capacidade do meio).
                                          # Medido com o solver p/ cs=200m: 14 drones -> ~50% de QoS,
                                          # 16 e 18 -> 0,0%. Sem teto, a calibração devolvia 16 e a
                                          # poda só cobrava acima da faixa operável. 0 = sem teto.
            calibration_min_drones_quantile=0.85,  # percentil dos mínimos por sorteio que vira o
                                              # baseline de drones. 0.5 = comportamento antigo
                                              # (mínimo TÍPICO); 0.85 = mínimo ROBUSTO ao fade.
                                              # O baseline alimenta DRONE_ACTIVE_COST_AFTER_QOS,
                                              # o gate de poda e o teto de ativação — com 0.5 a
                                              # poda cobra a redundância que garante o QoS.
            # ---- Hard masks ----
            hard_mask_zone_edges=1,  # bloqueia ADD de arestas que cruzam zona (REMOVE permanece permitido)
            hard_mask_left_links=1,  # bloqueia ADD de links à esquerda (x_dst < x_src)
            hard_mask_cycles=1,      # bloqueia ADD de links que criariam ciclos (sem loops)
            hard_block_move_cross_zone=1,  # HARD: reverte movimento se AUMENTAR links que cruzam zona
            # ---- Currículo: "dirty reset" (começa sujo e termina 100% sujo) ----
            dirty_reset_prob_start=0.20,
            dirty_reset_prob_end=1.00,
            dirty_force_drone_in_zone_prob_start=0.20,
            dirty_force_drone_in_zone_prob_end=1.00,
            dirty_anneal_episodes=300,
            # ---- Modified Abreu model (optional) ----
            abreu_enabled=False,
            abreu_max_paths=64,
            abreu_max_hops=None,
            traffic_rate_mbps=10.0,
            frame_size_bytes=1500,
            buffer_size_packets=64,
            phy_rate_mbps=100.0,  # 802.11n 2x2 MIMO (estimativa conservadora ~1,5x
                                  # sobre 1 stream; o canal A2A é LOS-dominado, o que
                                  # reduz o ganho espacial vs o 2x teórico). Drone com
                                  # 2 antenas. 802.11g (54) não comporta 10 Mbps fim-a-
                                  # fim numa cadeia multi-hop por contenção (ver análise).
            slot_time_us=9.0,     # 802.11n (era 20 no 802.11g)
            difs_us=34.0,
            sifs_us=16.0,
            ack_us=44.0,
            ack_timeout_us=None,
            cw_min=16,
            cw_max=1024,
            max_retries=6,        # m = retries+1 = 7 tentativas (dot11ShortRetryLimit
                                  # do IEEE 802.11, aplicável a frames sem RTS/CTS —
                                  # consistente com o cenário do modelo de Abreu, que
                                  # rejeita RTS/CTS em multi-hop). Era 4 (LongRetryLimit,
                                  # p/ frames com RTS/CTS), inconsistente com o cenário.
                                  # Impacto nulo na faixa operacional (~200m); só altera
                                  # hops longos (>=400m) saturados.
            carrier_sense_range_m=120.0,
            # Raio máximo (m) para permitir criação de novos enlaces no action mask.
            # 0 desliga a restrição. Não altera REMOVE nem enlaces já existentes.
            link_neighbor_radius_m=0.0,
            abreu_reliability_min=0.99999,
            abreu_delay_max_ms=MAX_DELAY_MS_DEFAULT,
            abreu_multiflow_metric="min_flow_reliability",
            abreu_exogenous_flows=None,
            success_max_latency_ms=MAX_DELAY_MS_DEFAULT,
            reliability_target=RELIABILITY_TARGET,
            delivery_ratio_target=0.999,
            prune_delivery_min=0.30,
            prune_rel_min=0.20,
            drone_active_cost_before_qos=None,
            bad_deactivation_penalty=None,
            max_delay_ms=MAX_DELAY_MS_DEFAULT,
            throughput_max_mbps=THROUGHPUT_MAX_MBPS,
            # ---- Reset assistido ----
            # Se True, o reset já semeia uma cadeia dirigida 0 -> ... -> N-1.
            # Isso mantém o bootstrap focado em ajustar/remover/adicionar enlaces
            # a partir de uma topologia inicialmente conectada.
            seed_initial_path=True,
            # FASE 2 — dificuldade MÉDIA por padrão: 2 caminhos com saltos de
            # ~330 m geram warm-start com confiabilidade redundante ~0,76 (abaixo
            # do alvo 0,995), tornando o problema não-trivial — a política precisa
            # aprender a melhorar a topologia (encurtar saltos, adicionar
            # caminhos) para atingir os KPIs. Parametrizável: ajuste
            # seed_target_hop_m (maior = mais difícil) e seed_num_paths.
            seed_target_hop_m=140.0,
            seed_num_paths=2,
            # Folga de viabilidade da semeadura: a redução automática só atua se o
            # maior salto exceder (seed_target_hop_m * seed_viability_slack). Com
            # slack=1.15, saltos-alvo de ~330 m são PRESERVADOS (não "consertados"
            # de volta a <300 m), mantendo a dificuldade média. A redução ainda
            # protege contra saltos absurdos do jitter aleatório.
            seed_viability_slack=1.15,
            seed_randomness=0.0,
            # --- MODO LINHA MÍNIMA (warm-start construtivo) ---
            # seed_line_min=True: semeia a MENOR linha reta A->B que chega PRÓXIMO
            # de atender os KPIs sem fechá-los
            # (derivada ANALITICAMENTE com o solver de Abreu e os parâmetros do
            # próprio env: menor nº de hops que atende, menos um). Demais drones
            # ficam DESATIVADOS e espalhados pelo campo. A política aprende a
            # ATIVAR os drones necessários e criar rotas (redundantes ou não)
            # para fechar confiabilidade, vazão e latência.
            seed_line_min=False,
            # Fração do alvo de confiabilidade que a linha mínima semeada deve
            # alcançar NA MÉTRICA DA RECOMPENSA (calibração consistente; ver
            # _analytic_min_line_hops). 0.70 ~= linha com rel ~0.7*alvo.
            seed_line_rel_frac=0.70,
            # Custo por drone ativo DENTRO do potencial Phi (não gateado por QoS):
            # ativar um drone custa no diferencial (~ -c imediato) e só compensa
            # se a ativação melhorar KPIs/marcos (ganho medido ~ +7 ao fechar o
            # QoS vs -0.15 do custo). Ensina "ativar apenas o necessário" sem
            # bloquear ativações úteis. Desativar devolve o custo (potencial).
            drone_potential_cost=0.15,
            # Nascimento inteligente na ativação: ao ativar um drone, ele é
            # REPOSICIONADO para uma célula livre a até +-20 células da linha
            # média A->B, no QUARTO (trecho 1/4 do segmento) com MENOS drones
            # ativos, e tenta nascer com 1 LINK válido para o drone ativo mais próximo
            # (orientado por x, preservando o DAG). Elimina o "vale" entre
            # ativar e tornar o drone útil (antes: ~150-400 passos de transporte).
            activate_spawn_near_line=True,
            # --- RECOMPENSA HÍBRIDA ---
            use_hybrid_reward=True,
            hybrid_gamma=0.99,          # DEVE casar com o gamma do GAE (trainer)
            qos_maintain_bonus=6.0,
            # Calibração: MAINT deve superar o custo de drones no PIOR caso, senão
            # atingir o QoS com muitos drones é líquido-negativo e a política é
            # PUNIDA por ter sucesso. Pior caso ~ DRONE_ACTIVE_COST_AFTER_QOS(0.20)
            # x drones_excedentes(~27) = 5.4/passo. MAINT=6.0 > 5.4 garante que o
            # QoS é sempre líquido-positivo; a poda subsequente aumenta o retorno
            # (menos custo de drones). Medido: podar(1623) > manter(284) > pairar(-4).
            # Se True, o episódio termina ao atingir o QoS (comportamento antigo).
            # False (padrão): roda até o timeout, permitindo que a política MANTENHA
            # o QoS e PODE drones na fase pós-sucesso — necessário para os
            # objetivos secundários da recompensa híbrida terem efeito.
            terminate_on_success=False,
            # Terminação por CONVERGÊNCIA: encerra o episódio após `converge_patience`
            # passos consecutivos com |Δreward| < converge_eps, DESDE QUE o QoS já
            # tenha sido atingido. 0 desliga (roda até o horizonte). ~30 é um começo
            # razoável: curto o bastante para eficiência, longo o bastante para não
            # cortar durante micro-ajustes. Requer QoS atingido -> subótimo estável
            # (pairar) NÃO termina cedo, preservando a chance de reparo.
            converge_patience=30,
            converge_eps=0.01,
            converge_on_state=1,           # convergência por estabilidade do ESTADO (anti-farming)
            latency_headroom_penalty=0.5,  # penalidade densa quando a latência consome a folga (>80% do limite)
            rel_ramp_ext_w=0.0,            # extensão SUAVE da rampa de rel entre reliability_target e delivery_target (0=off)
    ):
        super().__init__()

        # ---------- parâmetros básicos ----------
        self.grid_size = int(grid_size)
        self.square_size_m = float(square_size_m)

        # Cache local de p(link) por distância (mesma semântica; evita recomputar erf/log10 repetidamente)
        self._prob_cache = {}
        self.max_out_connections = int(max_out_connections)
        self.max_drones = int(max_drones) if max_drones is not None else None
        self.num_drones = int(num_drones)

        # Define max_drones efetivo usado para dimensionar a observation_space
        if self.max_drones is None:
            self.max_drones = self.num_drones
        # Garante limites [1, MAX_DRONES]
        self.max_drones = max(1, min(int(self.max_drones), MAX_DRONES))
        # Garante que num_drones não ultrapasse max_drones
        if self.num_drones > self.max_drones:
            self.num_drones = self.max_drones

        # zonas
        self.zonas_proibidas_param = zonas_proibidas
        self.zonas_proibidas = set()
        # cache: grade booleana para consulta rápida de zona proibida (x,y)
        self._forbidden_grid = None
        self._zone_sig = 0  # deterministic signature of forbidden grid
        self.p_sem_zona = float(p_sem_zona)
        self.zonas_count = int(zonas_count) if zonas_count is not None else None
        self.zona_size = tuple(zona_size) if (zona_size and zona_size[0] and zona_size[1]) else None
        self.zona_single_fixed = bool(zona_single_fixed)

        # ---- Observação extra / hard masks ----
        self.obs_zone_map = bool(obs_zone_map)
        self.zone_map_scale = max(1, int(zone_map_scale))
        self.obs_edge_cross = bool(obs_edge_cross)
        self.obs_link_quality = bool(obs_link_quality)
        self.zone_violation_penalty = float(max(0.0, zone_violation_penalty))
        self.zone_link_weight_mult = float(max(0.0, zone_link_weight_mult))
        self.zone_violation_abs_penalty = float(max(0.0, zone_violation_abs_penalty))
        self.zone_abs_penalty_scale = float(max(1.0, zone_abs_penalty_scale))
        self.latency_from_longest_route = bool(latency_from_longest_route)
        self.zone_success_suppression = float(min(1.0, max(0.0, zone_success_suppression)))
        self.zone_success_viol_ref = float(max(1.0, zone_success_viol_ref))
        self.zone_anchor_margin = int(max(0, zone_anchor_margin))
        self.p_zone_violation_seed = float(min(1.0, max(0.0, p_zone_violation_seed)))
        self._linkq_cache = None
        self._linkq_sig = None
        # Calibrações PRÉ-COMPUTADAS (opcional): o processo pai calcula uma vez
        # e injeta via cfg; os workers pulam a varredura (~25s) do 1º reset.
        # Sem isso, 128 workers calibrando simultaneamente no lançamento viram
        # uma tempestade de CPU que parece travamento de dezenas de minutos.
        self.calibration_shadow_k = int(max(1, calibration_shadow_k))
        self.calibration_min_drones_quantile = float(min(1.0, max(0.5, calibration_min_drones_quantile)))
        self.calibration_min_drones_cap = int(max(0, calibration_min_drones_cap))
        if min_line_hops_precomputed is not None:
            try:
                self._min_line_cache = int(min_line_hops_precomputed)
                self._min_line_refined = True
            except Exception:
                pass
        if min_qos_drones_precomputed is not None:
            try:
                self._min_qos_drones_cache = int(min_qos_drones_precomputed)
            except Exception:
                pass
        self.hard_mask_zone_edges = bool(hard_mask_zone_edges)
        self.hard_mask_left_links = bool(hard_mask_left_links)
        self.hard_mask_cycles = bool(hard_mask_cycles)
        self.hard_block_move_cross_zone = bool(hard_block_move_cross_zone)
        self.seed_initial_path = bool(seed_initial_path)
        self.seed_target_hop_m = float(seed_target_hop_m)
        self.seed_num_paths = int(seed_num_paths)
        self.seed_viability_slack = float(max(1.0, seed_viability_slack))
        # Intensidade da aleatoriedade da semeadura (0 = fixa/determinística,
        # como antes; cresce até 1 = máxima variação dentro da faixa viável).
        # Controla jitter posicional, variação de K e da posição de A/B.
        self.seed_randomness = float(max(0.0, min(1.0, seed_randomness)))
        self.seed_line_min = bool(seed_line_min)
        self.seed_line_rel_frac = float(seed_line_rel_frac)
        self.drone_potential_cost = float(max(0.0, drone_potential_cost))
        self.activate_spawn_near_line = bool(activate_spawn_near_line)
        self._min_line_cache = None  # cache do cálculo analítico (params não mudam)

        # ---- Currículo: dirty reset ----
        self.dirty_reset_prob_start = float(dirty_reset_prob_start)
        self.dirty_reset_prob_end = float(dirty_reset_prob_end)
        self.dirty_force_drone_in_zone_prob_start = float(dirty_force_drone_in_zone_prob_start)
        self.dirty_force_drone_in_zone_prob_end = float(dirty_force_drone_in_zone_prob_end)
        self.dirty_anneal_episodes = max(1, int(dirty_anneal_episodes))
        self._reset_count = 0
        self._dirty_episode = False
        self._force_drone_in_zone = False

        # cache: zone_map (downsample) e edge_cross (matriz NxN)
        self._zone_map_down = None
        self._zone_map_down_sig = None
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None


        # métricas correntes (para logging externo)
        self.current_reliability = 0.0
        self.current_reliability_block = 0.0
        self.current_delivery_ratio = 0.0
        self.current_reliability_hybrid = 0.0
        self.current_latency_penalty = 0.0
        self.current_latency_ms = 0.0
        self.current_throughput = 0.0
        self.current_loss_rate = 1.0
        self.current_max_feasible_rate = 0.0
        self.current_num_paths = 0
        self.current_abreu_debug = {}

        # Caches exatos por estado/topologia (otimização sem alterar semântica)
        self._graph_metrics_cache = {}
        self._graph_metrics_cache_limit = 2048
        self._abreu_eval_cache = {}
        self._abreu_eval_cache_limit = 1024
        self._paths_cache = {}
        self._paths_cache_limit = 4096
        self._ber_path_cache = {}
        self._ber_path_cache_limit = 8192
        self._phy_path_cache = {}
        self._phy_path_cache_limit = 8192

        # Modified-Abreu configuration
        self.abreu_enabled = bool(abreu_enabled and solve_modified_abreu_redundant is not None)
        self.abreu_max_paths = max(1, int(abreu_max_paths))
        self.abreu_max_hops = None if abreu_max_hops is None else max(1, int(abreu_max_hops))
        self.traffic_rate_mbps = float(traffic_rate_mbps)
        self.frame_size_bytes = int(frame_size_bytes)
        self.buffer_size_packets = int(buffer_size_packets)
        self.phy_rate_mbps = float(phy_rate_mbps)
        self.slot_time_us = float(slot_time_us)
        self.difs_us = float(difs_us)
        self.sifs_us = float(sifs_us)
        self.ack_us = float(ack_us)
        self.ack_timeout_us = float(self.ack_us if ack_timeout_us is None else ack_timeout_us)
        self.cw_min = int(cw_min)
        self.cw_max = int(cw_max)
        self.max_retries = int(max_retries)
        self.carrier_sense_range_m = float(carrier_sense_range_m)
        self.link_neighbor_radius_m = float(max(0.0, link_neighbor_radius_m))
        # Modelo de rádio compartilhado entre _prob_link (recompensa) e o solver
        # de Abreu (avaliação terminal). Usar a MESMA instância garante que a
        # física de enlace seja idêntica nos dois caminhos (opção B). Se a classe
        # não estiver disponível, fica None e _link_success_prob_from_distance usa
        # seus defaults internos (também os mesmos do solver).
        # OPÇÃO B: o sombreamento agora é ESTOCÁSTICO por enlace (ver reset e
        # _prob_link_pair), com sigma ~3,5 dB da literatura A2A. Portanto a MARGEM
        # fixa do link_model volta ao valor de base pequeno (não somar duas vezes);
        # a variabilidade vem do termo estocástico, não de uma margem constante.
        # O expoente de path loss (2,6, medido por Goddemeier 2015) é mantido.
        try:
            if LinkModelParams is not None:
                _shadow_margin = float(getattr(self, "link_shadow_margin_db", 0.21))
                try:
                    self._link_model = LinkModelParams(shadow_margin_db=_shadow_margin)
                except TypeError:
                    self._link_model = LinkModelParams()
                    try:
                        self._link_model.shadow_margin_db = _shadow_margin
                    except Exception:
                        pass
            else:
                self._link_model = None
        except Exception:
            self._link_model = None
        # Effective QoS targets are centralized below; legacy Abreu args do not override them.
        self.abreu_multiflow_metric = str(abreu_multiflow_metric)
        self.abreu_exogenous_flows = list(abreu_exogenous_flows) if abreu_exogenous_flows is not None else None
        self.reliability_target = float(reliability_target)
        self.delivery_ratio_target = float(max(0.0, min(1.0, delivery_ratio_target)))
        self.prune_delivery_min = float(max(0.0, min(1.0, prune_delivery_min)))
        self.prune_rel_min = float(max(0.0, min(1.0, prune_rel_min)))
        self.max_delay_ms = float(max_delay_ms)
        self.throughput_max_mbps = float(throughput_max_mbps)

        # ---------- pesos efetivos do fitness ----------
        # IMPORTANTE: precisa ser inicializado ANTES dos overrides abaixo.
        # Com multiprocessing spawn, cada worker instancia DroneEnv do zero;
        # se self.fit_w for criado depois, o worker morre com AttributeError.
        self.fit_w = dict(FIT_WEIGHTS_DEFAULT)
        if fit_weights:
            self.fit_w.update({k: float(v) for k, v in fit_weights.items()})
        if drone_active_cost_before_qos is not None:
            self.fit_w["DRONE_ACTIVE_COST_BEFORE_QOS"] = float(drone_active_cost_before_qos)
        if bad_deactivation_penalty is not None:
            self.fit_w["BAD_DEACTIVATION_PENALTY"] = float(bad_deactivation_penalty)

        # Legacy aliases: kept aligned with the effective target unless a caller explicitly inspects them.
        self.success_max_latency_ms = float(self.max_delay_ms)
        self.abreu_reliability_min = float(self.reliability_target)
        self.abreu_delay_max_ms = float(self.max_delay_ms)
        self.last_deactivated_drone = -1
        self.last_removed_links = 0
        self.last_qos_target_met = 0.0
        # --- RECOMPENSA HÍBRIDA (potential-based no primário + secundários pós-QoS) ---
        # use_hybrid_reward=True: o componente PRIMÁRIO da recompensa (construção de
        # topologia + KPIs até o alvo) vira diferencial potential-based
        # r_prim = gamma*Phi(s') - Phi(s), o que ELIMINA acúmulo em subótimo
        # (pairar rende ~0 por telescopagem). Os SECUNDÁRIOS (manutenção do QoS,
        # minimização de drones, latência baixa) permanecem ABSOLUTOS e só ligam
        # após qos_target_met — dando incentivo forte e consistente para atingir E
        # manter o QoS. Requer a porta de throughput consistente (já corrigida).
        self.use_hybrid_reward = bool(use_hybrid_reward)
        self.hybrid_gamma = float(hybrid_gamma)
        self._primary_potential_prev = None  # Phi(s) do passo anterior; None no reset
        # Bônus de MANUTENÇÃO do QoS (por passo enquanto qos pleno). É o que faz
        # "manter o ótimo" render mais que pairar, já que o diferencial primário
        # satura (~0) uma vez atingido o alvo. Calibrado ~ escala dos KPIs.
        self.qos_maintain_bonus = float(qos_maintain_bonus)
        self.terminate_on_success = bool(terminate_on_success)
        self.converge_patience = int(converge_patience)
        self.converge_eps = float(converge_eps)
        self.converge_on_state = bool(converge_on_state)
        self.latency_headroom_penalty = float(max(0.0, latency_headroom_penalty))
        self.rel_ramp_ext_w = float(max(0.0, rel_ramp_ext_w))
        self._prev_state_sig = None
        self._prev_step_reward = None
        self._stable_steps = 0
        # ---------- espaços ----------
        # ação: 4 dimensões (move 1 drone por step + operação de link "achatada")
        # [move_idx, step_code, dir_code, toggle_op]
        #
        # toggle_op:
        #   0              -> não altera links
        #   1..E           -> ADD (src,dst) para cada par dirigido src!=dst (ordem fixa em self._edge_pairs)
        #   E+1..2E        -> REMOVE (src,dst)
        #
        # Isso permite action masking REAL (sem combinações inválidas src/dst).
        self._edge_pairs = [(i, j) for i in range(self.max_drones) for j in range(self.max_drones) if i != j]
        self._edge_pair_to_idx = {p: k for k, p in enumerate(self._edge_pairs)}
        self._E = len(self._edge_pairs)
        self._activate_op_base = 1 + 2 * self._E
        self._deactivate_op_base = 1 + 2 * self._E + self.max_drones
        self._toggle_ops_n = 1 + 2 * self._E + 2 * self.max_drones

        # Vetores prontos para indexação vetorizada no action_masks()
        # (evita loop Python por par (src,dst) a cada passo)
        self._edge_src = np.fromiter((p[0] for p in self._edge_pairs), dtype=np.int16, count=self._E)
        self._edge_dst = np.fromiter((p[1] for p in self._edge_pairs), dtype=np.int16, count=self._E)

        self.action_space = spaces.MultiDiscrete(
            np.array([self.max_drones, STEP_CELLS_N, 5, self._toggle_ops_n], dtype=np.int64)
        )

        # cache simples para action_masks (evita recomputar quando estado não muda)
        self._mask_version = 0
        self._mask_cache_version = -1
        self._mask_cache = None
        # observação: base FIXA com padding até max_drones.
        # Partes:
        #   - pos (max_drones x 2) normalizado em [-1, 1]
        #   - adj (max_drones x max_drones) em {0,1}
        #   - mask (max_drones) em {0,1} indicando drones ativos
        # Extras opcionais (via args):
        #   - zone_map (downsample do grid de zonas proibidas)
        #   - edge_cross (matriz NxN: 1 se o segmento (i,j) cruza zona)
        obs_dim_pos = self.max_drones * 2
        obs_dim_adj = self.max_drones * self.max_drones
        obs_dim_mask = self.max_drones

        # zone-map (downsample)
        if self.obs_zone_map:
            self._zone_map_ds = int(math.ceil(float(self.grid_size) / float(self.zone_map_scale)))
            obs_dim_zone = int(self._zone_map_ds * self._zone_map_ds)
        else:
            self._zone_map_ds = 0
            obs_dim_zone = 0

        # edge-cross (NxN)
        obs_dim_edge_cross = int(self.max_drones * self.max_drones) if self.obs_edge_cross else 0
        # Canal NxN de QUALIDADE POR ENLACE (prob. de sucesso do par, distância
        # + sombreamento do episódio). Sem ele o agente é cego para QUAL enlace
        # está com fade: a política converge para UMA topologia fixa e o
        # resultado terminal vira função do sorteio da sombra (atrator rel~0.35
        # observado em treino). Com o canal, "reparar" vira aprendível.
        obs_dim_linkq = int(self.max_drones * self.max_drones) if self.obs_link_quality else 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim_pos + obs_dim_adj + obs_dim_mask + obs_dim_zone + obs_dim_edge_cross + obs_dim_linkq,),
            dtype=np.float32
        )

        # ---------- estado ----------
        self._rng = np.random.RandomState()
        self._dist_matrix = None
        self._dist_pos_sig = None
        self._bonus_sucesso_dado = False
        self.log_fit_terms = bool(log_fit_terms)

        self._reset_zonas()
        self._init_state()
        # SEMEADURA DE VIOLAÇÃO: com prob. p_zone_violation_seed, reposiciona uma
        # zona SOBRE a linha inicial — criando drones dentro dela e links que a
        # cruzam. É o cenário de EVACUAÇÃO ("uma no-fly zone foi declarada sobre
        # a rede já implantada: saia e reconfigure"). Sem isso, o estado inicial
        # nunca viola e a política jamais treina a evacuação — só o contorno.
        self._seed_zone_violation()
        self._sync_active_mask()

        self._last_reward = 0.0
        # Tracking para shaping de zona (delta de violações e bônus 1x quando limpa)
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._zone_clean_once = False

        self._last_reward = 0.0
        self._episode_steps = 0
        self._max_episode_steps = 2000

        self._success_streak = 0  # passos consecutivos em sucesso
        self._success_need = 1

    # =========================
    # Zonas proibidas
    # =========================
    def _reset_zonas(self):
        """(Re)gera as zonas proibidas e reconstrói a grade booleana de consulta rápida.

        Importante: mantém a semântica original:
        - Se zonas_proibidas_param foi fornecido: usa exatamente esse conjunto.
        - Caso contrário: amostra proceduralmente (p_sem_zona, count, tamanho, etc.).
        """
        self.zonas_proibidas.clear()

        # Caso 1: zonas fixas fornecidas externamente
        if self.zonas_proibidas_param is not None:
            for (x, y) in self.zonas_proibidas_param:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.zonas_proibidas.add((x, y))
            self._rebuild_forbidden_grid()
            return

        # Caso 2: chance de não ter zona (modo procedural)
        if self.p_sem_zona > 0.0 and self._rng.rand() < self.p_sem_zona:
            self._rebuild_forbidden_grid()
            return

        # contagem e tamanho
        count = self.zonas_count if self.zonas_count is not None else self._rng.randint(0, 3)
        if count <= 0:
            self._rebuild_forbidden_grid()
            return

        if self.zona_size:
            w, h = map(int, self.zona_size)
        else:
            w, h = max(2, self.grid_size // 12), max(2, self.grid_size // 12)

        if self.zona_single_fixed:
            # única zona central fixa
            cx, cy = self.grid_size // 2, self.grid_size // 2
            x0 = max(0, cx - w // 2)
            y0 = max(0, cy - h // 2)
            for xx in range(x0, min(self.grid_size, x0 + w)):
                for yy in range(y0, min(self.grid_size, y0 + h)):
                    self.zonas_proibidas.add((xx, yy))
            self._rebuild_forbidden_grid()
            return

        # múltiplas zonas aleatórias — com GARANTIA DE DESVIABILIDADE.
        # Uma zona sorteada sem restrição pode cobrir uma ÂNCORA (drone 0 ou
        # N-1), que não pode se mover nem ser desativada: o episódio nasce com
        # violação PERMANENTE e impossível de resolver — penalidade eterna e
        # sinal de aprendizado envenenado (medido: ~0.5% dos resets com zonas
        # 15x15). Também pode bloquear o corredor inteiro (nenhuma rota A->B
        # possível). Aqui cada zona é re-sorteada até ser desviável.
        gmax = self.grid_size - 1
        anchors = [(0, gmax // 2), (gmax, gmax // 2)]
        margin = max(1, int(getattr(self, "zone_anchor_margin", 3)))
        placed = 0
        for _ in range(count):
            for _try in range(40):
                x0 = self._rng.randint(0, max(1, self.grid_size - w))
                y0 = self._rng.randint(0, max(1, self.grid_size - h))
                x1, y1 = x0 + w - 1, y0 + h - 1
                # (a) não cobre âncora (com margem de folga p/ o 1º hop existir)
                hits_anchor = any(
                    (x0 - margin) <= ax <= (x1 + margin) and (y0 - margin) <= ay <= (y1 + margin)
                    for (ax, ay) in anchors
                )
                if hits_anchor:
                    continue
                # (b) não bloqueia o corredor: simula a adição e exige que reste
                # ao menos uma faixa livre em y para cada coluna x da zona
                cand = set()
                for xx in range(x0, min(self.grid_size, x0 + w)):
                    for yy in range(y0, min(self.grid_size, y0 + h)):
                        cand.add((xx, yy))
                merged = self.zonas_proibidas | cand
                blocked = False
                for xx in range(x0, min(self.grid_size, x0 + w)):
                    if all((xx, yy) in merged for yy in range(self.grid_size)):
                        blocked = True
                        break
                if blocked:
                    continue
                self.zonas_proibidas |= cand
                placed += 1
                break
        # (se 40 tentativas falharem para alguma zona, ela simplesmente não é
        # colocada — melhor menos zonas que um episódio impossível)

        self._rebuild_forbidden_grid()

    def _seed_zone_violation(self):
        """Reposiciona uma zona sobre a linha semeada, garantindo violação inicial.

        Chamado APÓS _init_state (a linha já existe). Escolhe um drone ativo
        intermediário ao acaso e centra uma zona nele — garante >=1 drone dentro
        e (por construção, já que a linha é uma corrente) >=1 link cruzando.
        Não move nenhum drone: é a ZONA que aparece sobre a rede, o que é
        fisicamente o cenário realista (no-fly zone declarada a posteriori)."""
        p = float(getattr(self, "p_zone_violation_seed", 0.0))
        if p <= 0.0 or self._rng.rand() >= p:
            return
        if self.zonas_count is not None and int(self.zonas_count) <= 0:
            return
        try:
            n = int(self.num_drones)
            enabled = self._enabled_mask_np()
            # candidatos: ativos INTERMEDIÁRIOS (nunca as âncoras 0 e N-1, que
            # são fixas e não podem evacuar)
            # nunca centra a zona nas âncoras nem tão perto a ponto de cobri-las
            # (âncora coberta = violação impossível de resolver: ela não se move).
            gmax_c = self.grid_size - 1
            anchors_c = [(0, gmax_c // 2), (gmax_c, gmax_c // 2)]
            cands = [i for i in range(1, n - 1) if bool(enabled[i])]
            if not cands:
                return
            idx = int(cands[self._rng.randint(0, len(cands))])
            pos = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2])
            cx = int(round(float(pos[idx, 0])))
            cy = int(round(float(pos[idx, 1])))
            if self.zona_size:
                w, h = map(int, self.zona_size)
            else:
                w = h = max(2, self.grid_size // 12)
            x0 = max(0, min(self.grid_size - w, cx - w // 2))
            y0 = max(0, min(self.grid_size - h, cy - h // 2))
            # rejeita se a zona cobriria uma âncora (violação irresolúvel)
            x1c, y1c = x0 + w - 1, y0 + h - 1
            if any(x0 <= ax <= x1c and y0 <= ay <= y1c for (ax, ay) in anchors_c):
                return
            # ACRESCENTA (NÃO substitui) — as zonas aleatórias sorteadas em
            # _reset_zonas são PRESERVADAS. A política precisa aprender as duas
            # competências no MESMO episódio: EVACUAR a zona que caiu sobre a
            # rota e CONTORNAR as demais espalhadas pelo mapa.
            for xx in range(x0, min(self.grid_size, x0 + w)):
                for yy in range(y0, min(self.grid_size, y0 + h)):
                    self.zonas_proibidas.add((xx, yy))
            self._rebuild_forbidden_grid()
        except Exception:
            pass

    def _rebuild_forbidden_grid(self):
        """Reconstrói uma máscara booleana [grid_size x grid_size] com True nas células proibidas.

        - Mantém a representação original em set (self.zonas_proibidas)
        - Apenas acelera consultas repetidas no hot-path (step/fit)
        """
        if not self.zonas_proibidas:
            self._forbidden_grid = None
            self._zone_sig = 0
            # mantém shapes de observação consistentes mesmo sem zona
            self._rebuild_zone_map_cache()
            self._invalidate_edge_cross_cache()
            return
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
        for (x, y) in self.zonas_proibidas:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                g[int(x), int(y)] = True
        self._forbidden_grid = g
        self._zone_sig = int(zlib.crc32(g.tobytes()))
        # atualiza caches dependentes de zona
        self._rebuild_zone_map_cache()
        self._invalidate_edge_cross_cache()


    # =========================
    # Caches auxiliares: zone_map / edge_cross
    # =========================

    def _rebuild_zone_map_cache(self):
        """Recalcula o mapa (downsample) da zona proibida usado em obs_zone_map."""
        if not getattr(self, "obs_zone_map", False):
            self._zone_map_down = None
            self._zone_map_down_sig = None
            return

        ds = int(getattr(self, "_zone_map_ds", 0) or 0)
        if ds <= 0:
            self._zone_map_down = np.zeros((0,), dtype=np.float32)
            self._zone_map_down_sig = None
            return

        # Se não existe zona neste episódio, ainda assim retornamos um vetor
        # de tamanho FIXO (ds*ds) cheio de zeros para manter obs consistente.
        if self._forbidden_grid is None:
            self._zone_map_down = np.zeros((ds * ds,), dtype=np.float32)
            self._zone_map_down_sig = (int(getattr(self, "_zone_sig", 0)) ^ (int(self.zone_map_scale) << 16) ^ (ds << 1)) & 0xFFFFFFFF
            return

        scale = max(1, int(self.zone_map_scale))
        g = self._forbidden_grid.astype(bool, copy=False)  # (grid,grid) em [x,y]

        # padding para múltiplo de scale
        target = ds * scale
        pad_x = max(0, target - g.shape[0])
        pad_y = max(0, target - g.shape[1])
        if pad_x or pad_y:
            g = np.pad(g, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=False)

        # reshape -> (ds, scale, ds, scale) e faz OR por bloco
        try:
            g4 = g.reshape(ds, scale, ds, scale)
            down = g4.any(axis=(1, 3)).astype(np.float32, copy=False)
        except Exception:
            # fallback seguro
            down = np.zeros((ds, ds), dtype=np.float32)
            for ix in range(ds):
                for iy in range(ds):
                    x0 = ix * scale
                    y0 = iy * scale
                    block = g[x0:x0 + scale, y0:y0 + scale]
                    down[ix, iy] = 1.0 if np.any(block) else 0.0

        self._zone_map_down = down.reshape(-1).astype(np.float32, copy=False)
        # assinatura: zona + scale
        self._zone_map_down_sig = (int(getattr(self, "_zone_sig", 0)) ^ (scale << 16) ^ (ds << 1)) & 0xFFFFFFFF

    def _invalidate_edge_cross_cache(self):
        """Invalida caches que dependem de zona/posições (edge_cross)."""
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None

    def _zone_success_factor(self) -> float:
        """Fator multiplicativo [supp..1.0] aplicado aos bônus de SUCESSO/QoS,
        PROPORCIONAL ao grau de violação (não binário).

        Retorna 1.0 quando LIMPO. Conforme aumentam as violações, decai suave
        até o piso `zone_success_suppression`. A proporcionalidade é ESSENCIAL:
        a versão binária (qualquer violação -> supp) punia 1 link residual tanto
        quanto 5 violações, então a política não recebia crédito por limpar
        quase tudo e ficava presa com o último link (sucesso zerado, sem
        gradiente para fechar). Aqui, reduzir de 2 violações para 1 JÁ recupera
        parte do bônus -> gradiente contínuo rumo ao limpo.

        fator = 1 - (1 - supp) * min(1, viol / viol_ref), viol = drones + links.
        viol_ref normaliza a escala (poucas violações já pesam, mas o decaimento
        é gradual). O piso supp garante que NUNCA zere: mesmo violando, resta um
        fio de sinal de KPI para a política não ficar cega ao objetivo de rede."""
        supp = float(getattr(self, "zone_success_suppression", 1.0))
        if supp >= 1.0 or not self.zonas_proibidas:
            return 1.0
        zd, zl = self._count_zone_violations()
        # PESO DE LINK CONSISTENTE COM O Φ: antes contava drones e links
        # igualmente (viol = zd + zl) enquanto o Φ pesa link em
        # zone_link_weight_mult (3.5x). As duas pressões apontavam para lados
        # diferentes, e o residual medido é justamente link (links_cruzando
        # ~1.3 contra drones_dentro ~0.7).
        link_mult = float(getattr(self, "zone_link_weight_mult", 1.0))
        viol = float(zd) + link_mult * float(zl)
        if viol <= 0.0:
            return 1.0  # limpo -> bônus cheio
        viol_ref = float(getattr(self, "zone_success_viol_ref", 3.0))
        frac = min(1.0, viol / max(1e-9, viol_ref))
        # decai de 1.0 (limpo) até supp (>= viol_ref violações), linear
        return float(1.0 - (1.0 - supp) * frac)

    def _count_zone_violations(self) -> tuple:
        """(drones ATIVOS dentro de zona, links ATIVOS que cruzam zona).

        Base da EVACUAÇÃO: a máscara já impede CRIAR dentro da zona (mover para
        dentro e ADD de aresta que cruza), mas nada empurra para SAIR quem já
        está lá — e uma rota que atravessa a zona pode ter KPI ótimo, então os
        KPIs sozinhos não dão gradiente de evacuação. Este contador alimenta o
        termo de potencial que recompensa REDUZIR violações."""
        if not self.zonas_proibidas:
            return 0, 0
        n = int(self.num_drones)
        try:
            enabled = self._enabled_mask_np()
            pos = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2])
            grid = self._forbidden_grid
            if grid is None:
                return 0, 0
            gmax = int(self.grid_size) - 1
            n_drones = 0
            for i in range(n):
                if not bool(enabled[i]):
                    continue
                xi = int(np.clip(int(round(float(pos[i, 0]))), 0, gmax))
                yi = int(np.clip(int(round(float(pos[i, 1]))), 0, gmax))
                if bool(grid[xi, yi]):
                    n_drones += 1
            n_links = 0
            self._ensure_edge_cross_mat()
            if self._edge_cross_mat is not None and n > 1:
                adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n])
                cross = self._edge_cross_mat[:n, :n]
                n_links = int((((adj_np > 0.5).astype(np.uint8)) & (cross > 0)).sum())
            return int(n_drones), int(n_links)
        except Exception:
            return 0, 0

    def _ensure_edge_cross_mat(self):
        """Garante self._edge_cross_mat (NxN) para o número atual de drones.

        edge_cross_mat[i,j] = 1 se o segmento entre i e j cruza zona proibida.
        """
        if not (getattr(self, "obs_edge_cross", False) or getattr(self, "hard_mask_zone_edges", False)):
            return None

        n = int(self.num_drones)
        if n <= 1:
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return self._edge_cross_mat

        if not getattr(self, "zonas_proibidas", None):
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return self._edge_cross_mat

        zone_sig = int(getattr(self, "_zone_sig", 0))
        if (self._edge_cross_mat is None) or (self._edge_cross_n != n) or (self._edge_cross_zone_sig != zone_sig):
            mat = np.zeros((n, n), dtype=np.uint8)
            pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)

            # calcula apenas triângulo superior e espelha (simétrico)
            for i in range(n):
                xi = int(round(float(pos_np[i, 0])))
                yi = int(round(float(pos_np[i, 1])))
                p1 = (xi, yi)
                for j in range(i + 1, n):
                    xj = int(round(float(pos_np[j, 0])))
                    yj = int(round(float(pos_np[j, 1])))
                    p2 = (xj, yj)
                    cross = self._linha_passa_por_zona(p1, p2, (i, j))
                    v = 1 if cross else 0
                    mat[i, j] = v
                    mat[j, i] = v

            self._edge_cross_mat = mat
            self._edge_cross_n = n
            self._edge_cross_zone_sig = zone_sig

        return self._edge_cross_mat

    def _update_edge_cross_after_move(self, moved_idx: int):
        """Atualiza incrementalmente a matriz edge_cross quando 1 drone move."""
        if not (getattr(self, "obs_edge_cross", False) or getattr(self, "hard_mask_zone_edges", False)):
            return

        n = int(self.num_drones)
        if n <= 1:
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return

        if not getattr(self, "zonas_proibidas", None):
            self._edge_cross_mat = np.zeros((n, n), dtype=np.uint8)
            self._edge_cross_n = n
            self._edge_cross_zone_sig = int(getattr(self, "_zone_sig", 0))
            return

        moved_idx = int(moved_idx)
        if moved_idx < 0 or moved_idx >= n:
            return

        # se cache inválido (n mudou ou zona mudou), recalcula tudo
        zone_sig = int(getattr(self, "_zone_sig", 0))
        if (self._edge_cross_mat is None) or (self._edge_cross_n != n) or (self._edge_cross_zone_sig != zone_sig):
            self._ensure_edge_cross_mat()
            return

        pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
        xi = int(round(float(pos_np[moved_idx, 0])))
        yi = int(round(float(pos_np[moved_idx, 1])))
        p1 = (xi, yi)

        for j in range(n):
            if j == moved_idx:
                self._edge_cross_mat[moved_idx, j] = 0
                continue
            xj = int(round(float(pos_np[j, 0])))
            yj = int(round(float(pos_np[j, 1])))
            p2 = (xj, yj)
            cross = self._linha_passa_por_zona(p1, p2, (moved_idx, j))
            v = 1 if cross else 0
            self._edge_cross_mat[moved_idx, j] = v
            self._edge_cross_mat[j, moved_idx] = v




    def _enabled_mask_np(self) -> np.ndarray:
        n = int(max(0, int(getattr(self, "num_drones", 0))))
        cur = getattr(self, "drone_enabled", None)
        if cur is None or len(cur) != n:
            cur = np.ones((n,), dtype=np.bool_)
            self.drone_enabled = cur
        return np.asarray(cur[:n], dtype=np.bool_)

    def _current_adj_np(self) -> np.ndarray:
        n = int(self.num_drones)
        adj_np = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5).astype(np.float32)
        enabled = self._enabled_mask_np()
        if enabled.size == n:
            adj_np = adj_np * (enabled[:, None].astype(np.float32) * enabled[None, :].astype(np.float32))
        return adj_np

    # ---- TRAVA DE CONECTIVIDADE (0 -> N-1) ----
    # Impede que REMOVE/desativação quebrem o último caminho entre o primeiro
    # (A) e o último (B) drone. A verificação SIMULA a mutação e checa
    # _has_path(0, N-1) no grafo resultante — cobrindo qualquer aresta-ponte,
    # não apenas arestas incidentes às âncoras. Se a rede JÁ está desconectada,
    # nada é bloqueado (não há caminho a preservar; evita deadlock de reparo).
    def _remove_disconnects(self, src: int, dst: int) -> bool:
        """True se remover a aresta (src,dst) desconectar 0 -> N-1."""
        n = int(self.num_drones)
        if n < 2:
            return False
        adj = self._current_adj_np().copy()
        if not self._has_path(0, n - 1, adj):
            return False
        adj[int(src), int(dst)] = 0.0
        return not self._has_path(0, n - 1, adj)

    def _deactivate_disconnects(self, drone_idx: int) -> bool:
        """True se desativar drone_idx (zerando TODOS os seus enlaces)
        desconectar 0 -> N-1."""
        n = int(self.num_drones)
        if n < 2:
            return False
        adj = self._current_adj_np().copy()
        if not self._has_path(0, n - 1, adj):
            return False
        i = int(drone_idx)
        adj[i, :] = 0.0
        adj[:, i] = 0.0
        return not self._has_path(0, n - 1, adj)

    def _current_prune_gate(self) -> bool:
        """Permite poda leve usando apenas métricas rápidas/cached.

        Importante: esta função é chamada durante validação de ação/máscara.
        Portanto ela não deve chamar o solver Abreu completo.
        """
        n = int(self.num_drones)
        if n < 2:
            return False
        # ESCAPE DA CATRACA: podar EXCEDENTE sobre o mínimo analítico do cenário
        # nunca exige margem de KPI. Sem isso, a sequência "exploração ativa
        # demais -> redundância disjunta derruba delivery/vazão para baixo dos
        # limiares -> gate fecha -> desativação mascarada para sempre" prendia a
        # política num estado de ~N drones ativos sem porta de saída (observado
        # em treino real: delivery ~0.27 < prune_delivery_min=0.30 por 2M steps,
        # bad_deactivation e prune_keep_path zerados). A trava de conectividade
        # (_deactivate_disconnects) e o min_keep continuam protegendo cada
        # desativação individual.
        try:
            active = int(np.count_nonzero(self._enabled_mask_np()))
            if active > int(self._analytic_min_qos_drones()):
                return True
        except Exception:
            pass
        try:
            pos_np = np.asarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float32)
            metrics = self._evaluate_graph_metrics(pos_np, self._current_adj_np(), use_abreu=False)
            if not bool(metrics.get("path_exists", False)):
                return False
            rel_block = float(metrics.get("reliability_block", metrics.get("reliability", 0.0)))
            delivery_ratio = float(metrics.get("delivery_ratio", 0.0))
            return bool(
                delivery_ratio + 1e-12 >= float(getattr(self, "prune_delivery_min", 0.30))
                and rel_block + 1e-12 >= float(getattr(self, "prune_rel_min", 0.20))
            )
        except Exception:
            return False

    def _current_qos_gate(self) -> bool:
        """Gate barato para uso em action masking/step.

        O QoS físico com Abreu é caro demais para ser reavaliado a cada ação.
        Aqui usa-se o último resultado terminal/cached e, se necessário, uma
        aproximação leve. O Abreu completo permanece no bloco terminal/teste.
        """
        if float(getattr(self, "last_qos_target_met", 0.0)) >= 1.0:
            return True
        n = int(self.num_drones)
        if n < 2:
            self.last_qos_target_met = 0.0
            return False
        try:
            pos_np = np.asarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float32)
            metrics = self._evaluate_graph_metrics(pos_np, self._current_adj_np(), use_abreu=False)
            qos_ok = bool(self._success_targets_met(metrics))
            self.last_qos_target_met = 1.0 if qos_ok else 0.0
            return qos_ok
        except Exception:
            self.last_qos_target_met = 0.0
            return False

    def _current_reliability_gate(self) -> bool:
        n = int(self.num_drones)
        if n < 2:
            return False
        pos_np = np.asarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float32)
        metrics = self._evaluate_graph_metrics(pos_np, self._current_adj_np())
        _, target_rel, _ = self._min_drone_targets()
        rel = float(metrics.get("reliability", 0.0))
        path_ok = bool(metrics.get("path_exists", False))
        self.current_reliability = rel
        return bool(path_ok and (rel + 1e-12 >= float(target_rel)))

    def _birth_link_valid(self, src: int, dst: int, adj_np: np.ndarray, pos_np: np.ndarray, enabled: np.ndarray | None = None) -> bool:
        """Valida o link criado automaticamente na ativação de um drone.

        Usa as mesmas restrições hard do ADD normal: nós ativos, sem self-loop,
        sem duplicar aresta, max_out, raio de vizinhança, zona, left-link e ciclo.
        """
        try:
            n = int(self.num_drones)
            src = int(src)
            dst = int(dst)
            if src < 0 or dst < 0 or src >= n or dst >= n or src == dst:
                return False
            if enabled is not None:
                if src >= enabled.size or dst >= enabled.size or (not bool(enabled[src])) or (not bool(enabled[dst])):
                    return False
            if float(adj_np[src, dst]) > 0.5:
                return False
            if int(np.sum(adj_np[src, :] > 0.5)) >= int(self.max_out_connections):
                return False
            radius_m = float(getattr(self, "link_neighbor_radius_m", 0.0))
            if radius_m > 0.0:
                dx_m = (float(pos_np[src, 0]) - float(pos_np[dst, 0])) * float(self.square_size_m)
                dy_m = (float(pos_np[src, 1]) - float(pos_np[dst, 1])) * float(self.square_size_m)
                if math.sqrt(dx_m * dx_m + dy_m * dy_m) > radius_m:
                    return False
            if bool(getattr(self, "hard_mask_zone_edges", False)) and getattr(self, "zonas_proibidas", None):
                p1 = (int(round(float(pos_np[src, 0]))), int(round(float(pos_np[src, 1]))))
                p2 = (int(round(float(pos_np[dst, 0]))), int(round(float(pos_np[dst, 1]))))
                if self._linha_passa_por_zona(p1, p2, (src, dst)):
                    return False
            if bool(getattr(self, "hard_mask_left_links", False)):
                if float(pos_np[dst, 0]) < float(pos_np[src, 0]):
                    return False
            if bool(getattr(self, "hard_mask_cycles", False)):
                reach = self._reachability_bits(np.asarray(adj_np[:n, :n] > 0.5, dtype=np.bool_))
                if ((int(reach[int(dst)]) >> int(src)) & 1) != 0:
                    return False
            return True
        except Exception:
            return False

    def _activate_drone(self, drone_idx: int) -> bool:
        n = int(self.num_drones)
        drone_idx = int(drone_idx)
        if n <= 2 or drone_idx <= 0 or drone_idx >= n - 1:
            return False
        enabled = self._enabled_mask_np().copy()
        if drone_idx >= enabled.size or bool(enabled[drone_idx]):
            return False
        enabled[drone_idx] = True
        self.drone_enabled = enabled
        adj_np = np.asarray(_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]), dtype=np.float32).copy()
        adj_np[drone_idx, :] = 0.0
        adj_np[:, drone_idx] = 0.0

        if bool(getattr(self, "activate_spawn_near_line", True)):
            # ===== NASCIMENTO INTELIGENTE =====
            # O drone ativado é reposicionado para uma célula LIVRE a até +-20
            # células da linha média A->B, no QUARTO (trecho 1/4 do segmento
            # origem->destino, ao longo do eixo) com MENOS drones ativos — o
            # novo drone preenche o trecho mais vazio. E já nasce com 1 LINK
            # para o drone ativo mais próximo (orientado por x p/ manter DAG).
            try:
                pos_np = np.asarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float64).copy()
                gmax = int(self.grid_size) - 1
                p0 = pos_np[0].astype(np.float64)
                p1 = pos_np[n - 1].astype(np.float64)
                seg = p1 - p0
                seg_len2 = float(seg[0] * seg[0] + seg[1] * seg[1])
                # projeção t in [0,1] de cada drone ATIVO no segmento A->B
                act_idx_list = [i for i in range(n) if bool(enabled[i]) and i != drone_idx]
                counts = [0, 0, 0, 0]
                if seg_len2 > 1e-9:
                    for i in act_idx_list:
                        t = float(np.dot(pos_np[i] - p0, seg)) / seg_len2
                        q = min(3, max(0, int(t * 4.0)))
                        counts[q] += 1
                q_min = int(np.argmin(counts))  # quarto com menos drones
                # perpendicular unitária à linha média (p/ offset +-20 células)
                if seg_len2 > 1e-9:
                    seg_n = seg / math.sqrt(seg_len2)
                    perp = np.array([-seg_n[1], seg_n[0]], dtype=np.float64)
                else:
                    seg_n = np.array([1.0, 0.0]); perp = np.array([0.0, 1.0])
                occupied = {(int(round(pos_np[i][0])), int(round(pos_np[i][1]))) for i in range(n) if i != drone_idx}
                spawn = None
                for _try in range(40):
                    t = (q_min + float(self._rng.uniform(0.05, 0.95))) / 4.0
                    off = float(self._rng.uniform(-20.0, 20.0))
                    cand = p0 + t * seg + off * perp
                    cx = int(round(min(gmax, max(0, cand[0]))))
                    cy = int(round(min(gmax, max(0, cand[1]))))
                    if (cx, cy) not in occupied:
                        spawn = (float(cx), float(cy))
                        break
                if spawn is None:  # fallback: aceita o último candidato mesmo ocupado
                    spawn = (float(cx), float(cy))
                pos_np[drone_idx, 0] = spawn[0]
                pos_np[drone_idx, 1] = spawn[1]
                self.drone_positions = torch.tensor(pos_np, dtype=torch.float32, device=DEVICE)
                # LINK de nascimento: tenta conectar ao ativo mais próximo, mas
                # só cria a aresta se ela respeitar as mesmas restrições hard do ADD.
                if act_idx_list:
                    d2 = sorted((float((pos_np[i][0] - spawn[0]) ** 2 + (pos_np[i][1] - spawn[1]) ** 2), i) for i in act_idx_list)
                    for _, j in d2:
                        if float(pos_np[j][0]) <= spawn[0]:
                            u, v = int(j), int(drone_idx)
                        else:
                            u, v = int(drone_idx), int(j)
                        if self._birth_link_valid(u, v, adj_np, pos_np, enabled):
                            adj_np[u, v] = 1.0
                            break
            except Exception:
                pass  # em falha, mantém o comportamento antigo (posição atual, sem link)

        self.adjacency_matrix = torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)
        self._dist_matrix = None
        self._dist_pos_sig = None
        self._invalidate_edge_cross_cache()
        self._mask_version += 1
        self._mask_cache = None
        self._mask_cache_version = -1
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._sync_active_mask()
        return True

    def _deactivate_drone(self, drone_idx: int) -> bool:
        n = int(self.num_drones)
        drone_idx = int(drone_idx)
        if n <= 2 or drone_idx <= 0 or drone_idx >= n - 1:
            return False
        enabled = self._enabled_mask_np().copy()
        if drone_idx >= enabled.size or (not bool(enabled[drone_idx])):
            return False
        if int(np.count_nonzero(enabled)) <= 2:
            return False
        # TRAVA: não desativa um drone cuja remoção dos enlaces desconectaria
        # o caminho 0 -> N-1 (a máscara também bloqueia; aqui é defesa em
        # profundidade contra ações que cheguem sem passar pela máscara).
        if self._deactivate_disconnects(drone_idx):
            return False
        adj_np = np.asarray(_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]), dtype=np.float32)
        removed_links = int(np.sum(adj_np[drone_idx, :] > 0.5) + np.sum(adj_np[:, drone_idx] > 0.5))
        adj_np[drone_idx, :] = 0.0
        adj_np[:, drone_idx] = 0.0
        self.adjacency_matrix = torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)
        enabled[drone_idx] = False
        self.drone_enabled = enabled
        self.last_removed_links += int(removed_links)
        self.last_deactivated_drone = int(drone_idx)
        self._mask_version += 1
        self._mask_cache = None
        self._mask_cache_version = -1
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._sync_active_mask()
        return True

    def _compute_active_and_used_drones(self) -> tuple[int, int]:
        """Retorna (ativos, usados) para logging/terminal.

        'usados' = UNIÃO dos drones que participam de QUALQUER caminho primário
        A->B (Interpretação A): no modelo de redundância (all_paths) o tráfego é
        duplicado por todos os caminhos enumerados, então todos esses drones de
        fato transmitem. Usa _enumerate_primary_paths — A MESMA fonte de caminhos
        que o solver de Abreu e a figura do teste —, garantindo consistência
        entre 'drones usados', 'nº de caminhos' e os caminhos plotados.
        """
        try:
            n = int(max(0, min(int(getattr(self, "num_drones", 0)), int(getattr(self, "max_drones", 0) or 0 or int(getattr(self, "num_drones", 0))))))
            if n <= 0:
                return 0, 0
            enabled = self._enabled_mask_np()
            active = int(np.count_nonzero(enabled))
            if active <= 0:
                return 0, 0
            adj_np = self._current_adj_np()
            used_set = set()
            # UNIÃO de todos os caminhos primários (mesma enumeração do Abreu)
            try:
                primary = self._enumerate_primary_paths(adj_np, 0, n - 1) if n >= 2 else []
            except Exception:
                primary = []
            if primary:
                for p in primary:
                    for x in p:
                        xi = int(x)
                        if 0 <= xi < enabled.size and bool(enabled[xi]):
                            used_set.add(xi)
                used = int(max(2, len(used_set)))
            else:
                # Fallback: um caminho qualquer (BFS), como antes
                out_neighbors = [[] for _ in range(n)]
                ei, ej = np.nonzero(adj_np > 0)
                for i, j in zip(ei.tolist(), ej.tolist()):
                    out_neighbors[int(i)].append(int(j))
                path_nodes = self._get_any_path_nodes(0, n - 1, adj_np, out_neighbors=out_neighbors) if n >= 2 else [0]
                if len(path_nodes) >= 2:
                    used = int(len(set(int(x) for x in path_nodes if 0 <= int(x) < enabled.size and bool(enabled[int(x)]))))
                    used = max(2, used)
                else:
                    deg = (adj_np > 0).sum(axis=0) + (adj_np > 0).sum(axis=1)
                    used = int(np.count_nonzero((deg > 0) & enabled))
                    if active > 0:
                        used = max(1, used)
            used = int(max(0, min(active, used)))
            return active, used
        except Exception:
            active = int(np.count_nonzero(self._enabled_mask_np())) if getattr(self, 'drone_enabled', None) is not None else int(self.num_drones)
            return active, active

    def _sync_active_mask(self):
        """Mantém uma máscara de drones ativos compatível com o wrapper multiagente."""
        try:
            mask = torch.zeros((int(self.max_drones),), dtype=torch.float32, device=DEVICE)
            n = int(max(0, min(int(self.num_drones), int(self.max_drones))))
            enabled = self._enabled_mask_np()
            if n > 0 and enabled.size >= n:
                mask[:n] = torch.tensor(enabled[:n].astype(np.float32), dtype=torch.float32, device=DEVICE)
            self.active_mask = mask
        except Exception:
            self.active_mask = None

    @staticmethod
    def _exact_array_signature(arr: np.ndarray):
        arr_c = np.ascontiguousarray(arr)
        return (tuple(arr_c.shape), arr_c.dtype.str, arr_c.tobytes())

    def _cache_put_bounded(self, cache: dict, key, value, limit: int):
        if len(cache) >= int(limit):
            cache.clear()
        cache[key] = value
        return value

    def _evaluate_graph_metrics(self, pos_np: np.ndarray, adj_np: np.ndarray, use_abreu: bool = True) -> dict:
        """Avalia QoS do grafo atual/candidato sem alterar o estado do ambiente.

        SEMPRE usa o solver completo de Abreu (decisão de projeto): a antiga
        aproximação leve `throughput_fast` subestimava a vazão — calculava o
        gargalo de UM caminho (~6 Mbps) em vez do goodput agregado real (~12),
        o que travava o KPI de vazão e impedia o `success`. Para eliminar esse
        erro de medição, o parâmetro `use_abreu` é ignorado e o Abreu é usado em
        todas as chamadas. A aproximação leve permanece apenas como FALLBACK se
        o solver falhar/retornar None (para não interromper o treino).

        DESEMPENHO: o cache (_graph_metrics_cache) reaproveita topologias já
        avaliadas; com cenário fixo e política convergente, absorve boa parte do
        custo. Se ficar lento demais, considerar a aproximação SOMENTE na
        recompensa por step, mas com a vazão corrigida para AGREGAR caminhos.
        """
        use_abreu = True  # força sempre Abreu, ignorando o argumento
        pos_arr = np.asarray(pos_np[:, :2], dtype=np.float32, order="C")
        adj_bin = (np.asarray(adj_np, dtype=np.float32, order="C") > 0.0).astype(np.uint8, copy=False)
        cache_key = (self._exact_array_signature(pos_arr), self._exact_array_signature(adj_bin))
        cached = self._graph_metrics_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        n = int(adj_bin.shape[0])
        metrics = {
            "path_exists": False,
            "reliability": 0.0,
            "reliability_block": 0.0,
            "delivery_ratio": 0.0,
            "reliability_hybrid": 0.0,
            "throughput": 0.0,
            "latency_ms": float('inf'),
            "loss_rate": 1.0,
            "max_feasible_rate_mbps": 0.0,
            "num_paths": 0,
            "max_hop_m": 0.0,
        }
        if n < 2:
            return dict(self._cache_put_bounded(self._graph_metrics_cache, cache_key, dict(metrics), self._graph_metrics_cache_limit))

        out_neighbors = [[] for _ in range(n)]
        ei, ej = np.nonzero(adj_bin > 0)
        for i, j in zip(ei.tolist(), ej.tolist()):
            out_neighbors[int(i)].append(int(j))
        path_exists = bool(self._has_path(0, n - 1, adj_bin, out_neighbors=out_neighbors))
        metrics["path_exists"] = path_exists
        if not path_exists:
            return dict(self._cache_put_bounded(self._graph_metrics_cache, cache_key, dict(metrics), self._graph_metrics_cache_limit))

        path_nodes = self._get_any_path_nodes(0, n - 1, adj_bin, out_neighbors=out_neighbors)
        if len(path_nodes) >= 2:
            dists = []
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                d = float(np.linalg.norm((pos_arr[int(b), :2] - pos_arr[int(a), :2]) * float(self.square_size_m)))
                dists.append(d)
            metrics["max_hop_m"] = float(max(dists) if dists else 0.0)

        if bool(use_abreu) and self.abreu_enabled:
            mm = self._evaluate_abreu_modified(adj_bin, pos_arr, compute_max_feasible=False)
            if mm is not None:
                metrics.update({
                    "reliability": float(getattr(mm, 'reliability', 0.0)),
                    "reliability_block": float(getattr(mm, 'reliability_block', getattr(mm, 'reliability', 0.0))),
                    "delivery_ratio": float(getattr(mm, 'delivery_ratio', 0.0)),
                    "reliability_hybrid": float(getattr(mm, 'reliability_hybrid', getattr(mm, 'reliability', 0.0))),
                    "throughput": float(getattr(mm, 'goodput_mbps', 0.0)),
                    "latency_ms": float(getattr(mm, 'delay_ms', float('inf'))),
                    "loss_rate": float(getattr(mm, 'loss_rate', 1.0)),
                    "max_feasible_rate_mbps": float(getattr(mm, 'max_feasible_rate_mbps', 0.0)),
                    "num_paths": int(len(getattr(mm, 'paths', []))),
                    # métricas de rede adicionais (já computadas pelo solver de
                    # Abreu; expostas para avaliação/relatório da tese):
                    "num_hops": int(getattr(mm, 'latency_max_hops', 0)),
                    "rho_max": float(getattr(mm, 'rho_max', getattr(mm, 'max_rho', 0.0))),
                    "interference_load": float(getattr(mm, 'interference_load', 0.0)),
                })
                return dict(self._cache_put_bounded(self._graph_metrics_cache, cache_key, dict(metrics), self._graph_metrics_cache_limit))

        # ---------------------------------------------------------------
        # Aproximação leve para o loop de treino (use_abreu=False).
        #
        # IMPORTANTE: esta aproximação precisa refletir a MESMA física de
        # multicaminhos/redundância usada na avaliação terminal de Abreu, caso
        # contrário a recompensa por step otimiza um objetivo diferente do que
        # define sucesso (era a causa principal de não-convergência).
        #
        # Modelo de redundância (duplicação de pacotes em caminhos paralelos):
        #   - rel_k       = Π_{enlaces de k} p_succ(enlace)        (confiabilidade do caminho k)
        #   - reliability = 1 - Π_k (1 - rel_k)                    (entrega por QUALQUER caminho)
        # Isso premia explicitamente adicionar caminhos disjuntos (mais drones),
        # exatamente o comportamento desejado.
        #
        # Latência: a duplicação só termina quando a cópia mais lenta chega; usa
        # a maior latência entre os caminhos redundantes (coerente com
        # _extract_longest_redundant_route_latency no caminho terminal).
        # Throughput: limitado pelo gargalo (menor taxa PHY entre os enlaces do
        # melhor caminho), saturado por traffic_rate e throughput_max.
        # ---------------------------------------------------------------
        dist_m = None
        try:
            dist_m = self._ensure_dist_matrix()
        except Exception:
            dist_m = None

        # Enumera caminhos simples 0 -> N-1 (mesmo enumerador usado por Abreu).
        primary_paths = []
        try:
            primary_paths = list(self._enumerate_primary_paths(adj_bin, 0, n - 1))
        except Exception:
            primary_paths = []
        # Fallback: ao menos uma rota BFS, garantindo coerência com path_exists.
        if not primary_paths:
            try:
                bfs_path = self._get_any_path_nodes(0, n - 1, adj_bin, out_neighbors=out_neighbors)
            except Exception:
                bfs_path = []
            if len(bfs_path) >= 2:
                primary_paths = [tuple(int(x) for x in bfs_path)]

        def _hop_dist(a: int, b: int) -> float:
            if dist_m is not None:
                try:
                    return float(dist_m[int(a), int(b)])
                except Exception:
                    pass
            return float(np.linalg.norm((pos_arr[int(b), :2] - pos_arr[int(a), :2]) * float(self.square_size_m)))

        # Tempo por salto coerente com a aproximação anterior (apenas shaping).
        PER_HOP_MS = 4.1284

        if primary_paths:
            path_rels = []
            path_latencies = []
            path_bottleneck_rate = []
            max_hop_global = 0.0

            # Para evitar custo excessivo no loop de treino, limita o nº de
            # caminhos considerados na combinação de confiabilidade.
            max_paths_fast = int(min(len(primary_paths), max(2, int(getattr(self, "abreu_max_paths", 8)))))

            for p in primary_paths[:max_paths_fast]:
                if len(p) < 2:
                    continue
                rel_k = 1.0
                worst_rate = float(self.phy_rate_mbps)
                hops_k = 0
                for a, b in zip(p[:-1], p[1:]):
                    d = _hop_dist(a, b)
                    max_hop_global = max(max_hop_global, d)
                    try:
                        pk = float(self._prob_link(d))
                    except Exception:
                        pk = 0.0
                    rel_k *= pk
                    hops_k += 1
                    # taxa PHY do enlace (gargalo do caminho)
                    try:
                        x = d / max(1.0, float(self.carrier_sense_range_m))
                        ladder = [54.0, 48.0, 36.0, 24.0, 18.0, 12.0, 9.0, 6.0]
                        thr = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
                        r = ladder[-1]
                        for ti, tv in enumerate(thr):
                            if x <= tv:
                                r = ladder[ti]
                                break
                        worst_rate = min(worst_rate, float(r))
                    except Exception:
                        pass
                rel_k = float(max(0.0, min(1.0, rel_k)))
                path_rels.append(rel_k)
                path_latencies.append(float(hops_k) * PER_HOP_MS)
                path_bottleneck_rate.append(float(worst_rate))

            if path_rels:
                # Redundância: entrega se QUALQUER caminho entregar.
                prod_fail = 1.0
                for rk in path_rels:
                    prod_fail *= (1.0 - rk)
                reliability_redundant = float(max(0.0, min(1.0, 1.0 - prod_fail)))

                # Latência da redundância. Consistente com o modo Abreu (ver
                # latency_from_longest_route): sob 'all_paths' a PRIMEIRA cópia
                # que chega entrega, logo o tempo de entrega é o MÍNIMO entre os
                # caminhos com confiabilidade não-desprezível. O legado usava
                # max() — a cópia mais lenta — herdado dos modelos de espaço
                # livre. Mantido sob a mesma flag para não misturar semânticas
                # entre o caminho rápido e o solver.
                _legacy_lat = bool(getattr(self, "latency_from_longest_route", False))
                useful = [(lat, rk) for lat, rk in zip(path_latencies, path_rels) if rk > 1e-3]
                if useful:
                    _lats = [lat for lat, _ in useful]
                    latency_fast = float(max(_lats) if _legacy_lat else min(_lats))
                else:
                    latency_fast = float(max(path_latencies) if path_latencies else PER_HOP_MS) \
                        if _legacy_lat else float(min(path_latencies) if path_latencies else PER_HOP_MS)

                # Throughput: melhor gargalo entre os caminhos confiáveis,
                # saturado por demanda e teto.
                best_rate = max(path_bottleneck_rate) if path_bottleneck_rate else float(self.phy_rate_mbps)
                throughput_fast = float(min(float(self.traffic_rate_mbps),
                                            float(self.throughput_max_mbps),
                                            float(best_rate)))

                metrics["max_hop_m"] = float(max_hop_global)
                metrics.update({
                    "reliability": reliability_redundant,
                    "reliability_block": reliability_redundant,
                    "delivery_ratio": reliability_redundant,
                    "reliability_hybrid": reliability_redundant,
                    "throughput": throughput_fast,
                    "latency_ms": latency_fast,
                    "loss_rate": float(1.0 - reliability_redundant),
                    "max_feasible_rate_mbps": float(best_rate),
                    "num_paths": int(len(path_rels)),
                })

        return dict(self._cache_put_bounded(self._graph_metrics_cache, cache_key, dict(metrics), self._graph_metrics_cache_limit))

    def _min_drone_targets(self) -> tuple[float, float, float]:
        """Return effective QoS targets.

        RELIABILITY_TARGET, MAX_DELAY_MS_DEFAULT and THROUGHPUT_MAX_MBPS are the
        authoritative defaults. They are overridden only by explicit trainer/env
        parameters: reliability_target, max_delay_ms and throughput_max_mbps.
        Legacy Abreu parameters no longer raise or replace these targets.
        """
        target_goodput = float(max(0.0, min(float(self.traffic_rate_mbps), float(self.throughput_max_mbps))))
        target_rel = float(self.reliability_target)
        target_delay = float(self.max_delay_ms)
        return float(target_goodput), float(target_rel), float(target_delay)

    def _effective_goodput_target(self) -> float:
        """Alvo mínimo efetivo de goodput coerente com a taxa de entrega.

        No modelo Abreu, o goodput agregado é afetado pela entrega efetiva.
        Exigir `traffic_rate_mbps` integral enquanto também se aceita
        delivery_ratio < 1.0 cria um alvo implícito mais rígido que o KPI.
        Portanto o alvo físico consistente é a demanda oferecida multiplicada
        pelo delivery_ratio_target.
        """
        target_goodput, _target_rel, _target_delay = self._min_drone_targets()
        delivery_target = float(max(0.0, min(1.0, getattr(self, "delivery_ratio_target", 0.999))))
        return float(target_goodput) * float(delivery_target)

    def _success_targets_met(self, metrics: dict) -> bool:
        target_goodput, target_rel, target_delay = self._min_drone_targets()
        if not bool(metrics.get("path_exists", False)):
            return False
        rel_block = float(metrics.get("reliability_block", metrics.get("reliability", 0.0)))
        delivery_ratio = float(metrics.get("delivery_ratio", 0.0))
        if rel_block + 1e-12 < float(target_rel):
            return False
        if delivery_ratio + 1e-12 < float(getattr(self, "delivery_ratio_target", 0.999)):
            return False
        throughput = float(metrics.get("throughput", 0.0))
        # Porta de vazão CONSISTENTE com o KPI de entrega. Como a física de
        # Abreu acopla goodput à entrega efetiva, exigir goodput >= target_goodput
        # EXATO implicava entrega/confiabilidade praticamente iguais a 1.0.
        # Agora a vazão exigida é target_goodput * delivery_ratio_target.
        effective_goodput_target = float(self._effective_goodput_target())
        if throughput + 1e-9 < effective_goodput_target:
            return False
        if throughput > float(getattr(self, "throughput_max_mbps", THROUGHPUT_MAX_MBPS)) + 1e-9:
            return False
        if float(metrics.get("latency_ms", float('inf'))) > float(target_delay) + 1e-9:
            return False
        return True

    def _extract_longest_redundant_route_latency(self, mm) -> tuple[float, int]:
        """Latência terminal baseada na rota redundante ativa com maior número de saltos.

        Em empate de hops, usa a maior latência entre as empatadas.
        Se não houver path_metrics disponíveis, cai para worst_delay_ms/delay_ms.
        """
        default_delay = float(getattr(mm, 'worst_delay_ms', getattr(mm, 'delay_ms', 0.0)))
        best_delay = float(default_delay)
        best_hops = 0
        try:
            all_paths = []
            for fm in getattr(mm, 'flows', []) or []:
                for apm in getattr(fm, 'path_metrics', []) or []:
                    pm = getattr(apm, 'path_metrics', None)
                    if pm is not None:
                        all_paths.append(pm)
            if not all_paths:
                all_paths = list(getattr(mm, 'paths', []) or [])
            for pm in all_paths:
                path = list(getattr(pm, 'path', []) or [])
                hops = max(0, len(path) - 1)
                delay = float(getattr(pm, 'delay_ms', default_delay))
                if (hops > best_hops) or (hops == best_hops and delay > best_delay):
                    best_hops = int(hops)
                    best_delay = float(delay)
        except Exception:
            pass
        return float(best_delay), int(best_hops)

    # =========================
    # Estado inicial

    # =========================
    # Estado inicial
    # =========================
    def _analytic_min_line_hops(self) -> int:
        """Deriva ANALITICAMENTE, com o solver de Abreu e os parâmetros do
        próprio env, o número de hops da menor linha reta A->B que chega
        PRÓXIMO de atender os KPIs sem fechá-los: encontra o menor n_hops cujo
        caminho único em linha atende (rel >= alvo, goodput >= alvo efetivo,
        latência <= limite) e retorna n_hops - 1. Ex. (2km, phy 100, retries 6):
        6 hops (7 drones) atende (rel 1.0, lat 6ms); a semente usa 5 hops
        (6 drones): rel ~0.74, vazão ~7.4, lat ~271ms — toca os três limites
        sem fechar nenhum, forçando a política a ATIVAR drones para completar.
        Resultado cacheado (os parâmetros físicos não mudam entre episódios)."""
        cached = getattr(self, "_min_line_cache", None)
        if cached is not None:
            return int(cached)
        n_seed = 5  # fallback medido para os parâmetros padrão (2km, phy 100)
        # CONSISTÊNCIA COM A RECOMPENSA: a calibração antiga usava
        # _solve_single_path com ber_per_link=None e interference_load=0
        # (modelo otimista, com ganho de retransmissão), enquanto a recompensa
        # avalia pelo caminho rápido de _evaluate_graph_metrics: por caminho,
        # rel = PRODUTO de _prob_link(d_hop) (sem retransmissão), latência
        # = hops * PER_HOP_MS e throughput = min(demanda, teto, gargalo da
        # escada PHY). Resultado: a linha "que quase atende" tinha rel ~0.83
        # na calibração, mas ~0.01 na métrica que treina a política.
        # Além disso, uma linha ÚNICA nunca fecha os KPIs nessa métrica
        # (delivery_ratio_target=0.999 exige redundância e o gargalo PHY
        # limita o throughput), então o critério "menor linha que atende - 1"
        # é vazio. O critério fiel à intenção ("chega PRÓXIMO de atender sem
        # fechar") passa a ser: a MENOR linha cuja confiabilidade, NA MESMA
        # métrica da recompensa, alcança seed_line_rel_frac (padrão 0.70) do
        # alvo, respeitando a latência máxima.
        try:
            span_m = float(int(self.grid_size) - 1) * float(getattr(self, "square_size_m", 20.0))
            target_rel = float(getattr(self, "reliability_target", 0.995))
            max_lat = float(getattr(self, "max_delay_ms", 300.0))
            rel_frac = float(getattr(self, "seed_line_rel_frac", 0.70))
            rel_goal = max(0.0, min(1.0, rel_frac * target_rel))
            per_hop_ms = 4.1284  # mesmo PER_HOP_MS do caminho rápido de _evaluate_graph_metrics
            n_ok = None
            for nh in range(3, 16):
                hop_m = span_m / float(nh)
                p = float(self._prob_link(hop_m))  # MESMA física por enlace da recompensa
                rel = p ** nh
                lat = float(nh) * per_hop_ms
                if rel + 1e-12 >= rel_goal and lat <= max_lat + 1e-9:
                    n_ok = nh
                    break
            if n_ok is not None:
                n_seed = max(3, int(n_ok))
            # REFINAMENTO SOB SOMBREAMENTO: o modelo produto (acima) ignora
            # sombra e interferência; sob sigma=5.5 dB a rel medida da linha
            # calibrada caía para mediana ~0.4 (observado em treino). Partindo
            # do palpite do produto, busca a MENOR linha cuja rel MEDIANA,
            # medida com a métrica real da recompensa sob K sorteios fixos de
            # sombra, alcança rel_goal (respeitando a latência mediana).
            try:
                shadows = self._calibration_shadow_draws(k=11)
                n = int(self.num_drones)
                gmax = int(self.grid_size) - 1

                def _line_median(nh: int):
                    if nh + 1 > n or nh < 3:
                        return None
                    pos = np.zeros((n, 2), dtype=np.float32)
                    adj = np.zeros((n, n), dtype=np.float32)
                    pos[0] = (0.0, gmax / 2.0); pos[n - 1] = (float(gmax), gmax / 2.0)
                    prev = 0; idx = 1
                    for h in range(1, nh):
                        pos[idx] = (gmax * h / float(nh), gmax / 2.0)
                        adj[prev, idx] = 1.0; prev = idx; idx += 1
                    adj[prev, n - 1] = 1.0
                    ms = self._measure_topology_under_shadows(pos, adj, shadows)
                    rels = sorted(float(m.get("reliability_block", 0.0)) for m in ms if m)
                    lats = sorted(float(m.get("latency_ms", 1e9)) for m in ms if m)
                    if not rels:
                        return None
                    return rels[len(rels) // 2], lats[len(lats) // 2]

                start_nh = max(3, int(n_seed))
                refined = None
                for nh in range(start_nh, min(n - 1, start_nh + 14) + 1):
                    r = _line_median(nh)
                    if r is None:
                        continue
                    rel_med, lat_med = r
                    if rel_med + 1e-12 >= rel_goal and lat_med <= max_lat + 1e-9:
                        refined = nh
                        break
                if refined is not None:
                    n_seed = max(3, int(refined))
                self._min_line_refined = bool(refined is not None)
            except Exception:
                self._min_line_refined = False  # mantém o palpite do modelo produto
        except Exception:
            self._min_line_refined = False  # mantém o fallback medido
        # Só cacheia quando o refinamento sob sombra rodou de fato. Chamadas
        # muito precoces (durante __init__, antes de todos os atributos
        # existirem) caíam no fallback e o CACHEAVAM para sempre — o warm start
        # ficava calibrado pelo modelo produto otimista o processo inteiro.
        if bool(getattr(self, "_min_line_refined", False)):
            self._min_line_cache = int(n_seed)
        return int(n_seed)

    def _calibration_shadow_draws(self, k: int = 3) -> list:
        """K matrizes de sombreamento com seeds FIXOS para calibração.

        As calibrações analíticas (linha mínima e mínimo de drones p/ QoS) não
        podem usar sombra zero: sob sigma=5.5 dB a métrica real fica muito
        abaixo da prevista (observado em treino: warm start previsto 0.83,
        medido 0.11-1.00 com média 0.41; baseline previsto sem sombra=8 drones
        levou a política a podar até rel~0.11). Avaliar sob K sorteios fixos e
        decidir pela mediana/maioria torna a calibração representativa do que a
        recompensa vai medir de fato, mantendo-a determinística entre processos.
        """
        sigma = float(getattr(self, "link_shadow_sigma_db", 5.5))
        nmax = int(getattr(self, "max_drones", self.num_drones))
        draws = []
        rng = np.random.default_rng(987654321)
        for _ in range(max(1, int(k))):
            if sigma > 0.0:
                raw = rng.normal(0.0, sigma, size=(nmax, nmax)).astype(np.float32)
                sh = np.triu(raw, 1)
                sh = sh + sh.T
            else:
                sh = np.zeros((nmax, nmax), dtype=np.float32)
            draws.append(sh)
        return draws

    def _measure_topology_under_shadows(self, pos: np.ndarray, adj: np.ndarray, shadows: list) -> list:
        """Avalia (pos, adj) com a métrica REAL da recompensa sob cada matriz de
        sombra, com troca/restauração integral de estado e limpeza dos caches
        dependentes de sombra entre sorteios."""
        results = []
        pos_backup = getattr(self, "drone_positions", None)
        dm_backup, ds_backup = getattr(self, "_dist_matrix", None), getattr(self, "_dist_pos_sig", None)
        shadow_backup = getattr(self, "_shadow_db", None)

        def _clear_shadow_caches():
            try:
                self._prob_cache.clear(); self._ber_path_cache.clear()
                self._graph_metrics_cache.clear(); self._abreu_eval_cache.clear()
                self._phy_path_cache.clear()
            except Exception:
                pass

        try:
            self.drone_positions = torch.tensor(pos, dtype=torch.float32, device=DEVICE)
            self._dist_matrix = None; self._dist_pos_sig = None
            for sh in shadows:
                self._shadow_db = sh
                _clear_shadow_caches()
                try:
                    results.append(self._evaluate_graph_metrics(pos, adj))
                except Exception:
                    results.append({})
        finally:
            self.drone_positions = pos_backup
            self._dist_matrix, self._dist_pos_sig = dm_backup, ds_backup
            self._shadow_db = shadow_backup
            _clear_shadow_caches()
        return results

    def _analytic_min_qos_drones(self) -> int:
        """Estima o número TÍPICO mínimo de drones ativos para QoS pleno neste
        cenário: para cada um de K sorteios FIXOS de sombreamento, encontra o
        menor tronco+bypasses (avaliado com a MESMA métrica da recompensa) que
        fecha os KPIs SOB AQUELE sorteio; o baseline é a MEDIANA desses mínimos.

        Semântica: modela uma política ADAPTATIVA, que posiciona a redundância
        conforme o fade de cada episódio. Um template fixo julgado por maioria
        de sorteios é pessimista demais (nenhuma topologia fixa fecha QoS na
        maioria dos episódios sob sigma=5.5 dB), enquanto sombra zero é otimista
        demais (baseline 8 sob a física phy=100/CS=200 levou a política a podar
        até rel~0.11). A mediana dos mínimos por sorteio é o meio-termo fiel.

        Usos: baseline do custo de excedente, gate de poda e teto de ativação.
        Cacheado por processo; fallback = num_drones (custo nunca dispara)."""
        cached = getattr(self, "_min_qos_drones_cache", None)
        if cached is not None:
            return int(cached)
        n = int(self.num_drones)
        result = n  # fallback seguro: QoS tipicamente inatingível -> custo nunca dispara
        try:
            gmax = int(self.grid_size) - 1
            # k=21 (era 5): um percentil 85 sobre 5 pontos é praticamente o
            # máximo amostral e varia muito entre seeds. 21 sorteios dão ~3
            # pontos acima do percentil, o bastante para o valor ser estável.
            # Custo: a calibração roda UMA vez no processo pai (~3,5 s com k=5),
            # então ~15 s — irrelevante frente a blocos de 5M steps.
            k = int(getattr(self, "calibration_shadow_k", 21))
            shadows = self._calibration_shadow_draws(k=k)

            def _build(nh: int, nb: int):
                pos = np.zeros((n, 2), dtype=np.float32)
                adj = np.zeros((n, n), dtype=np.float32)
                pos[0] = (0.0, gmax / 2.0); pos[n - 1] = (float(gmax), gmax / 2.0)
                chain = [0]
                idx = 1
                for h in range(1, nh):
                    pos[idx] = (gmax * h / float(nh), gmax / 2.0)
                    adj[chain[-1], idx] = 1.0
                    chain.append(idx); idx += 1
                adj[chain[-1], n - 1] = 1.0
                chain.append(n - 1)
                # nb bypasses espalhados ao longo do tronco (redundância local)
                span = max(2, (len(chain) - 2) // max(1, nb))
                for b in range(nb):
                    i = min(1 + b * span, len(chain) - 3)
                    u, wnode = chain[i], chain[i + 2]
                    pos[idx] = ((pos[u][0] + pos[wnode][0]) / 2.0, gmax / 2.0 + 3.0)
                    adj[u, idx] = 1.0; adj[idx, wnode] = 1.0
                    idx += 1
                return pos, adj

            def _min_for_shadow(sh) -> int | None:
                # menor total de drones que fecha QoS sob ESTE sorteio
                for nh in range(4, n - 1, 2):
                    for nb in (1, 2, 3):
                        if (nh - 1) + 2 + nb > n:
                            continue
                        pos, adj = _build(nh, nb)
                        ms = self._measure_topology_under_shadows(pos, adj, [sh])
                        if ms and ms[0] and bool(self._success_targets_met(ms[0])):
                            return (nh - 1) + 2 + nb
                return None

            # RECALIBRAÇÃO POR PERCENTIL (era MEDIANA).
            # A mediana devolve o mínimo que fecha QoS no sorteio TÍPICO — por
            # construção, o percentil 50 do sombreamento. Medido com o solver
            # sobre este mesmo template (25 sorteios, sigma=5.5): mediana=12,
            # percentil 85=14. Com baseline 12 o custo por drone excedente
            # (DRONE_ACTIVE_COST_AFTER_QOS) passa a cobrar exatamente a
            # redundância que torna o cenário robusto ao fade, empurrando a
            # política para o regime frágil: 12-13 drones fecham QoS em ~62%
            # dos episódios, contra ~83% com saltos mais curtos. O baseline
            # deve ser o mínimo ROBUSTO, não o mínimo típico.
            #
            # Sorteios SEM solução no grid de busca contam como `n` (precisam de
            # mais que o teto testado). Descartá-los, como antes, removia
            # justamente os episódios mais difíceis da estatística e enviesava o
            # baseline para baixo — o oposto do que um critério de robustez quer.
            solved = 0
            mins = []
            for sh in shadows:
                m = _min_for_shadow(sh)
                if m is None:
                    mins.append(int(n))
                else:
                    mins.append(int(m))
                    solved += 1
            q = float(getattr(self, "calibration_min_drones_quantile", 0.85))
            q = min(1.0, max(0.5, q))
            # exige que a MAIORIA dos sorteios tenha solução; senão, fallback n
            if solved >= (len(shadows) // 2) + 1:
                result = int(math.ceil(float(np.percentile(np.asarray(mins, dtype=np.float64), q * 100.0))))
        except Exception:
            result = n
        # TETO OPERACIONAL. O percentil escolhe o mínimo ROBUSTO ao fade, mas nada
        # nele sabe que a rede tem um limite de CAPACIDADE. Com carrier_sense=200 m
        # todos os enlaces dividem um único domínio de colisão: medido com o solver
        # (zona 12x12, 200 sorteios), 14 drones fecham os KPIs em ~50% dos casos,
        # enquanto 16 e 18 fecham em 0,0% — rho explode e a fila satura.
        #
        # Sem este teto a calibração devolvia 16, ou seja, o custo por drone
        # excedente só começava a cobrar ACIMA da faixa em que a rede funciona:
        # na prática, pressão de poda nenhuma. Nos logs do run D a correlação
        # active_drones x qos é -0,965 (13 drones -> qos 0,996; >15,5 -> 0,483).
        #
        # O teto é um parâmetro físico (capacidade do meio), não estatístico —
        # por isso é aplicado DEPOIS do percentil, e não embutido nele.
        _cap = int(getattr(self, "calibration_min_drones_cap", 0) or 0)
        if _cap > 0 and result > _cap:
            print(f"[CALIB] baseline_qos={result} acima do teto operacional {_cap} "
                  f"(capacidade do meio); limitado a {_cap}.", flush=True)
            result = _cap
        self._min_qos_drones_cache = int(max(3, min(n, result)))
        return int(self._min_qos_drones_cache)

    def _init_state(self):
        """Inicializa uma topologia conectada (warm start), opcionalmente
        ALEATORIZADA de forma reprodutível e mantendo a viabilidade dos KPIs.

        Estrutura: K caminhos paralelos de A (nó 0) a B (nó N-1), com saltos de
        ~seed_target_hop_m. Com seed_randomness=0 a topologia é fixa/determinística
        (mesma de antes). Com seed_randomness>0, varia entre episódios em três
        eixos seguros — posição vertical de A/B, número de caminhos K, e jitter
        posicional dos intermediários — usando self._rng (já semeado pela semente
        do reset), portanto REPRODUTÍVEL.

        Viabilidade GARANTIDA por verificação: gera a topologia e, se o maior
        salto exceder o alcance de enlace confiável (~300 m), reduz a intensidade
        efetiva pela metade e regenera, até passar (r=0 é sempre viável). Assim os
        KPIs de confiabilidade do warm-start são preservados em qualquer
        intensidade. As zonas proibidas NÃO são evitadas: o estado inicial pode
        violar zonas (drones/links dentro delas) e cabe ao agente realocar.
        """
        n = int(self.num_drones)
        gmax = int(self.grid_size) - 1
        center_y = self.grid_size // 2
        square_m = float(getattr(self, "square_size_m", SQUARE_SIZE_M_DEFAULT))
        target_hop_m = float(getattr(self, "seed_target_hop_m", 250.0))
        hop_cells = max(1, int(round(target_hop_m / max(1e-6, square_m))))
        span_cells = float(gmax)
        hops_per_path = max(2, int(math.ceil(span_cells / hop_cells)))
        inter_per_path = hops_per_path - 1
        n_inter = max(0, n - 2)
        K_target = int(getattr(self, "seed_num_paths", 3))

        def _generate(r_eff: float):
            """Gera (pos, path_node_lists) para uma intensidade efetiva r_eff,
            consumindo self._rng quando r_eff>0 (reprodutível)."""
            rng = self._rng
            pos = np.zeros((n, 2), dtype=np.float32)

            # Eixo 3: deslocamento vertical de A/B (proporcional a r_eff).
            amp = r_eff * (self.grid_size // 4)
            a_dy = int(rng.randint(-int(amp), int(amp) + 1)) if amp >= 1 else 0
            b_dy = int(rng.randint(-int(amp), int(amp) + 1)) if amp >= 1 else 0
            a_y = max(0, min(gmax, center_y + a_dy))
            b_y = max(0, min(gmax, center_y + b_dy))
            if n > 0:
                pos[0, 0] = 0.0; pos[0, 1] = float(a_y)
            if n > 1:
                pos[n - 1, 0] = float(gmax); pos[n - 1, 1] = float(b_y)

            # Eixo 2: K variável (só aumenta; mais caminhos = mais confiável).
            if inter_per_path > 0:
                K_fit = max(1, n_inter // inter_per_path)
                K_base = max(1, min(K_target, K_fit))
                if r_eff > 0.0 and K_fit > K_base:
                    k_extra_max = int(round(r_eff * (K_fit - K_base)))
                    K = K_base + (int(rng.randint(0, k_extra_max + 1)) if k_extra_max > 0 else 0)
                else:
                    K = K_base
            else:
                K = 1

            lane_amp = float(self.grid_size // (K + 1))
            jitter = r_eff * 6.0  # amplitude de jitter em células

            idx = 1
            occupied = set()
            if n > 0:
                occupied.add((int(pos[0, 0]), int(pos[0, 1])))
            if n > 1:
                occupied.add((int(pos[n - 1, 0]), int(pos[n - 1, 1])))

            path_node_lists = []
            for k in range(K):
                offset = (k - (K - 1) / 2.0) * lane_amp
                this_path = [0]
                for h in range(1, hops_per_path):
                    if idx >= n - 1:
                        break
                    frac = h / hops_per_path
                    cx = int(round(frac * span_cells))
                    base_y = a_y + (b_y - a_y) * frac
                    env = math.sin(math.pi * frac)
                    cy = base_y + offset * env
                    if jitter > 0:
                        cx += int(round(rng.uniform(-1.0, 1.0) * jitter))
                        cy += rng.uniform(-1.0, 1.0) * jitter
                    cx = max(1, min(gmax - 1, int(round(cx))))
                    cy = max(0, min(gmax, int(round(cy))))
                    shift = 0
                    while (cx, cy) in occupied and shift < self.grid_size:
                        shift += 1
                        cy2 = max(0, min(gmax, cy + shift))
                        if (cx, cy2) not in occupied:
                            cy = cy2
                            break
                    pos[idx, 0] = float(cx)
                    pos[idx, 1] = float(cy)
                    occupied.add((cx, cy))
                    this_path.append(idx)
                    idx += 1
                this_path.append(n - 1)
                path_node_lists.append(this_path)

            # Sobras de drones (sem links): distribuídas ALEATORIAMENTE numa
            # faixa de +-40 células em torno da linha média (Y) entre o primeiro
            # (A) e o último (B) drone, espalhadas ao longo de X. Antes ficavam
            # empilhadas na borda direita (gmax-1), criando acúmulo perto de B.
            # Espalhá-las as deixa em posições mais plausíveis para a política
            # eventualmente integrá-las. Usa self._rng (reprodutível por seed).
            mid_y = 0.5 * (float(pos[0, 1]) + float(pos[n - 1, 1]))
            band = 40.0  # +-40 células em torno da linha média
            y_lo = int(max(0, math.floor(mid_y - band)))
            y_hi = int(min(gmax, math.ceil(mid_y + band)))
            while idx < n - 1:
                cx = int(self._rng.randint(1, gmax)) if gmax > 1 else 0
                cy = int(self._rng.randint(y_lo, y_hi + 1)) if y_hi >= y_lo else int(round(mid_y))
                shift = 0
                while (cx, cy) in occupied and shift < self.grid_size:
                    shift += 1
                    cy = max(0, min(gmax, cy + 1))
                    if (cx, cy) not in occupied:
                        break
                    cx = max(1, min(gmax - 1, cx + (1 if (shift % 2 == 0) else 0)))
                pos[idx, 0] = float(cx)
                pos[idx, 1] = float(cy)
                occupied.add((cx, cy))
                idx += 1

            return pos, path_node_lists

        def _max_hop_cells(pos, paths):
            mx = 0.0
            for p in paths:
                for a, b in zip(p[:-1], p[1:]):
                    d = math.hypot(float(pos[b, 0] - pos[a, 0]), float(pos[b, 1] - pos[a, 1]))
                    if d > mx:
                        mx = d
            return mx

        # Viabilidade: a redução do r efetivo só atua se o maior salto exceder o
        # salto-alvo COM FOLGA (seed_viability_slack). Assim os saltos-alvo
        # deliberados (ex.: ~330 m na dificuldade média) são PRESERVADOS — a
        # redução protege apenas contra saltos absurdos gerados pelo jitter
        # aleatório, não "conserta" a dificuldade intencional. (Antes usava um
        # limiar fixo de 300 m, que anulava qualquer salto-alvo acima disso.)
        viab_hop_cells = (target_hop_m * float(getattr(self, "seed_viability_slack", 1.15))) / max(1e-6, square_m)
        r_eff = float(getattr(self, "seed_randomness", 0.0))
        line_enabled_mask = None

        if bool(getattr(self, "seed_line_min", False)):
            # ===== MODO LINHA MÍNIMA (warm-start construtivo) =====
            # Semeia a MENOR linha reta A->B que chega PRÓXIMO de atender os KPIs
            # (derivada analiticamente em _analytic_min_line_hops). Drones fora da
            # linha ficam DESATIVADOS e espalhados; a política aprende a ATIVÁ-los
            # e criar rotas (redundantes ou não) para fechar os requisitos.
            n_hops_line = int(self._analytic_min_line_hops())
            k = max(2, min(n - 1, n_hops_line))  # k hops -> k-1 intermediários
            pos = np.zeros((n, 2), dtype=np.float64)
            # drones fora da linha: em STANDBY na coluna do primeiro drone
            # (x=0, a "base" de origem), em posições aleatórias ao longo dela.
            # A posição de standby é cosmética: ao ser ATIVADO, o drone é
            # reposicionado perto da linha média (ver _activate_drone).
            for i in range(n):
                pos[i, 0] = 0.0
                pos[i, 1] = float(self._rng.randint(0, gmax + 1)) if gmax > 0 else 0.0
            y_mid = gmax / 2.0
            line_nodes = [0] + list(range(1, k)) + [n - 1]
            for j, nd in enumerate(line_nodes):
                pos[nd, 0] = float(gmax) * (j / float(k))
                pos[nd, 1] = y_mid
            path_node_lists = [line_nodes]
            line_enabled_mask = np.zeros((n,), dtype=np.bool_)
            for nd in line_nodes:
                line_enabled_mask[int(nd)] = True
        else:
            pos, path_node_lists = _generate(r_eff)
            # Loop de viabilidade: protege contra saltos absurdos gerados pelo jitter
            # aleatório, sem sobrescrever a dificuldade controlada por seed_target_hop_m.
            for _ in range(6):
                if _max_hop_cells(pos, path_node_lists) <= viab_hop_cells or r_eff <= 0.0:
                    break
                r_eff *= 0.5
                pos, path_node_lists = _generate(r_eff)

        self.drone_positions = torch.tensor(pos, dtype=torch.float32, device=DEVICE)
        # Modo linha: só os drones da linha nascem ATIVOS; os demais desativados
        # (a política os ativa quando precisar). Demais modos: todos ativos.
        if line_enabled_mask is not None:
            self.drone_enabled = line_enabled_mask
        else:
            self.drone_enabled = np.ones((n,), dtype=np.bool_)

        # Semeia os links dos K caminhos paralelos.
        adj_np = np.zeros((n, n), dtype=np.float32)
        for path in path_node_lists:
            for u, v in zip(path[:-1], path[1:]):
                u = int(u); v = int(v)
                if u != v and 0 <= u < n and 0 <= v < n:
                    if float(pos[v, 0]) >= float(pos[u, 0]):
                        adj_np[u, v] = 1.0
                    else:
                        adj_np[v, u] = 1.0

        self.adjacency_matrix = torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)

        self.last_seeded_path_nodes = tuple(tuple(int(x) for x in p) for p in path_node_lists)
        self.last_seeded_num_paths = int(len(path_node_lists))
        self.last_seeded_path_hops = int(hops_per_path)

        self._bonus_sucesso_dado = False
        self.last_removed_links = 0
        self.last_deactivated_drone = -1
        self.last_qos_target_met = 0.0
        self._dist_matrix = None
        self._dist_pos_sig = None
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None

        if hasattr(self, "_mask_version"):
            self._mask_version += 1
            self._mask_cache = None
            self._mask_cache_version = -1
        self._sync_active_mask()

    # =========================
    # Helpers
    # =========================
    def _calc_distance_matrix(self):
        # cache: evita recomputar O(N²) sem mudanças de posição
        n = int(self.num_drones)
        pos = np.ascontiguousarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float32)
        # assinatura robusta para evitar colisões (soma de coords pode colidir)
        sig = (int(zlib.crc32(pos.tobytes())), int(pos.shape[0]))
        if self._dist_pos_sig == sig and self._dist_matrix is not None:
            return self._dist_matrix

        # versão vetorizada (mesma semântica, muito mais rápida que loop Python)
        px = (pos[:, 0] * float(self.square_size_m)).astype(np.float32, copy=False)
        py = (pos[:, 1] * float(self.square_size_m)).astype(np.float32, copy=False)
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        d = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)

        self._dist_matrix = d
        self._dist_pos_sig = sig
        return d

    def _ensure_dist_matrix(self):
        return self._calc_distance_matrix()

    def _prob_link_pair(self, i: int, j: int, d_m: float) -> float:
        """Probabilidade de sucesso de quadro para o ENLACE (i,j), aplicando o
        SOMBREAMENTO ESTOCÁSTICO específico do par (Opção B).

        O sombreamento s (dB), sorteado no reset e fixo no episódio, desloca o
        SNR. Como o SNR cai com 10*n*log10(d), um sombreamento de s dB equivale,
        no modelo log-distance, a escalar a distância por 10^(s/(10*n)). Assim
        reaproveitamos exatamente a física do Abreu (_prob_link) passando uma
        DISTÂNCIA EFETIVA, sem reescrever o solver. s>0 => enlace pior (distância
        efetiva maior); s<0 => melhor. Cacheia por par (i,j), não por distância,
        pois agora a probabilidade não é função apenas de d."""
        sd = getattr(self, "_shadow_db", None)
        if sd is None:
            return self._prob_link(d_m)
        try:
            s = float(sd[int(i), int(j)])
        except Exception:
            s = 0.0
        if s == 0.0:
            return self._prob_link(d_m)
        n = float(getattr(self._link_model, "path_loss_exp", 2.6)) if getattr(self, "_link_model", None) is not None else 2.6
        n = max(1e-3, n)
        # distância efetiva que reproduz o deslocamento de SNR do sombreamento
        d_eff = float(max(1e-6, d_m) * (10.0 ** (s / (10.0 * n))))
        return self._prob_link(d_eff)

    def _prob_link(self, d_m: float) -> float:
        """Probabilidade de sucesso de quadro por distância.

        OPÇÃO (B): usa a MESMA física do solver de Abreu — Friis/log-distance
        + outage Rician (`_link_success_prob_from_distance` de abreu_modified) —
        para que a recompensa por step e a avaliação terminal compartilhem
        exatamente o mesmo modelo de enlace. Antes havia uma curva logística
        substituta, escalada por carrier_sense_range_m, que subestimava o
        alcance real do rádio em ~8x e desalinhava treino x teste.

        O carrier_sense_range_m NÃO entra mais aqui: ele é, por definição, o
        alcance de escuta de portadora (contenção/colisão no Abreu), não o
        alcance de comunicação. A distância de comunicação confiável passa a
        ser governada pelos parâmetros físicos do rádio (LinkModelParams:
        potência, frequência, expoente de perda, ruído, limiar de SNR).

        Fallback: se a função física não estiver disponível (import falhou),
        cai para a curva logística antiga apenas para não quebrar o ambiente.
        """
        d = float(max(1e-6, d_m))
        pc = self._prob_cache.get(d)
        if pc is None:
            if _link_success_prob_from_distance is not None:
                try:
                    lm = getattr(self, "_link_model", None)
                    p_succ = float(_link_success_prob_from_distance(d, link_model=lm))
                except Exception:
                    # fallback físico-aproximado em caso de erro pontual
                    p_succ = 1.0 / (1.0 + math.exp(10.0 * (d / max(1.0, float(self.carrier_sense_range_m)) - 0.65)))
            else:
                # Fallback legado (sem abreu_modified disponível).
                x = d / max(1.0, float(self.carrier_sense_range_m))
                p_succ = 1.0 / (1.0 + math.exp(10.0 * (x - 0.65)))
            pc = float(max(1e-6, min(1.0 - 1e-6, p_succ)))
            self._prob_cache[d] = pc
            if len(self._prob_cache) > 200000:
                self._prob_cache.clear()
        return float(pc)


    def _enumerate_primary_paths(self, adj_np: np.ndarray, src: int, dst: int):
        if enumerate_simple_paths is None:
            return []
        adj_bin = (np.asarray(adj_np, dtype=np.float32, order="C") > 0.0).astype(np.uint8, copy=False)
        cap = int(self.abreu_max_paths)
        # SOBREAMOSTRAGEM + TRUNCAMENTO POR QUALIDADE: a enumeração DFS corta em
        # ordem lexicográfica de vizinhos, não pelos MELHORES caminhos — dois
        # estados igualmente bons podiam medir diferente só pela ordem de
        # visita. Enumeramos até 4x o cap (topológico, cacheável) e entregamos
        # ao solver os `cap` caminhos de maior prob. estimada (produto de
        # _prob_link_pair por hop, sensível a distância + sombra do episódio).
        # A seleção fica FORA do cache porque depende de posições/sombra.
        oversample = int(min(64, max(cap, 4 * cap)))
        cache_key = (
            self._exact_array_signature(adj_bin),
            int(src),
            int(dst),
            int(oversample),
            -1 if self.abreu_max_hops is None else int(self.abreu_max_hops),
        )
        cached = self._paths_cache.get(cache_key)
        if cached is None:
            try:
                paths = enumerate_simple_paths(
                    adj_bin.astype(np.float32, copy=False), int(src), int(dst),
                    max_paths=oversample,
                    max_hops=self.abreu_max_hops,
                )
            except Exception:
                paths = []
            paths = tuple(tuple(int(x) for x in p) for p in paths)
            cached = self._cache_put_bounded(self._paths_cache, cache_key, paths, self._paths_cache_limit)
        if len(cached) <= cap:
            return cached
        try:
            dist_m = self._ensure_dist_matrix()

            def _log_quality(p):
                s = 0.0
                for a, b in zip(p[:-1], p[1:]):
                    s += float(np.log(max(1e-9, self._prob_link_pair(int(a), int(b), float(dist_m[int(a), int(b)])))))
                return s

            best = sorted(cached, key=_log_quality, reverse=True)[:cap]
            return tuple(best)
        except Exception:
            return tuple(cached[:cap])

    def _estimate_link_ber_path(self, path_nodes, dist_m: np.ndarray, dist_sig=None):
        path_key = tuple(int(x) for x in path_nodes)
        cache_key = ((dist_sig if dist_sig is not None else self._exact_array_signature(dist_m)), path_key)
        cached = self._ber_path_cache.get(cache_key)
        if cached is not None:
            return cached
        bers = []
        frame_bits = int(max(64, int(self.frame_size_bytes) * 8))
        for i in range(len(path_key) - 1):
            u = int(path_key[i])
            v = int(path_key[i + 1])
            d = float(dist_m[u, v])
            p_succ = float(max(1e-6, min(1.0 - 1e-6, self._prob_link_pair(u, v, d))))
            if _link_ber_from_success_prob is not None:
                ber = float(_link_ber_from_success_prob(p_succ, frame_bits))
            else:
                fer = 1.0 - p_succ
                ber = float(max(1e-12, min(1e-2, 1.0 - (1.0 - fer) ** (1.0 / max(1, frame_bits)))))
            bers.append(ber)
        bers = tuple(bers)
        return self._cache_put_bounded(self._ber_path_cache, cache_key, bers, self._ber_path_cache_limit)

    def _estimate_link_phy_path(self, path_nodes, dist_m: np.ndarray, dist_sig=None):
        path_key = tuple(int(x) for x in path_nodes)
        cache_key = ((dist_sig if dist_sig is not None else self._exact_array_signature(dist_m)), path_key, str(getattr(self, "phy_rate_mode", "global")).lower())
        cached = self._phy_path_cache.get(cache_key)
        if cached is not None:
            return cached
        if str(getattr(self, "phy_rate_mode", "global")).lower() == "global":
            rates = tuple(float(self.phy_rate_mbps) for _ in range(len(path_key) - 1))
            return self._cache_put_bounded(self._phy_path_cache, cache_key, rates, self._phy_path_cache_limit)
        ladder = [54.0, 48.0, 36.0, 24.0, 18.0, 12.0, 9.0, 6.0]
        rates = []
        for i in range(len(path_key) - 1):
            u = int(path_key[i]); v = int(path_key[i + 1])
            d = float(dist_m[u, v])
            x = d / max(1.0, float(self.carrier_sense_range_m))
            if x <= 0.30:
                r = ladder[0]
            elif x <= 0.40:
                r = ladder[1]
            elif x <= 0.50:
                r = ladder[2]
            elif x <= 0.60:
                r = ladder[3]
            elif x <= 0.70:
                r = ladder[4]
            elif x <= 0.80:
                r = ladder[5]
            elif x <= 0.90:
                r = ladder[6]
            else:
                r = ladder[7]
            rates.append(float(r))
        rates = tuple(rates)
        return self._cache_put_bounded(self._phy_path_cache, cache_key, rates, self._phy_path_cache_limit)

    def _evaluate_abreu_modified(self, adj_np: np.ndarray, pos_np_cells: np.ndarray,
                                 compute_max_feasible: bool = True):
        # compute_max_feasible=False desliga a busca binária da capacidade máxima
        # (~18 iterações do solver), que NÃO é necessária na recompensa por step
        # (lá só interessa o goodput na taxa oferecida). Isso acelera muito o caso
        # frequente. O terminal/diagnóstico mantém True (capacidade é útil lá).
        if not self.abreu_enabled or solve_modified_abreu_multiflow is None or FlowRequest is None:
            return None
        adj_bin = (np.asarray(adj_np, dtype=np.float32, order="C") > 0.0).astype(np.uint8, copy=False)
        pos_arr = np.asarray(pos_np_cells[:, :2], dtype=np.float32, order="C")
        cache_key = (self._exact_array_signature(adj_bin), self._exact_array_signature(pos_arr))
        cached = self._abreu_eval_cache.get(cache_key)
        if cached is not None:
            return cached

        n = int(adj_bin.shape[0])
        if n < 2:
            return None
        pos_m = pos_arr * float(self.square_size_m)
        dist_m = self._ensure_dist_matrix()
        dist_sig = self._exact_array_signature(dist_m)

        def make_req(src: int, dst: int, offered_rate: float, label: str, redundancy_mode: str = 'all_paths', max_active_paths: int | None = None):
            paths = self._enumerate_primary_paths(adj_bin, int(src), int(dst))
            if not paths:
                return None
            ber_per_path = [self._estimate_link_ber_path(p, dist_m, dist_sig=dist_sig) for p in paths]
            phy_rates_per_path = [self._estimate_link_phy_path(p, dist_m, dist_sig=dist_sig) for p in paths]
            return FlowRequest(
                paths=paths,
                offered_rate_mbps=float(offered_rate),
                redundancy_mode=str(redundancy_mode),
                max_active_paths=max_active_paths,
                ber_per_path=ber_per_path,
                phy_rates_per_path=phy_rates_per_path,
                label=label,
            )

        flow_reqs = []
        if self.abreu_exogenous_flows:
            for i, spec in enumerate(self.abreu_exogenous_flows):
                src = int(spec.get('src', 0))
                dst = int(spec.get('dst', n - 1))
                offered = float(spec.get('offered_rate_mbps', self.traffic_rate_mbps))
                mode = str(spec.get('redundancy_mode', 'all_paths'))
                max_active = spec.get('max_active_paths', None)
                req = make_req(src, dst, offered, str(spec.get('label', f'flow_{i}')), mode, max_active)
                if req is not None:
                    flow_reqs.append(req)
        else:
            req = make_req(0, n - 1, self.traffic_rate_mbps, 'primary_flow', 'all_paths', None)
            if req is not None:
                flow_reqs.append(req)

        if not flow_reqs:
            return None

        mm = solve_modified_abreu_multiflow(
            flows=flow_reqs,
            pos_m=pos_m,
            frame_size_bytes=int(self.frame_size_bytes),
            buffer_size_packets=int(self.buffer_size_packets),
            phy_rate_mbps=float(self.phy_rate_mbps),
            carrier_sense_range_m=float(self.carrier_sense_range_m),
            slot_time_us=float(self.slot_time_us),
            difs_us=float(self.difs_us),
            sifs_us=float(self.sifs_us),
            ack_us=float(self.ack_us),
            ack_timeout_us=float(self.ack_timeout_us),
            cw_min=int(self.cw_min),
            cw_max=int(self.cw_max),
            max_retries=int(self.max_retries),
            reliability_min=float(self.reliability_target),
            delay_max_ms=float(self.max_delay_ms),
            compute_max_feasible_rate=bool(compute_max_feasible),
        )
        if mm is None:
            return None

        metric_name = str(getattr(self, 'abreu_multiflow_metric', 'min_flow_reliability')).lower()
        if metric_name == 'weighted_flow_reliability':
            rel = float(mm.weighted_flow_reliability)
        elif metric_name == 'mean_flow_reliability':
            rel = float(mm.mean_flow_reliability)
        elif metric_name == 'probability_all_flows_succeed':
            rel = float(mm.probability_all_flows_succeed)
        else:
            rel = float(mm.min_flow_reliability)

        all_path_metrics = []
        for fm in mm.flows:
            for fpm in fm.path_metrics:
                all_path_metrics.append(fpm.path_metrics)
        mm.reliability = rel
        mm.loss_rate = float(max(0.0, min(1.0, 1.0 - rel)))
        # LATÊNCIA: agregação do modelo de Abreu (padrão) vs legado.
        #
        # O legado (`latency_from_longest_route=True`) reporta o atraso do
        # caminho com MAIS SALTOS. Isso vem dos modelos antigos, de propagação
        # em espaço livre, quando o atraso era essencialmente proporcional ao
        # nº de saltos e "mais saltos" ≡ "mais lento". Com o modelo de Abreu
        # isso deixou de valer: o atraso é Σ Dq_ms + Σ service_us, dominado por
        # FILA e contenção MAC, então um caminho curto congestionado pode ser
        # muito mais lento que um longo e descarregado.
        #
        # Pior: o solver agrega multi-caminho como MÍNIMO entre os caminhos
        # ativos (redundancy_mode='all_paths' — a primeira cópia que chega
        # entrega), enquanto `reliability` e `goodput` logo acima JÁ usam essa
        # agregação. Só a latência usava outra semântica, tornando os três
        # critérios de qos_target_met inconsistentes entre si.
        #
        # Impacto medido (2 caminhos de 12 e 17 hops, 200 sorteios de sombra):
        #   worst_delay_ms  p50=112ms p90=200ms  -> passa 300ms em 93%
        #   caminho+longo   p50=150ms p90=730ms  -> passa 300ms em 82%
        #
        # latency_max_hops continua sendo extraído para as métricas de saltos.
        latency_max_ms, latency_max_hops = self._extract_longest_redundant_route_latency(mm)
        if bool(getattr(self, "latency_from_longest_route", False)):
            mm.delay_ms = float(latency_max_ms)
        else:
            mm.delay_ms = float(getattr(mm, "worst_delay_ms", latency_max_ms))
        mm.latency_longest_route_ms = float(latency_max_ms)
        mm.latency_max_hops = int(latency_max_hops)
        mm.goodput_mbps = float(mm.total_goodput_mbps)
        mm.max_feasible_rate_mbps = float(sum(max(0.0, fm.max_feasible_rate_mbps) for fm in mm.flows))
        mm.interference_load = float(np.max(mm.omega) if getattr(mm, 'omega', None) is not None and mm.omega.size else 0.0)
        mm.rho_max = float(mm.max_rho)
        mm.paths = all_path_metrics
        return self._cache_put_bounded(self._abreu_eval_cache, cache_key, mm, self._abreu_eval_cache_limit)

    def _has_path(self, src, dst, adj, out_neighbors=None):
        """BFS para verificar existência de caminho src->dst.

        out_neighbors (opcional): lista de vizinhos de saída por nó.
        Se fornecida, evita varrer range(n) (mais rápido) sem mudar a lógica.
        """
        n = int(adj.shape[0])
        if src < 0 or dst < 0 or src >= n or dst >= n:
            return False
        if src == dst:
            return True

        visited = np.zeros(n, dtype=bool)
        queue = deque([int(src)])
        visited[int(src)] = True
        if out_neighbors is None:
            while queue:
                u = queue.popleft()
                if u == dst:
                    return True
                for v in range(n):
                    if adj[u, v] > 0 and not visited[v]:
                        visited[v] = True
                        queue.append(v)
            return False

        # versão usando lista de adjacência
        while queue:
            u = queue.popleft()
            if u == dst:
                return True
            for v in out_neighbors[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return False


    def _get_any_path_nodes(self, src: int, dst: int, adj_np, out_neighbors=None):
        """Retorna uma rota (lista de nós) src->dst via BFS em arestas dirigidas.
        Se não houver caminho, retorna lista vazia.

        out_neighbors (opcional): lista de vizinhos de saída por nó.
        """
        n = int(adj_np.shape[0])
        if src < 0 or dst < 0 or src >= n or dst >= n:
            return []
        if src == dst:
            return [int(src)]

        parent = [-1] * n
        q = deque([int(src)])
        parent[int(src)] = int(src)

        if out_neighbors is None:
            while q:
                u = q.popleft()
                if u == dst:
                    break
                for v in range(n):
                    if adj_np[u, v] > 0.0 and parent[v] == -1:
                        parent[v] = u
                        q.append(v)
        else:
            while q:
                u = q.popleft()
                if u == dst:
                    break
                for v in out_neighbors[u]:
                    if parent[v] == -1:
                        parent[v] = u
                        q.append(v)

        if parent[int(dst)] == -1:
            return []
        path = [int(dst)]
        cur = int(dst)
        while cur != int(src):
            cur = parent[cur]
            if cur == -1:
                return []
            path.append(cur)
        path.reverse()
        return path



    def _linha_passa_por_zona(self, p1, p2, link=None):
        """Verifica se o segmento (p1->p2) passa por alguma célula proibida.

        OTIMIZAÇÃO DE DESEMPENHO (sem mudança de resultado): esta função era
        chamada ~18,6 milhões de vezes no profiling e dominava o tempo total
        (79%) — não pela lógica, mas pelo overhead de alocar arrays numpy
        (linspace/rint/clip) em segmentos pequenos. Aqui a amostragem é feita
        com aritmética ESCALAR pura, consultando a grade booleana diretamente,
        com saída antecipada ao primeiro acerto. A semântica de amostragem é
        idêntica à original: `steps` pontos uniformes em [0,1] e arredondamento
        round-half-to-even (banker's rounding, igual a np.rint).

        Mantém o fallback vetorizado original caso a grade não exista ou ocorra
        qualquer condição inesperada.
        """
        if not self.zonas_proibidas:
            return False
        x1, y1 = p1
        x2, y2 = p2
        # int() trunca para baixo; max+1 reproduz exatamente o 'steps' original.
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) + 1)
        if steps <= 1:
            return False

        grid = self._forbidden_grid
        if grid is None:
            # fallback (caso raro): comportamento vetorizado original.
            t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
            xs = np.rint(x1 + t * (x2 - x1)).astype(np.int32, copy=False)
            ys = np.rint(y1 + t * (y2 - y1)).astype(np.int32, copy=False)
            xs = np.clip(xs, 0, self.grid_size - 1)
            ys = np.clip(ys, 0, self.grid_size - 1)
            for x, y in zip(xs.tolist(), ys.tolist()):
                if (int(x), int(y)) in self.zonas_proibidas:
                    return True
            return False

        # ---- Memoização por par de células ----
        # O mesmo par (p1,p2) é consultado milhões de vezes ao montar a máscara
        # de ação, sempre com células INTEIRAS (cell_j/cell_k/p_i_new). Só nesse
        # caso usamos o cache por par (chave inteira). Entradas float (raras,
        # vindas de posições contínuas) são computadas direto, sem cache, para
        # manter o resultado bit-a-bit idêntico ao numpy original.
        use_cache = (type(x1) is int and type(y1) is int
                     and type(x2) is int and type(y2) is int)
        cache = None
        key = None
        if use_cache:
            zone_sig = int(getattr(self, "_zone_sig", 0))
            cache = getattr(self, "_cross_pair_cache", None)
            if cache is None or getattr(self, "_cross_pair_cache_sig", None) != zone_sig:
                cache = {}
                self._cross_pair_cache = cache
                self._cross_pair_cache_sig = zone_sig
            key = (x1, y1, x2, y2)
            hit = cache.get(key)
            if hit is not None:
                return hit

        gmax = int(self.grid_size) - 1
        # Caminho escalar BIT-A-BIT idêntico ao numpy:
        # linspace(0,1,steps,float32) == float32(i/(steps-1)); aritmética float32.
        ax = np.float32(x2 - x1)
        ay = np.float32(y2 - y1)
        x1_32 = np.float32(x1)
        y1_32 = np.float32(y1)
        div = steps - 1
        _rint = self._rint_half_even
        prev_x = -1
        prev_y = -1
        result = False
        for i in range(steps):
            t = np.float32(i / div)
            xi = _rint(float(x1_32 + t * ax))
            yi = _rint(float(y1_32 + t * ay))
            if xi < 0:
                xi = 0
            elif xi > gmax:
                xi = gmax
            if yi < 0:
                yi = 0
            elif yi > gmax:
                yi = gmax
            if xi == prev_x and yi == prev_y:
                continue
            prev_x = xi
            prev_y = yi
            if grid[xi, yi]:
                result = True
                break
        # cache limitado para não crescer indefinidamente. Limite modesto:
        # com 128 workers, 400k entradas/worker desperdiçariam ~16 GB no total.
        # As posições são limitadas ao grid, então poucas dezenas de milhar de
        # pares já capturam quase todos os acertos.
        if use_cache and len(cache) < 50000:
            cache[key] = result
        return result

    @staticmethod
    def _rint_half_even(v: float) -> int:
        """Arredondamento round-half-to-even idêntico ao np.rint, em escalar."""
        f = math.floor(v)
        diff = v - f
        if diff < 0.5:
            return int(f)
        if diff > 0.5:
            return int(f) + 1
        # empate exato em .5 -> arredonda para o par
        fi = int(f)
        return fi if (fi % 2 == 0) else fi + 1


    # =========================
    # Gym API

    # =========================
    # Gym API
    # =========================

    def _max_edge_disjoint_paths_cap(self, cap: np.ndarray, src: int, dst: int) -> int:
        """Max-flow (capacidade unitária) para contar caminhos edge-disjoint.

        Substitui Edmonds-Karp por Dinic (mesma resposta lógica, bem mais rápido em grafos mais densos).
        `cap` NÃO precisa ser preservado; por compatibilidade, tratamos como entrada somente.
        """
        n = int(cap.shape[0])
        src = int(src)
        dst = int(dst)
        if n <= 1 or src == dst:
            return 0

        # Dinic (grafos pequenos: implementação Python simples)
        class _Edge:
            __slots__ = ("to", "rev", "cap")
            def __init__(self, to: int, rev: int, cap: int):
                self.to = to
                self.rev = rev
                self.cap = cap

        g = [[] for _ in range(n)]

        def add_edge(fr: int, to: int, c: int):
            g[fr].append(_Edge(to, len(g[to]), c))
            g[to].append(_Edge(fr, len(g[fr]) - 1, 0))

        # monta grafo a partir da matriz de capacidades
        ei, ej = np.nonzero(cap > 0)
        for u, v in zip(ei.tolist(), ej.tolist()):
            add_edge(int(u), int(v), int(cap[int(u), int(v)]))

        flow = 0
        INF = 10 ** 9

        while True:
            level = [-1] * n
            q = deque([src])
            level[src] = 0
            while q:
                v = q.popleft()
                for e in g[v]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[v] + 1
                        q.append(e.to)
            if level[dst] < 0:
                break

            it = [0] * n

            def dfs(v: int, f: int) -> int:
                if v == dst:
                    return f
                i = it[v]
                while i < len(g[v]):
                    e = g[v][i]
                    if e.cap > 0 and level[e.to] == level[v] + 1:
                        ret = dfs(e.to, f if f < e.cap else e.cap)
                        if ret:
                            e.cap -= ret
                            g[e.to][e.rev].cap += ret
                            return ret
                    i += 1
                    it[v] = i
                return 0

            while True:
                pushed = dfs(src, INF)
                if not pushed:
                    break
                flow += pushed

        return int(flow)



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Zera a instrumentação do custo absoluto de zona. Sem isto os
        # contadores acumulariam entre episódios e as médias por passo seriam
        # inúteis (o denominador cresceria sem limite).
        self._ep_zone_abs_pen = 0.0
        self._ep_zone_viol_steps = 0
        self._ep_zone_drones_sum = 0.0
        self._ep_zone_links_sum = 0.0
        self._ep_steps_counted = 0

        if seed is not None:
            # Seed o RNG interno do ambiente
            self._rng.seed(int(seed))

        if self.num_drones > self.max_drones:
            self.num_drones = self.max_drones

        # OPÇÃO B — SOMBREAMENTO ESTOCÁSTICO por enlace (fiel à literatura A2A).
        # Sorteia um desvio de sombreamento ~ N(0, sigma) em dB para CADA par de
        # drones (i,j), simétrico e FIXO durante o episódio (o sombreamento de
        # larga escala é estável para uma dada geometria). Reprodutível (usa
        # self._rng já semeado). A variabilidade é real — alguns enlaces ficam
        # melhores, outros piores, tornando a confiabilidade não-determinística.
        #
        # CENÁRIO: comunicação ar-ar (A2A) entre drones a 30-50 m em área
        # SUBURBANA. O expoente de path loss é 2,6 (medido por Goddemeier &
        # Wietfeld 2015 para A2A a 2,4 GHz, <50 m) e o K de Rician ~10 dB (a 40 m,
        # mesmo paper) — cf. LinkModelParams em abreu_modified.py. Como a 30-50 m
        # em suburbano há uma fração de enlaces obstruídos (NLoS) ocasionais, o
        # SOMBREAMENTO é elevado de 3,5 para ~5,5 dB (solução intermediária):
        # embute a variabilidade extra dos enlaces ocasionalmente obstruídos na
        # componente estocástica, sem modelar explicitamente a probabilidade de
        # LoS. Valor conservador e defensável para baixa altitude suburbana.
        try:
            sigma = float(getattr(self, "link_shadow_sigma_db", 5.5))
            nmax = int(self.max_drones)
            if sigma > 0.0:
                raw = self._rng.normal(0.0, sigma, size=(nmax, nmax)).astype(np.float32)
                # simetriza: sombra(i,j) = sombra(j,i)
                self._shadow_db = np.triu(raw, 1)
                self._shadow_db = self._shadow_db + self._shadow_db.T
            else:
                self._shadow_db = np.zeros((nmax, nmax), dtype=np.float32)
        except Exception:
            self._shadow_db = None
        # caches de probabilidade/BER dependem do sombreamento -> limpa por episódio
        try:
            self._prob_cache.clear()
            self._ber_path_cache.clear()
        except Exception:
            pass

        self._reset_zonas()

        # Estado inicial sempre VIÁVEL e fixo (Alternativa 1, warm start).
        # O currículo de "dirty reset" (degradar o estado inicial) foi removido:
        # agora todos episodio comeca de uma topologia que ja atende os KPIs, e o
        # agente aprende a OTIMIZAR (minimizar drones) e a manter/aprimorar os
        # KPIs. A aleatorização progressiva para generalização, se desejada,
        # deve ser reintroduzida depois de forma controlada.
        self._reset_count += 1
        self._dirty_episode = False
        self._force_drone_in_zone = False

        # Recalcula o estado inicial viável
        self._init_state()
        # SEMEADURA DE VIOLAÇÃO: com prob. p_zone_violation_seed, reposiciona uma
        # zona SOBRE a linha inicial — criando drones dentro dela e links que a
        # cruzam. Cenário de EVACUAÇÃO ("no-fly zone declarada sobre a rede já
        # implantada: saia e reconfigure"). Sem isso o estado inicial nunca viola
        # e a política jamais treina a evacuação — apenas o contorno.
        self._seed_zone_violation()
        self._sync_active_mask()
        # reset tracking de zona
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._zone_clean_once = False
        self._episode_steps = 0
        # Reset de sucesso (evita "carregar" streak/bonus entre episódios)
        self._success_streak = 0
        self._bonus_sucesso_dado = False
        # Zera o potencial anterior: o primeiro passo do episódio usa Phi do estado
        # semeado como baseline (não vaza potencial entre episódios). Sem isso, a
        # diferencial do 1º passo compararia com o Phi do episódio anterior.
        self._primary_potential_prev = None
        # Reset do detector de convergência (não vaza entre episódios).
        self._prev_step_reward = None
        self._stable_steps = 0
        self._prev_state_sig = None
        # POTENCIAL INICIAL = Phi(s0): computa o potencial do estado SEMEADO e o
        # usa como baseline, para a PRIMEIRA ação do episódio já ser avaliada
        # (gamma*Phi(s1)-Phi(s0)). Sem isso (baseline None -> r=0 no 1º passo), a
        # primeira ação destrutiva passava IMPUNE. A chamada também computa as
        # métricas do warm-start, das quais capturamos a confiabilidade INICIAL
        # (p/ o delta de reparo terminal-inicial no TensorBoard).
        self._primary_potential_prev = None
        self._initial_rel_block = 0.0
        if bool(getattr(self, "use_hybrid_reward", True)):
            try:
                self._calculate_fit()  # seta _primary_potential_prev = phi(s0)
            except Exception:
                self._primary_potential_prev = None
            # Neutraliza QUALQUER efeito colateral da chamada de baseline.
            self._success_streak = 0
            self._bonus_sucesso_dado = False
            self._prev_step_reward = None
            self._stable_steps = 0
            self._episode_steps = 0
            self._initial_rel_block = float(getattr(self, "current_reliability_block", 0.0))
        obs = self._get_obs()
        if _GYMNASIUM:
            return obs, {"reset": True}
        return obs

    def step(self, action):
        """Executa 1 passo do ambiente.

        Ajuste (mask): aplica toggle_op (ADD/REMOVE) ANTES do movimento.
        Assim, o action-mask calculado no estado pré-ação continua coerente
        com a validação do link, evitando muitos `invalid_link_action` artificiais
        quando a ação inclui mover + togglar link no mesmo step.

        action: [move_idx, step_code, dir_code, toggle_op]
        """

        # ---------- decodifica ação ----------
        move_idx = int(action[0]) if len(action) > 0 else 0

        step_code = int(action[1]) if len(action) > 1 else 0
        step_code = max(0, min(STEP_CELLS_N - 1, step_code))
        step_cells = int(STEP_CELLS_CHOICES[step_code])

        dir_code = int(action[2]) if len(action) > 2 else 0
        dx, dy = 0, 0
        if dir_code == 1:
            dx = +1
        elif dir_code == 2:
            dx = -1
        elif dir_code == 3:
            dy = +1
        elif dir_code == 4:
            dy = -1

        # toggle_op (maskable)
        toggle_op = int(action[3]) if len(action) > 3 else 0

        # Compatibilidade: se alguém passar a ação antiga 6D, converte para toggle_op.
        # action antiga: [move_idx, step_code, dir_code, toggle_flag, src, dst]
        if len(action) >= 6:
            old_toggle_flag = int(action[3])
            old_src = int(action[4])
            old_dst = int(action[5])
            if old_toggle_flag == 1 and old_src != old_dst:
                edge_idx = self._edge_pair_to_idx.get((old_src, old_dst), None)
                if edge_idx is not None:
                    # no formato antigo era "toggle" (se existe remove, senão adiciona)
                    if (old_src < self.num_drones and old_dst < self.num_drones and
                            self.adjacency_matrix[old_src, old_dst].item() > 0.5):
                        toggle_op = 1 + self._E + edge_idx  # REMOVE
                    else:
                        toggle_op = 1 + edge_idx  # ADD

        # ---------- 1) aplica toggle ANTES do movimento ----------
        invalid_link_action = False
        link_changed = False
        activation_applied = False
        activated_idx = -1
        deactivation_applied = False
        deactivated_idx = -1

        activate_base = int(getattr(self, "_activate_op_base", 1 + 2 * self._E))
        deactivate_base = int(getattr(self, "_deactivate_op_base", 1 + 2 * self._E + self.max_drones))
        acting_idx = int(move_idx) if len(action) > 0 else 0
        acting_is_anchor = bool(0 <= acting_idx < self.num_drones and (acting_idx == 0 or acting_idx == self.num_drones - 1))

        if toggle_op != 0:
            enabled = self._enabled_mask_np()
            if activate_base <= int(toggle_op) < (activate_base + self.max_drones):
                act_idx = int(toggle_op - activate_base)
                try:
                    _act_cap = int(self._analytic_min_qos_drones()) + int(getattr(self, "activation_slack_over_min", 6))
                except Exception:
                    _act_cap = int(self.num_drones)
                over_cap = int(np.count_nonzero(enabled)) >= _act_cap
                if (not acting_is_anchor) or over_cap or act_idx <= 0 or act_idx >= self.num_drones - 1 or act_idx >= enabled.size or bool(enabled[act_idx]):
                    invalid_link_action = True
                else:
                    if self._activate_drone(act_idx):
                        activation_applied = True
                        activated_idx = int(act_idx)
                        link_changed = True
                    else:
                        invalid_link_action = True
            elif deactivate_base <= int(toggle_op) < (deactivate_base + self.max_drones):
                deact_idx = int(toggle_op - deactivate_base)
                min_keep = 2
                # Controle ambicioso: somente as âncoras podem ativar/desativar
                # intermediários, mas a desativação é liberada com QoS parcial
                # mínimo, antes do QoS completo, para permitir poda gradual.
                prune_gate_ok = bool(float(getattr(self, "last_qos_target_met", 0.0)) >= 1.0 or self._current_prune_gate())
                if (not acting_is_anchor) or (not prune_gate_ok) or deact_idx <= 0 or deact_idx >= self.num_drones - 1 or deact_idx >= enabled.size or (not bool(enabled[deact_idx])) or int(np.count_nonzero(enabled)) <= min_keep:
                    invalid_link_action = True
                else:
                    if self._deactivate_drone(deact_idx):
                        deactivation_applied = True
                        deactivated_idx = int(deact_idx)
                        link_changed = True
                    else:
                        invalid_link_action = True
            else:
                op = int(toggle_op) - 1
                is_remove = (op >= self._E)
                edge_idx = (op - self._E) if is_remove else op
                if 0 <= edge_idx < self._E:
                    src, dst = self._edge_pairs[edge_idx]
                    enabled = self._enabled_mask_np()
                    if src >= self.num_drones or dst >= self.num_drones:
                        invalid_link_action = True
                    elif src == dst:
                        invalid_link_action = True
                    elif src >= enabled.size or dst >= enabled.size or (not bool(enabled[src])) or (not bool(enabled[dst])):
                        invalid_link_action = True
                    else:
                        is_on = (self.adjacency_matrix[src, dst].item() > 0.5)
                        if is_remove:
                            if is_on:
                                # TRAVA: não remove se esta for a última aresta
                                # que mantém o caminho 0 -> N-1 (aresta-ponte).
                                if self._remove_disconnects(src, dst):
                                    invalid_link_action = True
                                else:
                                    self.adjacency_matrix[src, dst] = 0.0
                                    link_changed = True
                            else:
                                invalid_link_action = True
                        else:
                            if is_on:
                                invalid_link_action = True
                            else:
                                out_deg_src = int(torch.sum(self.adjacency_matrix[src] > 0.5).item())
                                if out_deg_src >= self.max_out_connections:
                                    invalid_link_action = True

                                if (not invalid_link_action) and self.hard_mask_zone_edges and self.zonas_proibidas:
                                    p1 = (int(round(float(self.drone_positions[src, 0].item()))), int(round(float(self.drone_positions[src, 1].item()))))
                                    p2 = (int(round(float(self.drone_positions[dst, 0].item()))), int(round(float(self.drone_positions[dst, 1].item()))))
                                    if self._linha_passa_por_zona(p1, p2, (src, dst)):
                                        invalid_link_action = True

                                if (not invalid_link_action) and self.hard_mask_left_links:
                                    if float(self.drone_positions[dst, 0].item()) < float(self.drone_positions[src, 0].item()):
                                        invalid_link_action = True

                                if (not invalid_link_action) and self.hard_mask_cycles:
                                    adj_on = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:self.num_drones, :self.num_drones]) > 0.5)
                                    reach = self._reachability_bits(adj_on)
                                    if ((int(reach[int(dst)]) >> int(src)) & 1) != 0:
                                        invalid_link_action = True

                                if not invalid_link_action:
                                    self.adjacency_matrix[src, dst] = 1.0
                                    link_changed = True

        if link_changed:
            self._mask_version += 1

        # ---------- 2) movimento (1 drone por step) ----------
        old_pos = self.drone_positions.clone()
        invalid_move_count = 0

        # HARD: guardamos quantos links EXISTENTES cruzam zona ANTES do movimento.
        prev_links_cross_zone = None
        if (getattr(self, "hard_block_move_cross_zone", True)
                and dir_code != 0 and step_cells > 0
                and getattr(self, "zonas_proibidas", None)):
            if self.obs_edge_cross or self.hard_mask_zone_edges:
                self._ensure_edge_cross_mat()
                if self._edge_cross_mat is not None:
                    n = int(self.num_drones)
                    if n > 1:
                        adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]).astype(np.uint8, copy=False)
                        cross_np = self._edge_cross_mat[:n, :n]
                        prev_links_cross_zone = int(((adj_np > 0) & (cross_np > 0)).sum())

        # Apenas UM drone intermediário ativo pode se mover por step.
        if dir_code != 0 and step_cells > 0:
            enabled = self._enabled_mask_np()
            if move_idx <= 0 or move_idx >= self.num_drones - 1 or move_idx >= enabled.size or (not bool(enabled[move_idx])):
                invalid_move_count += 1
            else:
                i = int(move_idx)
                x0 = int(self.drone_positions[i, 0].item())
                y0 = int(self.drone_positions[i, 1].item())
                x, y = x0, y0
                invalid_zone_move = False

                for _ in range(step_cells):
                    nx = max(0, min(self.grid_size - 1, x + dx))
                    ny = max(0, min(self.grid_size - 1, y + dy))

                    cur_in_zone = (x, y) in self.zonas_proibidas
                    nxt_in_zone = (nx, ny) in self.zonas_proibidas

                    if (not cur_in_zone) and nxt_in_zone:
                        invalid_zone_move = True
                        break

                    x, y = nx, ny

                if invalid_zone_move:
                    invalid_move_count += 1
                    x, y = x0, y0

                self.drone_positions[i, 0] = float(x)
                self.drone_positions[i, 1] = float(y)

        moved = bool(step_cells > 0 and (self.drone_positions != old_pos).any())
        if moved:
            self._dist_matrix = None
            self._dist_pos_sig = None

            if self.obs_edge_cross or self.hard_mask_zone_edges:
                self._update_edge_cross_after_move(int(move_idx))

            move_reverted = False
            if (prev_links_cross_zone is not None) and (self._edge_cross_mat is not None):
                n = int(self.num_drones)
                if n > 1:
                    adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]).astype(np.uint8, copy=False)
                    cross_np = self._edge_cross_mat[:n, :n]
                    new_links_cross_zone = int(((adj_np > 0) & (cross_np > 0)).sum())
                    if new_links_cross_zone > int(prev_links_cross_zone):
                        invalid_move_count += 1
                        mi = int(move_idx)
                        self.drone_positions[mi, 0] = old_pos[mi, 0]
                        self.drone_positions[mi, 1] = old_pos[mi, 1]
                        if self.obs_edge_cross or self.hard_mask_zone_edges:
                            self._update_edge_cross_after_move(mi)
                        move_reverted = True
                        moved = False

            if (not move_reverted) and self.hard_mask_zone_edges:
                self._mask_version += 1
                self._mask_cache = None
                self._mask_cache_version = -1

        # ---------- fit/reward ----------
        reward, info = self._calculate_fit()

        # Penaliza ações inválidas para ajudar a política a aprender mais rápido.
        # NÃO escalar por num_drones: a penalidade por ação ILEGAL deve ser
        # constante, senão frotas maiores (necessárias para redundância) ficam
        # artificialmente arriscadas — pressão anti-redundância indesejada. Esta
        # penalidade NÃO é gateada por KPI porque corrige ação ilegal (regra do
        # ambiente), não é um incentivo secundário de eficiência.
        if invalid_move_count > 0 or invalid_link_action:
            reward -= float(self.fit_w.get("INVALID_ACTION_PENALTY", 0.0))
        # DRONE_ACTIVATION_PENALTY removida: era redundante com o custo de estado
        # (DRONE_ACTIVE_COST_AFTER_QOS), que já penaliza drones excedentes de forma
        # contínua e muito mais forte (~20 em 100 passos vs 0.05 do evento, 0.2% do
        # total). Sua função anti-oscilação tornou-se obsoleta quando o bônus de
        # desativação foi removido — sem bônus para farmar, oscilar já dá prejuízo
        # pelo custo de estado. A minimização de drones fica inteiramente a cargo
        # do custo de estado, mecanismo único, contínuo e não-farmável.
        if deactivation_applied:
            # Alvos reais (não os limiares frouxos de pruning): o BÔNUS de
            # redução de drones só vale ENQUANTO confiabilidade E vazão estão
            # sendo atendidas nos alvos (objetivo secundário condicionado aos
            # primários). Avaliado a cada passo pelo estado atual (Markov).
            _tg_goodput, _tg_rel, _tg_delay = self._min_drone_targets()
            _rel_now = float(info.get("reliability_block", self.current_reliability_block))
            _tp_now = float(info.get("throughput", self.current_throughput))
            _deliv_now = float(info.get("delivery_ratio", self.current_delivery_ratio))
            _rel_ok_now = bool(_rel_now + 1e-12 >= float(_tg_rel))
            _deliv_ok_now = bool(_deliv_now + 1e-12 >= float(getattr(self, "delivery_ratio_target", 0.999)))
            _tp_ok_now = bool(_tp_now + 1e-9 >= float(_tg_goodput))
            # A PENALIDADE por desativação ruim continua atrelada a NÃO quebrar a
            # rede (limiares frouxos): só pune remoções que de fato derrubam o
            # caminho/confiabilidade básica — não pune exploração de minimização
            # enquanto a rede ainda está sendo construída.
            path_not_broken = bool(
                info.get("path_exists", False)
                and _deliv_now + 1e-12 >= float(getattr(self, "prune_delivery_min", 0.30))
                and _rel_now + 1e-12 >= float(getattr(self, "prune_rel_min", 0.20))
            )
            # BÔNUS de evento por desativação REMOVIDO. Antes a desativação com
            # KPIs OK pagava DEACTIVATION_BONUS (0.30) + PRUNE_KEEP_PATH_BONUS
            # (0.20) = 0.50 por evento. Isso era FARMÁVEL: o ciclo
            # ativar(-0.05)->desativar(+0.50) rendia +0.45 a cada 2 passos
            # (~112/episódio) sem ganho real de QoS — um reward hacking de
            # oscilação que destravaria assim que a confiabilidade cruzasse o
            # alvo. A minimização de drones agora vem do CUSTO DE ESTADO
            # (DRONE_ACTIVE_COST_AFTER_QOS, por passo por drone excedente, abaixo),
            # que premia o ESTADO de ter poucos drones — não o EVENTO de
            # desativar. Reativar apenas restaura o custo; não há o que farmar.
            # A PENALIDADE por desativação RUIM é mantida: protege contra remoções
            # que quebram a rede (não é farmável, pune dano real).
            # bonus_ok mantido apenas como MÉTRICA (prune_keep_path): indica que a
            # desativação ocorreu com todos os KPIs nos alvos. Não dá mais
            # recompensa (ver acima) — serve para monitorar poda saudável no log.
            bonus_ok = bool(info.get("path_exists", False) and _rel_ok_now and _deliv_ok_now and _tp_ok_now)
            if not path_not_broken:
                reward -= float(self.fit_w.get("BAD_DEACTIVATION_PENALTY", 0.0))
            info["bad_deactivation"] = int(0 if path_not_broken else 1)
            info["prune_keep_path"] = int(1 if bonus_ok else 0)

        # ----- critério de sucesso -----
        target_goodput, target_rel, target_delay = self._min_drone_targets()
        success_throughput = float(info.get("throughput", self.current_throughput))
        success_now = bool(
            bool(info.get("path_exists", False))
            and float(info.get("reliability_block", self.current_reliability_block)) + 1e-12 >= float(target_rel)
            and float(info.get("delivery_ratio", self.current_delivery_ratio)) + 1e-12 >= float(getattr(self, "delivery_ratio_target", 0.999))
            and success_throughput + 1e-9 >= float(self._effective_goodput_target())
            and success_throughput <= float(getattr(self, "throughput_max_mbps", THROUGHPUT_MAX_MBPS)) + 1e-9
            and float(info.get("latency_ms", self.current_latency_ms)) <= float(target_delay) + 1e-9
        )

        if success_now:
            self._success_streak += 1
        else:
            self._success_streak = 0

        success_reached = self._success_streak >= self._success_need
        if success_reached and not self._bonus_sucesso_dado:
            # SUPRESSÃO POR ZONA: o bônus de sucesso é escalado pelo fator de
            # violação — cheio se limpo, ~zerado se sujo. Torna o sucesso
            # inalcançável em recompensa enquanto violar.
            reward += float(self.fit_w.get("SUCCESS_BONUS", 0.0)) * self._zone_success_factor()
            self._bonus_sucesso_dado = True

        # ----- término de episódio -----
        self._episode_steps += 1

        # Terminação por CONVERGÊNCIA (padrão): o episódio termina quando a política
        # ESTABILIZA — |reward(t) - reward(t-1)| < eps por `converge_patience` passos
        # consecutivos — DESDE QUE o QoS já tenha sido atingido ao menos uma vez.
        # Racional: enquanto a política melhora (repara topologia, poda drones), a
        # recompensa muda a cada passo e o episódio continua; quando ela para de
        # melhorar tendo atingido o QoS, a solução convergiu e não há ganho em rodar
        # até o horizonte. Ganho: episódios ~6-13x mais curtos -> muito mais
        # episódios por compute, com o retorno preservado. O gate de QoS impede
        # que um subótimo estável
        # (ex.: pairar) termine cedo: nesse caso o episódio segue até o horizonte,
        # dando à política a chance de descobrir o reparo.
        # terminate_on_success=True restaura o corte imediato antigo (no 1º QoS).
        prev_r = getattr(self, "_prev_step_reward", None)
        eps = float(getattr(self, "converge_eps", 0.01))
        if bool(getattr(self, "converge_on_state", True)):
            # ANTI-FARMING: a convergência olha a estabilidade do ESTADO
            # (topologia + posições + habilitados), não da recompensa. Com o
            # critério antigo, "tremelicar" (mover um drone de um lado para o
            # outro) mudava a recompensa a cada passo e adiava o término
            # indefinidamente, farmando o bônus de manutenção (+6/passo) até o
            # horizonte — observado em avaliação real: episódio de 546 passos
            # com retorno 3083 vs mediana de 38 passos/retorno ~290, estourando
            # a escala do crítico (perda ~8e3). Com estabilidade de estado, o
            # jitter REINICIA a contagem em vez de sustentá-la.
            try:
                nloc = int(self.num_drones)
                pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:nloc, :2])
                adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix[:nloc, :nloc])
                sig = (
                    np.round(pos_np, 3).tobytes(),
                    (adj_np > 0.5).tobytes(),
                    self._enabled_mask_np().tobytes(),
                )
            except Exception:
                sig = None
            if sig is not None and sig == getattr(self, "_prev_state_sig", None):
                self._stable_steps = int(getattr(self, "_stable_steps", 0)) + 1
            else:
                self._stable_steps = 0
            self._prev_state_sig = sig
        else:
            # legado: estabilidade da recompensa (farmável por jitter)
            if prev_r is not None and abs(float(reward) - float(prev_r)) < eps:
                self._stable_steps = int(getattr(self, "_stable_steps", 0)) + 1
            else:
                self._stable_steps = 0
        self._prev_step_reward = float(reward)
        self._last_reward = float(reward)

        ever_qos = bool(getattr(self, "_bonus_sucesso_dado", False))
        converge_patience = int(getattr(self, "converge_patience", 0))
        converged = (
            converge_patience > 0
            and ever_qos
            and int(getattr(self, "_stable_steps", 0)) >= converge_patience
        )

        terminate_on_success = bool(getattr(self, "terminate_on_success", False))
        timeout = self._episode_steps >= self._max_episode_steps
        done = False
        if timeout or converged or (terminate_on_success and success_reached):
            done = True

        # truncation: True para término por horizonte. Convergência (QoS estável) e
        # sucesso-terminal NÃO são truncation — são "solução alcançada", então o
        # bootstrap do GAE trata o valor futuro de forma consistente (o estado é
        # terminal por mérito, não por corte de tempo).
        success_terminal = converged or (terminate_on_success and success_reached)
        truncation = done and not success_terminal
        obs = self._get_obs()
        info["reliability"] = float(self.current_reliability)
        info["loss_rate"] = float(getattr(self, "current_loss_rate", 1.0))
        info["latency_ms"] = float(getattr(self, "current_latency_ms", 0.0))
        info["max_feasible_rate_mbps"] = float(getattr(self, "current_max_feasible_rate", 0.0))
        info["num_paths"] = int(getattr(self, "current_num_paths", 0))
        info["invalid_action"] = bool(invalid_move_count > 0 or invalid_link_action)
        # Penalidade de ação inválida não é mais escalada por num_drones (agora
        # constante); métrica mantida em 1.0 para compatibilidade de logs.
        info["invalid_action_penalty_scale"] = 1.0
        info["invalid_move_count"] = int(invalid_move_count)
        active_drones_now, _used_drones_now = self._compute_active_and_used_drones()
        info["active_drones"] = int(active_drones_now)
        info["qos_target_met"] = float(getattr(self, 'last_qos_target_met', 0.0))
        info["activation_applied"] = int(1 if activation_applied else 0)
        info["activated_drone_idx"] = int(activated_idx)
        info["deactivation_applied"] = int(1 if deactivation_applied else 0)
        info["deactivated_drone_idx"] = int(deactivated_idx)
        info.setdefault("bad_deactivation", 0)
        info.setdefault("prune_keep_path", 0)
        info["success_max_latency_ms"] = float(getattr(self, 'max_delay_ms', MAX_DELAY_MS_DEFAULT))
        info["success"] = bool(success_reached)
        if success_reached:
            info["success_at_step"] = int(self._episode_steps)

        # Terminal shaping: alinha o treino com a métrica do TESTE (confiabilidade do estado FINAL do episódio).
        if done:
            info["is_success"] = bool(success_reached)
            # Recalcula confiabilidade no estado FINAL para evitar qualquer divergência (cache/stale)
            try:
                n = int(self.num_drones)
                adj_t = _torch_to_numpy_cpu_view(self.adjacency_matrix)
                pos_t = _torch_to_numpy_cpu_view(self.drone_positions)
                adj_np = (adj_t[:n, :n] > 0).astype(np.float32)
                pos_np = np.asarray(pos_t[:n, :2], dtype=np.float32)

                # Snapshot terminal: evita divergências no pós-processamento quando VecEnv faz auto-reset.
                # (O teste TesteMultiDrone.txt dá preferência a estes campos.)
                info["terminal_adj01"] = adj_np
                info["terminal_pos_cells"] = pos_np
                info["terminal_num_drones"] = int(n)
                info["terminal_square_size_m"] = float(self.square_size_m)
                path_final = bool(self._has_path(0, n - 1, adj_np)) if n > 1 else False
                info["path_exists_terminal"] = int(1 if path_final else 0)
                info["abreu_enabled_terminal"] = int(1 if bool(self.abreu_enabled) else 0)
                info["bootstrap_path_success_terminal"] = int(1 if (path_final and not bool(self.abreu_enabled)) else 0)
                if path_final and self.abreu_enabled:
                    term_metrics = self._evaluate_abreu_modified(adj_np, pos_np)
                    if term_metrics is not None:
                        self.current_reliability = float(term_metrics.reliability)
                        self.current_reliability_block = float(getattr(term_metrics, "reliability_block", term_metrics.reliability))
                        self.current_delivery_ratio = float(getattr(term_metrics, "delivery_ratio", 0.0))
                        self.current_reliability_hybrid = float(getattr(term_metrics, "reliability_hybrid", term_metrics.reliability))
                        self.current_latency_ms = float(term_metrics.delay_ms)
                        self.current_throughput = float(term_metrics.goodput_mbps)
                        self.current_loss_rate = float(term_metrics.loss_rate)
                        self.current_max_feasible_rate = float(term_metrics.max_feasible_rate_mbps)
                        # MultiFlowMetrics não tem 'paths'; conta os caminhos
                        # ATIVOS do fluxo primário (flows[0].path_metrics, onde
                        # cada ActivePathMetrics representa um caminho).
                        try:
                            _flows = getattr(term_metrics, "flows", []) or []
                            if _flows:
                                _pm = getattr(_flows[0], "path_metrics", []) or []
                                # conta caminhos efetivamente ativos quando o campo existir
                                _act = [p for p in _pm if getattr(p, "active", True)]
                                self.current_num_paths = int(len(_act) if _act else len(_pm))
                            else:
                                self.current_num_paths = 0
                        except Exception:
                            self.current_num_paths = 0
                    else:
                        self.current_reliability = 0.0
                        self.current_reliability_block = 0.0
                        self.current_delivery_ratio = 0.0
                        self.current_reliability_hybrid = 0.0
                        self.current_loss_rate = 1.0
                        self.current_latency_ms = 0.0
                        self.current_throughput = 0.0
                        self.current_max_feasible_rate = 0.0
                        self.current_num_paths = 0
                else:
                    # Sem caminho OU bootstrap com Abreu desligado: não publique confiabilidade física 100%.
                    # O sucesso de conectividade no bootstrap é registrado separadamente em
                    # bootstrap_path_success_terminal.
                    self.current_reliability = 0.0
                    self.current_reliability_block = 0.0
                    self.current_delivery_ratio = 0.0
                    self.current_reliability_hybrid = 0.0
                    self.current_loss_rate = 1.0
                    self.current_latency_ms = 0.0
                    self.current_throughput = 0.0
                    self.current_max_feasible_rate = 0.0
                    self.current_num_paths = 0
            except Exception:
                info["path_exists_terminal"] = 0
                info["abreu_enabled_terminal"] = int(1 if bool(getattr(self, "abreu_enabled", False)) else 0)
                info["bootstrap_path_success_terminal"] = 0

            info["reliability_terminal"] = float(self.current_reliability)
            info["reliability_block_terminal"] = float(getattr(self, "current_reliability_block", self.current_reliability))
            # Métricas de REPARO/CONSTRUÇÃO: rel inicial do warm-start e delta
            # (terminal - inicial). delta > 0 = política melhora a topologia;
            # delta < 0 = destrutiva. Indicador central da tese no TensorBoard.
            _rel_ini = float(getattr(self, "_initial_rel_block", 0.0))
            info["reliability_initial"] = _rel_ini
            info["repair_delta_terminal"] = float(info["reliability_block_terminal"]) - _rel_ini
            info["delivery_ratio_terminal"] = float(getattr(self, "current_delivery_ratio", 0.0))
            info["reliability_hybrid_terminal"] = float(getattr(self, "current_reliability_hybrid", self.current_reliability))
            # done_reason distingue as formas de término:
            #  - "converged": QoS atingido e recompensa estabilizou (solução ótima
            #     alcançada e sustentada) — o desfecho DESEJADO para a tese.
            #  - "timeout_qos": chegou ao horizonte mantendo o QoS (bom, mas não
            #     convergiu formalmente — ainda oscilava).
            #  - "timeout": chegou ao horizonte SEM QoS (falha de reparo).
            #  - "success": só no modo legado terminate_on_success.
            if terminate_on_success and success_reached:
                info["done_reason"] = "success"
            elif converged:
                info["done_reason"] = "converged"
            elif truncation or done:
                info["done_reason"] = "timeout_qos" if success_reached else "timeout"
            else:
                info["done_reason"] = "done"

            viol = int(info.get("drones_in_zone", 0) + info.get("links_cross_zone", 0))
            info["zone_violations_terminal"] = int(viol)

            # Terminal throughput (vazão final, em Mbps)
            try:
                info["throughput_terminal"] = float(info.get("throughput", getattr(self, "current_throughput", 0.0)))
            except Exception:
                info["throughput_terminal"] = 0.0
            info["latency_terminal_ms"] = float(getattr(self, "current_latency_ms", 0.0))
            info["loss_terminal"] = float(getattr(self, "current_loss_rate", 1.0))
            info["max_feasible_rate_terminal"] = float(getattr(self, "current_max_feasible_rate", 0.0))
            info["num_paths_terminal"] = int(getattr(self, "current_num_paths", 0))
            active_drones_terminal, used_drones_terminal = self._compute_active_and_used_drones()
            info["active_drones_terminal"] = int(active_drones_terminal)
            info["used_drones_terminal"] = int(used_drones_terminal)
            info["qos_target_met_terminal"] = float(getattr(self, 'last_qos_target_met', 0.0))
            info["activation_applied_terminal"] = int(info.get("activation_applied", 0))
            info["activated_drone_idx_terminal"] = int(info.get("activated_drone_idx", -1))
            info["deactivation_applied_terminal"] = int(info.get("deactivation_applied", 0))
            info["deactivated_drone_idx_terminal"] = int(info.get("deactivated_drone_idx", -1))
            info["bad_deactivation_terminal"] = int(info.get("bad_deactivation", 0))
            info["prune_keep_path_terminal"] = int(info.get("prune_keep_path", 0))
            info["success_max_latency_ms_terminal"] = float(getattr(self, 'max_delay_ms', MAX_DELAY_MS_DEFAULT))

            # sucesso terminal alinhado apenas aos objetivos centrais
            target_goodput, target_rel, target_delay = self._min_drone_targets()
            qos_terminal_ok = bool(
                bool(info.get("path_exists", False))
                and float(getattr(self, "current_reliability_block", self.current_reliability)) + 1e-12 >= float(target_rel)
                and float(getattr(self, "current_delivery_ratio", 0.0)) + 1e-12 >= float(getattr(self, "delivery_ratio_target", 0.999))
                and float(self.current_throughput) + 1e-9 >= float(self._effective_goodput_target())
                and float(self.current_throughput) <= float(getattr(self, "throughput_max_mbps", THROUGHPUT_MAX_MBPS)) + 1e-9
                and float(self.current_latency_ms) <= float(target_delay) + 1e-9
            )
            info["qos_target_met_terminal"] = float(1.0 if qos_terminal_ok else 0.0)
            # O sucesso estrito era gravado apenas pelo wrapper legado
            # (Drone5GMAEnv); o env HAPPO usado no treino nunca emitia a chave e
            # eval_terminal/success_* ficava cravado em 0 mesmo com QoS ~0.6.
            info["success_terminal"] = int(1 if qos_terminal_ok else 0)
            # violações de zona no estado TERMINAL (0 = evacuação completa)
            _zd, _zl = self._count_zone_violations()
            info["zone_drones_terminal"] = int(_zd)
            info["zone_links_terminal"] = int(_zl)
            info["zone_clear_terminal"] = int(1 if (_zd == 0 and _zl == 0) else 0)
            # DIAGNÓSTICO do custo absoluto de zona (ver _compute_reward).
            # zone_abs_pen_terminal = soma do custo aplicado no episódio. Se vier
            # ~0 com zone_viol_frac > 0, o termo NÃO está chegando ao reward e o
            # problema é de plumbing, não de magnitude. Se vier grande e
            # zone_clear seguir parado, o incentivo chega e não basta.
            _st = max(1, int(getattr(self, "_ep_steps_counted", 0)))
            info["zone_abs_pen_terminal"] = float(getattr(self, "_ep_zone_abs_pen", 0.0))
            info["zone_abs_pen_per_step_terminal"] = float(getattr(self, "_ep_zone_abs_pen", 0.0)) / _st
            info["zone_viol_frac_terminal"] = float(int(getattr(self, "_ep_zone_viol_steps", 0))) / _st
            info["zone_drones_mean_terminal"] = float(getattr(self, "_ep_zone_drones_sum", 0.0)) / _st
            info["zone_links_mean_terminal"] = float(getattr(self, "_ep_zone_links_sum", 0.0)) / _st

            self._last_reward = float(reward)

        # Return compatível Gymnasium vs Gym
        if _GYMNASIUM:
            return obs, float(reward), bool(done), bool(truncation), info
        return obs, float(reward), bool(done or truncation), info
    def _get_obs(self):
        """Observação com padding até max_drones e máscara de drones ativos.
        - As primeiras self.num_drones entradas representam drones ativos.
        - Os demais índices (até max_drones) ficam zerados e mask=0.
        """
        num = int(self.num_drones)
        pos = _torch_to_numpy_cpu_view(self.drone_positions[:num, :2])  # (num_drones, 2)
        pos_norm = (pos / max(1.0, (self.grid_size - 1))) * 2.0 - 1.0

        adj = _torch_to_numpy_cpu_view(self.adjacency_matrix[:num, :num])  # (num_drones, num_drones)

        max_n = int(self.max_drones)

        pos_full = np.zeros((max_n, 2), dtype=np.float32)
        adj_full = np.zeros((max_n, max_n), dtype=np.float32)
        mask = np.zeros((max_n,), dtype=np.float32)

        pos_full[:num, :] = pos_norm
        adj_full[:num, :num] = adj
        enabled = self._enabled_mask_np()
        lim = min(num, enabled.shape[0])
        if lim > 0:
            mask[:lim] = enabled[:lim].astype(np.float32, copy=False)

        obs_parts = [pos_full.reshape(-1), adj_full.reshape(-1), mask]

                # (opcional) mapa downsample da zona proibida
        if self.obs_zone_map:
            # Garante cache atualizado e, principalmente, tamanho FIXO
            # mesmo quando não há zona (p_sem_zona>0).
            expected = int(self._zone_map_ds * self._zone_map_ds) if int(getattr(self, '_zone_map_ds', 0) or 0) > 0 else 0
            sig = (int(getattr(self, '_zone_sig', 0)) ^ (int(self.zone_map_scale) << 16) ^ (int(self._zone_map_ds) << 1)) & 0xFFFFFFFF
            if (self._zone_map_down is None) or (self._zone_map_down_sig != sig) or (expected and (self._zone_map_down.shape[0] != expected)):
                self._rebuild_zone_map_cache()
            if expected > 0 and (self._zone_map_down is None or self._zone_map_down.shape[0] != expected):
                # fallback ultra-seguro: tamanho fixo
                self._zone_map_down = np.zeros((expected,), dtype=np.float32)
                self._zone_map_down_sig = sig
            obs_parts.append(self._zone_map_down if self._zone_map_down is not None else np.zeros((expected,), dtype=np.float32))

        # (opcional) matriz NxN indicando cruzamento de zona por aresta (i,j)
        if self.obs_edge_cross:
            cross_full = np.zeros((max_n, max_n), dtype=np.float32)
            cross_mat = self._ensure_edge_cross_mat()
            if cross_mat is not None and num > 0:
                # cross_mat é (num,num)
                cross_full[:num, :num] = cross_mat.astype(np.float32, copy=False)
            obs_parts.append(cross_full.reshape(-1))

        # (opcional) matriz NxN de qualidade por enlace: prob. de sucesso do par
        # (i,j) com distância ATUAL + sombreamento do episódio. Dá ao agente a
        # visibilidade de QUAL enlace/par está degradado — pré-requisito para
        # aprender reparo dirigido em vez de topologia fixa.
        if self.obs_link_quality:
            obs_parts.append(self._link_quality_obs_matrix().reshape(-1))

        return np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)

    def _link_quality_obs_matrix(self) -> np.ndarray:
        """Matriz (max_drones x max_drones) com a probabilidade de sucesso de
        quadro de cada par de drones HABILITADOS (0 fora disso). Cacheada por
        (posições, habilitados, sombra do episódio); recomputada só quando algo
        disso muda. O custo de recomputação é ~n² chamadas a _prob_link_pair,
        amortizadas pelo cache de _prob_link por distância."""
        max_n = int(self.max_drones)
        num = int(self.num_drones)
        try:
            dist_m = self._ensure_dist_matrix()
            enabled = self._enabled_mask_np()
            sig = (
                getattr(self, "_dist_pos_sig", None),
                enabled.tobytes(),
                id(getattr(self, "_shadow_db", None)),
            )
            if self._linkq_cache is not None and self._linkq_sig == sig:
                return self._linkq_cache
            q = np.zeros((max_n, max_n), dtype=np.float32)
            lim = int(min(num, enabled.shape[0]))
            idx_on = np.flatnonzero(enabled[:lim])
            for a in idx_on:
                ia = int(a)
                row = dist_m[ia]
                for b in idx_on:
                    ib = int(b)
                    if ib == ia:
                        continue
                    q[ia, ib] = self._prob_link_pair(ia, ib, float(row[ib]))
            self._linkq_cache = q
            self._linkq_sig = sig
            return q
        except Exception:
            return np.zeros((max_n, max_n), dtype=np.float32)

    # =========================
    # Fitness / Recompensa
    # =========================



    def _calculate_fit(self):
        """Calcula a aptidão do estado com reward ligado apenas aos objetivos centrais."""
        w = self.fit_w
        n = int(self.num_drones)

        if n <= 0:
            self.current_reliability = 0.0
            self.current_latency_penalty = 0.0
            self.current_latency_ms = 0.0
            self.current_throughput = 0.0
            self.current_loss_rate = 1.0
            self.current_max_feasible_rate = 0.0
            self.current_num_paths = 0
            self.last_qos_target_met = 0.0
            return 0.0, {
                "path_exists": False,
                "is_dag": 1,
                "left_links_count": 0,
                "reliability": 0.0,
                "throughput": 0.0,
                "latency_ms": 0.0,
                "loss_rate": 1.0,
                "max_feasible_rate_mbps": 0.0,
                "num_paths": 0,
                "active_drones": 0,
                "used_drones": 0,
                "qos_target_met": 0.0,
            }

        pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
        adj_np = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5).astype(np.float32, copy=False)

        ei, ej = np.nonzero(adj_np > 0.0)
        out_neighbors = [[] for _ in range(n)]
        if ei.size > 0:
            for i, j in zip(ei.tolist(), ej.tolist()):
                out_neighbors[int(i)].append(int(j))

        path_exists = bool(self._has_path(0, n - 1, adj_np, out_neighbors=out_neighbors)) if n > 1 else False

        is_dag = True
        if n > 1 and ei.size > 0:
            indeg_tmp = np.zeros(n, dtype=np.int32)
            for j in ej.tolist():
                indeg_tmp[int(j)] += 1
            stack = [i for i in range(n) if indeg_tmp[i] == 0]
            seen = 0
            while stack:
                u = stack.pop()
                seen += 1
                for v in out_neighbors[u]:
                    indeg_tmp[v] -= 1
                    if indeg_tmp[v] == 0:
                        stack.append(v)
            is_dag = (seen == n)

        left_links_count = 0
        if ei.size > 0:
            for i, j in zip(ei.tolist(), ej.tolist()):
                if float(pos_np[int(j), 0]) < float(pos_np[int(i), 0]):
                    left_links_count += 1

        metrics = self._evaluate_graph_metrics(pos_np, adj_np, use_abreu=False)
        reliability = float(metrics.get("reliability", 0.0))
        reliability_block = float(metrics.get("reliability_block", reliability))
        delivery_ratio = float(metrics.get("delivery_ratio", 0.0))
        reliability_hybrid = float(metrics.get("reliability_hybrid", reliability))
        throughput = float(metrics.get("throughput", 0.0))
        latency_ms = float(metrics.get("latency_ms", float('inf')))
        loss_rate = float(metrics.get("loss_rate", 1.0))
        max_feasible_rate = float(metrics.get("max_feasible_rate_mbps", 0.0))
        num_paths = int(metrics.get("num_paths", 0))

        self.current_reliability = reliability
        self.current_reliability_block = reliability_block
        self.current_delivery_ratio = delivery_ratio
        self.current_reliability_hybrid = reliability_hybrid
        self.current_throughput = throughput
        self.current_latency_ms = 0.0 if not np.isfinite(latency_ms) else latency_ms
        self.current_latency_penalty = float(max(0.0, self.current_latency_ms - float(getattr(self, "max_delay_ms", MAX_DELAY_MS_DEFAULT))))
        self.current_loss_rate = loss_rate
        self.current_max_feasible_rate = max_feasible_rate
        self.current_num_paths = num_paths

        active_drones, _used_drones = self._compute_active_and_used_drones()
        target_goodput, target_rel, target_delay = self._min_drone_targets()
        effective_goodput_target = float(self._effective_goodput_target())

        # -----------------------------
        # Recompensa lexicográfica (híbrida: primário potential-based)
        # -----------------------------
        # phi  = POTENCIAL PRIMÁRIO Phi(s): construção de topologia válida + KPIs
        #        até o alvo. Sob use_hybrid_reward, a contribuição ao reward é
        #        diferencial (gamma*Phi(s')-Phi(s)) -> pairar em subótimo rende ~0.
        # sec  = SECUNDÁRIO ABSOLUTO: manutenção do QoS, minimização de drones e
        #        bônus de latência baixa. Só liga após qos_target_met. Somado como
        #        valor absoluto (dá incentivo contínuo para manter/otimizar).
        # penalty_abs = penalidades ABSOLUTAS mesmo no modo híbrido (latência acima
        #        do limite): degradar SEMPRE custa, não só a variação.
        phi = 0.0
        sec = 0.0
        penalty_abs = 0.0

        # Progresso denso para o caso de reset sem links: fração de nós alcançáveis
        # a partir da origem e fração de nós que alcançam o destino.
        src_reach_frac = 0.0
        dst_reach_frac = 0.0
        if n > 1:
            try:
                reach = self._reachability_bits(adj_np > 0.5)
                src_reach_frac = float(int(reach[0]).bit_count()) / float(max(1, n))
                dst_bit = 1 << int(n - 1)
                dst_reach_frac = float(sum(1 for r in reach if (int(r) & dst_bit) != 0)) / float(max(1, n))
            except Exception:
                src_reach_frac = 1.0 / float(max(1, n))
                dst_reach_frac = 1.0 / float(max(1, n))
        connectivity_progress = max(0.0, min(1.0, 0.5 * (src_reach_frac + dst_reach_frac)))

        hard_topology_ok = bool(is_dag and left_links_count == 0)
        # --- MARCOS ESTRUTURAIS embutidos no potencial Phi (milestone shaping) ---
        # Ficam DENTRO de Phi, então a diferencial gamma*Phi'-Phi recompensa CRUZAR
        # cada marco (Phi salta) mantendo a garantia potential-based (Ng et al.).
        # Resolve o "refúgio perverso" medido: sem estes marcos, ficar SEM caminho
        # (penalidade fraca) pontuava MELHOR que manter um caminho de baixa
        # confiabilidade — a política era incentivada a DESTRUIR redes ruins.
        PATH_BASE_PENALTY = 5.0   # > pior phi com caminho: manter > quebrar, sempre
        if not path_exists:
            phi -= PATH_BASE_PENALTY + float(w.get("PATH_MISSING", 0.0)) * (1.0 - connectivity_progress)
        else:
            # MARCO: existe caminho A->B (degrau positivo).
            phi += 1.0 + 0.10 * float(w.get("PATH_MISSING", 0.0))
            # MARCO: redundância (>=2 caminhos). Com o custo de ativação abaixo,
            # a 2ª rota só compensa se seu ganho (marco + rel) superar o custo dos
            # drones ativados — a política decide "redundante ou não" pelo trade-off.
            if int(getattr(self, "current_num_paths", 0)) >= 2:
                phi += 1.0

        # CUSTO DE ATIVAÇÃO no potencial (modo construtivo): cada drone ativo
        # além do par origem/destino reduz Phi. Pela diferencial, ATIVAR custa
        # imediatamente (-c) e só compensa se melhorar KPIs/marcos (medido:
        # fechar o QoS ativando 1 drone rende ~ +7 >> custo 0.15). DESATIVAR
        # devolve o custo. Ensina "ativar apenas o necessário" sem gate de QoS.
        _act_cost = float(getattr(self, "drone_potential_cost", 0.0))
        if _act_cost > 0.0:
            phi -= _act_cost * float(max(0, int(active_drones) - 2))

        if not is_dag:
            phi -= float(w.get("CYCLE_PENALTY", 0.0))
        if left_links_count > 0:
            phi -= float(w.get("LEFT_LINK_COEF", 0.0)) * float(left_links_count)

        rel_ok = bool(float(reliability_block) + 1e-12 >= float(target_rel))
        delivery_ok = bool(float(delivery_ratio) + 1e-12 >= float(getattr(self, "delivery_ratio_target", 0.999)))
        tp_min_ok = bool(float(throughput) + 1e-9 >= float(effective_goodput_target))
        tp_max_ok = bool(float(throughput) <= float(getattr(self, "throughput_max_mbps", THROUGHPUT_MAX_MBPS)) + 1e-9)
        tp_ok = bool(tp_min_ok and tp_max_ok)

        if path_exists and hard_topology_ok:
            term_rel_w = float(w.get("TERM_REL", 0.0))
            throughput_w = float(w.get("THROUGHPUT_PENALTY_MAX", 0.0))
            latency_w = float(w.get("LATENCY_OVER_LIMIT_W", 0.0))
            delivery_w = 0.50 * term_rel_w

            # --------------------------------------------------------------
            # Recompensa de KPI que CONTINUA melhorando após atingir o alvo.
            # Antes os scores eram clampados em 1.0, então rel=0.995 e rel=0.999
            # davam a MESMA recompensa — o agente não tinha incentivo para
            # melhorar um KPI já atingido. Agora cada KPI usa:
            #   - até o alvo: score linear de -1 a +1 (mantém o gradiente de
            #     construção da viabilidade, igual a antes);
            #   - acima do alvo: bônus extra que cresce com a folga até o teto
            #     físico (1.0), com retornos decrescentes (sqrt) para não
            #     ofuscar a minimização de drones.
            # _kpi_reward(valor, alvo, teto) devolve esse score contínuo.
            # --------------------------------------------------------------
            def _kpi_reward(value: float, target: float, ceiling: float = 1.0) -> float:
                t = max(1e-12, float(target))
                v = float(value)
                if v <= t:
                    # região de construção: -1 (zero) .. +1 (no alvo)
                    return 2.0 * max(0.0, v / t) - 1.0
                # região de aprimoramento: +1 no alvo, sobe até +2 no teto,
                # com retornos decrescentes (raiz) na folga normalizada.
                head = max(1e-12, float(ceiling) - t)
                extra = max(0.0, min(1.0, (v - t) / head))
                return 1.0 + math.sqrt(extra)

            if term_rel_w != 0.0:
                # --------------------------------------------------------------
                # RAMPA "FAROL" para a confiabilidade (resolve o reward hacking
                # de pairar em subótimo). Estrutura:
                #   - platô baixo e levemente crescente até o "joelho" (knee):
                #     o subótimo rende pouco, então PAIRAR não acumula recompensa,
                #     mas o gradiente fraco e POSITIVO ainda dá direção (sem deserto);
                #   - subida íngreme e CONTÍNUA do joelho até o alvo (o "farol"):
                #     puxa a política para atingir o KPI, sem degrau/descontinuidade
                #     (mais estável para o crítico que um salto vertical);
                #   - acima do alvo: continua subindo de forma suave (sqrt).
                # O warm-start coloca a política já perto do alvo (tocando o farol
                # com frequência), então o platô fraco não inviabiliza o aprendizado.
                # --------------------------------------------------------------
                def _rel_ramp(value: float, target: float,
                              knee: float = 0.65, plato_top: float = 0.20,
                              peak: float = 2.2) -> float:
                    # Recalibrada após o bootstrap resolver o reward hacking na
                    # raiz: o platô largo (knee=0.90) existia para impedir
                    # acúmulo de recompensa pairando no subótimo, mas com o
                    # bootstrap essa proteção é desnecessária e o platô virava
                    # uma ZONA MORTA DE GRADIENTE — a política travava em
                    # rel~0.88 (logo abaixo do knee antigo), onde o Abreu mostra
                    # que 0.995 é fisicamente alcançável (rel_block=1.0 com 3-4
                    # caminhos). Agora: knee baixo (0.75) e platô curto, e a
                    # subida tem gradiente forte DESDE o knee (perfil côncavo
                    # sqrt, não convexo frac^2 que era plano logo acima do knee).
                    t = max(1e-12, float(target))
                    v = float(value)
                    kn = min(knee, t * 0.95)  # garante knee < alvo mesmo se alvo baixo
                    if v <= kn:
                        # platô curto: 0..plato_top, linear (gradiente fraco mas presente)
                        return plato_top * max(0.0, v / max(1e-12, kn))
                    if v <= t:
                        # farol: subida íngreme com gradiente FORTE desde o knee
                        # (sqrt é côncavo: derivada máxima em frac=0, ao contrário
                        # de frac^2 que era ~plano logo acima do knee).
                        frac = (v - kn) / max(1e-12, t - kn)
                        return plato_top + (peak - plato_top) * math.sqrt(frac)
                    # acima do alvo: SATURA em `peak` (constante). Antes era
                    # `peak + 0.8*sqrt(extra)`, cujo gradiente EXPLODIA logo acima
                    # do alvo (derivada infinita do sqrt em extra=0), criando
                    # incentivo de +2.4/passo para empurrar rel 0.995->1.0 — o que
                    # exige MAIS drones e conflitava diretamente com a minimização
                    # de drones (objetivo de tese). Com a saturação, atingir o alvo
                    # ZERA o gradiente de confiabilidade, e o custo de estado por
                    # drone ativo passa a dominar: a política minimiza drones SÓ
                    # depois de atingir os KPIs. Confiabilidade ≥ alvo é tratada
                    # como "objetivo cumprido", não como "quanto mais melhor".
                    return peak

                reward_rel = term_rel_w * _rel_ramp(reliability_block, target_rel)
                phi += reward_rel
                # EXTENSÃO SUAVE (opcional, default 0=off): pequeno segmento de
                # rampa entre reliability_target e delivery_ratio_target, para
                # não haver zona morta caso um episódio pare em [0.995, 0.999).
                # Peso deliberadamente pequeno: o custo de poda (0.2/passo)
                # continua dominando, então a extensão NÃO segura drones
                # redundantes só para defender excedente de confiabilidade.
                ext_w = float(getattr(self, "rel_ramp_ext_w", 0.0))
                if ext_w > 0.0:
                    deliv_t = float(getattr(self, "delivery_ratio_target", 0.999))
                    if deliv_t > target_rel and reliability_block > target_rel:
                        frac_ext = min(1.0, (float(reliability_block) - target_rel) / (deliv_t - target_rel))
                        phi += ext_w * frac_ext

            if delivery_w != 0.0:
                phi += delivery_w * _kpi_reward(
                    delivery_ratio, float(getattr(self, "delivery_ratio_target", 0.999)), 1.0)

            if throughput_w != 0.0:
                # Vazão: score de construção até o alvo; acima do alvo NÃO premia
                # subir indefinidamente (mais vazão não é objetivo) — só garante
                # que não caiu abaixo. O teto continua sendo penalizado adiante.
                tp_score = 2.0 * max(0.0, min(1.0, float(throughput) / max(1e-9, float(effective_goodput_target)))) - 1.0
                phi += throughput_w * tp_score
                # Depois que a vazão mínima foi atendida, evita carga/redundância excessiva.
                if tp_min_ok:
                    tp_limit_start = 0.85 * float(getattr(self, "throughput_max_mbps", THROUGHPUT_MAX_MBPS))
                    if throughput > tp_limit_start:
                        ratio = (throughput - tp_limit_start) / max(1e-9, float(getattr(self, "throughput_max_mbps", THROUGHPUT_MAX_MBPS)) - tp_limit_start)
                        ratio = max(0.0, min(1.0, float(ratio)))
                        phi -= throughput_w * ratio

            if latency_w != 0.0:
                if self.current_latency_ms > float(target_delay):
                    # Penalidade por exceder o limite: ABSOLUTA (latência alta é
                    # ruim sempre, e no modo híbrido deve custar mesmo sem variação
                    # de potencial — protege contra a política deixar a latência
                    # degradar após atingir os KPIs).
                    over = (self.current_latency_ms - float(target_delay)) / max(1.0, float(target_delay))
                    penalty_abs -= latency_w * float(over)
                else:
                    # FOLGA DE LATÊNCIA: gradiente denso ANTES do limite. As
                    # "quase-perdas" (rel~1.0 com lat 276-347ms por excesso de
                    # tranças) não recebiam sinal nenhum até estourar o limite —
                    # a penalidade cresce linearmente de 0 (em 80% do limite) a
                    # `latency_headroom_penalty` (no limite), incentivando podar
                    # tranças enquanto a folga se esgota, sem punir lat saudável.
                    hw = float(getattr(self, "latency_headroom_penalty", 0.0))
                    if hw > 0.0:
                        # GATE DE FASE: a penalidade de folga só age na fase de
                        # LAPIDAÇÃO (rel já razoável), nunca na OBRA. Sem o gate,
                        # a travessia do vale (adicionar tranças eleva rel E
                        # latência, com a rede em construção vivendo acima de
                        # 80% do limite) pagava pedágio contínuo de até
                        # hw/passo — gradiente contrário exatamente na direção
                        # do progresso. Observado em treino real: estrutura
                        # saturada em ~4M e rel terminal presa em 0.26-0.36 por
                        # 8M de steps, com a perda do crítico explodindo.
                        rel_gate = float(getattr(self, "current_reliability_block", 0.0))
                        if rel_gate >= 0.5:
                            frac = float(self.current_latency_ms) / max(1e-9, float(target_delay))
                            if frac > 0.80:
                                penalty_abs -= hw * (frac - 0.80) / 0.20
                    if rel_ok and tp_ok and delivery_ok:
                        # BÔNUS por latência baixa: objetivo SECUNDÁRIO de eficiência,
                        # gateado por todos os KPIs primários. Absoluto (não diferencial).
                        lat_score = 1.0 + (1.0 - float(self.current_latency_ms) / max(1e-9, float(target_delay)))
                        sec += latency_w * max(0.0, lat_score)

        qos_target_met = 1.0 if self._success_targets_met(metrics) else 0.0
        self.last_qos_target_met = float(qos_target_met)

        # MARCO: QoS pleno (degrau +3 no potencial). A diferencial recompensa
        # CRUZAR o QoS e re-penaliza PERDÊ-LO, mantendo a garantia potential-based.
        if qos_target_met >= 1.0:
            phi += 3.0

        prune_ready = bool(path_exists and delivery_ratio + 1e-12 >= float(getattr(self, "prune_delivery_min", 0.30)) and reliability_block + 1e-12 >= float(getattr(self, "prune_rel_min", 0.20)))
        # Conflito de gradiente corrigido: NÃO penalize drones enquanto a
        # confiabilidade ainda depende de adicionar caminhos redundantes.
        # O custo pré-QoS só entra quando a confiabilidade já tem folga sobre o
        # alvo (margem), indicando que pode haver drones de fato redundantes a podar.
        # Custo de drone PRÉ-QoS é um incentivo SECUNDÁRIO (minimizar drones).
        # Pela diretriz: só pode operar quando os KPIs PRIMÁRIOS já estão sendo
        # atendidos COM FOLGA — caso contrário fica bloqueado, para nunca empurrar
        # a política a remover drones às custas de um KPI. Antes a condição usava
        # 1.05*target_rel (>=1.045), fisicamente impossível, então o custo NUNCA
        # disparava. Agora exige rel/delivery/throughput OK com margem pequena e
        # absoluta sobre o alvo (folga real, não múltiplo inviável).
        rel_slack = float(min(1.0, float(target_rel) + 0.002))
        deliv_target = float(getattr(self, "delivery_ratio_target", 0.999))
        deliv_slack = float(min(1.0, deliv_target + 0.0005))
        kpis_with_margin = bool(
            float(reliability_block) + 1e-12 >= rel_slack
            and float(delivery_ratio) + 1e-12 >= deliv_slack
            and tp_ok
        )
        if prune_ready and kpis_with_margin and qos_target_met < 1.0:
            drone_cost = float(w.get("DRONE_ACTIVE_COST_BEFORE_QOS", 0.0))
            # BASELINE ANALÍTICO: cobra apenas o EXCEDENTE sobre o mínimo de
            # drones que o cenário exige para QoS pleno (ver
            # _analytic_min_qos_drones). Antes cobrava (ativos-2), o que punia
            # drones NECESSÁRIOS e invertia o incentivo em cenários grandes
            # (break-even do bônus de manutenção em 32 ativos).
            drones_min = int(self._analytic_min_qos_drones())
            sec -= drone_cost * float(max(0, int(active_drones) - drones_min))

        if qos_target_met >= 1.0:
            # BÔNUS DE MANUTENÇÃO: enquanto o QoS pleno é mantido, cada passo rende
            # este bônus. É o que faz "atingir E manter o ótimo" superar "pairar em
            # subótimo": o potencial primário satura (~diferencial 0) após o alvo,
            # então sem este termo não haveria incentivo para SUSTENTAR o QoS. É
            # secundário e absoluto, consistente entre episódios.
            # SUPRESSÃO POR ZONA: o bônus de manutenção de QoS também é escalado
            # pelo fator de violação. Enquanto sujo, manter QoS não rende — só
            # rende depois de limpar. (Φ continua função pura do estado: o fator
            # depende só das violações atuais, não de histórico -> não-farmável.)
            sec += float(getattr(self, "qos_maintain_bonus", 2.0)) * self._zone_success_factor()
            # Depois de cumprir QoS completo, a poda de drones passa a ser mais forte.
            # Cobra apenas o EXCEDENTE sobre o mínimo analítico do cenário: manter
            # o QoS com o número necessário de drones NUNCA rende menos que pairar
            # em subótimo, independentemente da escala do cenário (antes o
            # break-even era 32 ativos: 6.0/0.2 + 2).
            drone_cost = float(w.get("DRONE_ACTIVE_COST_AFTER_QOS", 0.0))
            drones_min = int(self._analytic_min_qos_drones())
            sec -= drone_cost * float(max(0, int(active_drones) - drones_min))

        # -----------------------------------------------------------------
        # COMBINAÇÃO HÍBRIDA
        # -----------------------------------------------------------------
        # ---- EVACUAÇÃO DE ZONAS PROIBIDAS (termo no POTENCIAL) ----
        # A máscara já IMPEDE criar dentro da zona (mover para dentro / ADD de
        # aresta que cruza), mas nada empurra para SAIR quem já está lá — e uma
        # rota que atravessa a zona pode ter KPI ÓTIMO, então os KPIs sozinhos
        # não dão gradiente de evacuação. Entra no Φ (não como penalidade
        # absoluta) porque assim o progresso PARCIAL é recompensado: evacuar 3
        # de 5 drones já rende, e a política não ganha nada terminando o
        # episódio para escapar do custo (Φ é diferencial).
        zone_drones_v, zone_links_v = 0, 0
        zone_w = float(getattr(self, "zone_violation_penalty", 0.0))
        if zone_w > 0.0 and self.zonas_proibidas:
            zone_drones_v, zone_links_v = self._count_zone_violations()
            if zone_drones_v or zone_links_v:
                # normaliza pelo total ativo: o gradiente por unidade evacuada é
                # estável independentemente do tamanho da rede.
                n_act = max(1, int(np.count_nonzero(self._enabled_mask_np())))
                # PESO MAIOR PARA LINKS: eliminar um link que cruza a zona exige
                # um DESVIO COMPLETO da rota (alonga caminho, adiciona hops,
                # degrada KPI), enquanto evacuar um drone é barato. Com peso
                # igual, a política racionalmente evacua os drones e PAGA os
                # links (observado: drones_dentro->0.3 mas links_cruzando travado
                # em ~1.2, impedindo zone_clear). O multiplicador compensa o
                # custo de KPI extra do desvio, tornando eliminar link vantajoso.
                link_mult = float(getattr(self, "zone_link_weight_mult", 1.0))
                viol = (float(zone_drones_v) / n_act) + link_mult * (float(zone_links_v) / max(1, n_act))
                phi -= zone_w * viol

        # ------------------------------------------------------------------
        # CUSTO ABSOLUTO DE ZONA (por passo, FORA do potencial Φ)
        # ------------------------------------------------------------------
        # Por que isto é necessário: o termo de zona no Φ é potential-based, logo
        # TELESCOPA — paga a TRANSIÇÃO de limpar e ~nada por PERMANECER limpo.
        # O custo episódico de continuar violando é ≈ 0. A supressão
        # (_zone_success_factor) tampouco resolve: ela MULTIPLICA bônus
        # positivos, então remove uma fração de algo positivo e nunca torna o
        # resultado negativo. Consequência medida (27M steps sem movimento):
        # com QoS alto, "atravessar a zona mantendo QoS" rende ~+194 contra 0 de
        # "contornar sem fechar QoS" — e o prêmio por atravessar CRESCE conforme
        # o QoS melhora. A política racionalmente paga a supressão e atravessa.
        #
        # Um custo ABSOLUTO por passo muda o sinal: violar passa a render menos
        # que zero, que é o que falta para o objetivo ser "limpar zona COM KPI"
        # em vez de "KPI a qualquer custo".
        #
        # Usa o MESMO viol normalizado do Φ (com zone_link_weight_mult), para
        # que as duas pressões sejam consistentes — antes, a supressão contava
        # drones e links com peso IGUAL enquanto o Φ pesava link 3.5x.
        # INSTRUMENTAÇÃO — contagem SEMPRE, independente do peso.
        # Se os contadores só rodassem dentro do `if zone_abs_w > 0`, um termo
        # que não executa produziria frac_viol=0, indistinguível de "não houve
        # violação" — exatamente a ambiguidade que esta instrumentação existe
        # para eliminar. Aqui viol é medido sempre que há zonas; a penalidade é
        # somada em separado. Assim, zviol>0 com zpen=0 significa
        # inequivocamente que o custo NÃO está chegando ao reward.
        if self.zonas_proibidas:
            try:
                _zdi, _zli = self._count_zone_violations()
            except Exception:
                _zdi, _zli = 0, 0
            if _zdi or _zli:
                self._ep_zone_viol_steps = int(getattr(self, "_ep_zone_viol_steps", 0)) + 1
                self._ep_zone_drones_sum = float(getattr(self, "_ep_zone_drones_sum", 0.0)) + float(_zdi)
                self._ep_zone_links_sum = float(getattr(self, "_ep_zone_links_sum", 0.0)) + float(_zli)
        self._ep_steps_counted = int(getattr(self, "_ep_steps_counted", 0)) + 1

        zone_abs_w = float(getattr(self, "zone_violation_abs_penalty", 0.0))
        if zone_abs_w > 0.0 and self.zonas_proibidas:
            if not (zone_w > 0.0):
                # o bloco do Φ não rodou; conta as violações aqui
                zone_drones_v, zone_links_v = self._count_zone_violations()
            if zone_drones_v or zone_links_v:
                # NORMALIZAÇÃO NÃO-DILUÍVEL. A versão anterior dividia por
                # n_act (herdado do termo do Φ), o que criava uma brecha: ATIVAR
                # MAIS DRONES reduzia a penalidade sem evacuar nada. Medido no
                # run D: active_drones subiu de ~13,5 para ~14,3 (picos de 16,2)
                # e o custo caiu ~5% sem nenhuma melhora de zone_clear.
                #
                # Pior, a brecha empurrava a rede para uma faixa em que ela não
                # funciona. Com carrier_sense=200 m todos os enlaces dividem o
                # mesmo domínio de colisão; medido com o solver, 16 e 18 drones
                # dão 0,0% de QoS (rho explode), contra ~50% com 14. Nos logs do
                # run D a correlação active_drones x qos é -0,965: com 13 drones
                # o qos médio é 0,996; acima de 15,5 cai para 0,483.
                #
                # Agora o custo é ABSOLUTO na contagem de violações. Um drone a
                # mais nunca reduz a penalidade; só evacuar reduz.
                link_mult = float(getattr(self, "zone_link_weight_mult", 1.0))
                viol_abs = float(zone_drones_v) + link_mult * float(zone_links_v)
                # escala mantida comparável à normalização antiga (~n_act típico),
                # para que os pesos por fase não precisem ser recalibrados do zero
                viol_abs /= float(max(1.0, getattr(self, "zone_abs_penalty_scale", 13.0)))
                _pen = zone_abs_w * viol_abs
                penalty_abs -= _pen
                self._ep_zone_abs_pen = float(getattr(self, "_ep_zone_abs_pen", 0.0)) + float(_pen)

        # phi é o POTENCIAL primário do estado atual. Sob use_hybrid_reward, a
        # contribuição primária ao reward é diferencial (potential-based shaping,
        # Ng et al. 1999): r_prim = gamma*Phi(s') - Phi(s). No 1º passo do episódio
        # (potential_prev=None), usa-se Phi como baseline (contribuição primária 0),
        # evitando um salto espúrio. Os secundários (sec) e as penalidades
        # absolutas (penalty_abs) são somados diretamente.
        if bool(getattr(self, "use_hybrid_reward", True)):
            phi_prev = self._primary_potential_prev
            g = float(getattr(self, "hybrid_gamma", 0.99))
            if phi_prev is None:
                r_primary = 0.0  # baseline no primeiro passo
            else:
                r_primary = g * float(phi) - float(phi_prev)
            self._primary_potential_prev = float(phi)
            reward = r_primary + sec + penalty_abs
        else:
            # modo legado: recompensa absoluta por estado (phi somado direto)
            reward = float(phi) + sec + penalty_abs

        info = {
            "path_exists": bool(path_exists),
            "is_dag": int(1 if is_dag else 0),
            "left_links_count": int(left_links_count),
            "reliability": float(reliability),
            "reliability_block": float(reliability_block),
            "delivery_ratio": float(delivery_ratio),
            "reliability_hybrid": float(reliability_hybrid),
            "throughput": float(throughput),
            "latency_ms": float(self.current_latency_ms),
            "loss_rate": float(loss_rate),
            "max_feasible_rate_mbps": float(max_feasible_rate),
            "num_paths": int(num_paths),
            "active_drones": int(active_drones),
            "qos_target_met": float(qos_target_met),
            "prune_ready": int(1 if prune_ready else 0),
            "delivery_ratio_target": float(getattr(self, "delivery_ratio_target", 0.999)),
        }
        return float(reward), info

    @staticmethod
    def _reachability_bits(adj_on: np.ndarray) -> np.ndarray:

        """Transitive closure (alcançabilidade) usando bitsets.
        Retorna ``reach[i]`` como um inteiro Python com bit ``j``=1 se ``j`` é
        alcançável a partir de ``i``.

        Observação: o ambiente agora pode usar até 50 drones, então ``uint32``
        não é suficiente. Aqui usamos inteiros Python (dtype=object), evitando
        overflow para ``1 << j`` quando ``j >= 32``.
        """
        n = int(adj_on.shape[0])
        reach = np.zeros(n, dtype=object)
        # Inicial: vizinhos diretos + self
        for i in range(n):
            bits = 1 << i
            js = np.nonzero(adj_on[i])[0]
            for j in js.tolist():
                bits |= 1 << int(j)
            reach[i] = bits
        # Warshall com bitsets
        for k in range(n):
            kbit = 1 << k
            rk = int(reach[k])
            for i in range(n):
                if (int(reach[i]) & kbit) != 0:
                    reach[i] = int(reach[i]) | rk
        return reach

    # =========================
    # Action masking (sb3-contrib)
    # =========================
    def action_masks(self) -> np.ndarray:
        """Máscara booleana (achatada) compatível com MultiDiscrete.

        Ordem: concatena máscaras de cada dimensão do MultiDiscrete.
        - move_idx: impede mover drones inativos e as âncoras (0 e N-1)
        - step_code/dir_code: sem máscara fina (sempre True)
        - toggle_op: permite ADD/REMOVE válidos no estado atual, e pode aplicar hard-masks
          (zona, left-link e ciclos) para impor restrições como *hard constraints*.
        """
        if self._mask_cache is not None and self._mask_cache_version == self._mask_version:
            return self._mask_cache

        nvec = np.asarray(self.action_space.nvec, dtype=np.int64)
        n = int(self.num_drones)
        enabled = self._enabled_mask_np()

        # --- move_idx ---
        m_move = np.zeros(int(nvec[0]), dtype=bool)
        if n > 0:
            lim = min(n, int(nvec[0]))
            m_move[:lim] = enabled[:lim]
            m_move[0] = False
            if n - 1 >= 0 and (n - 1) < m_move.shape[0]:
                m_move[n - 1] = False

        # --- step_code / dir_code ---
        m_step = np.ones(int(nvec[1]), dtype=bool)
        m_dir = np.ones(int(nvec[2]), dtype=bool)

        # --- toggle_op ---
        E = int(self._E)
        m_toggle = np.zeros(int(nvec[3]), dtype=bool)
        m_toggle[0] = True

        # Pré-definições para evitar NameError quando n<2
        src_all = self._edge_src
        dst_all = self._edge_dst
        idx_add = np.array([], dtype=np.int64)
        pos_np = None

        if n >= 2:
            # Views CPU->numpy (sem cópia) + lógica vetorizada: evita loop Python por aresta.
            adj_on = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5)
            out_deg = adj_on.sum(axis=1).astype(np.int32, copy=False)

            src_all = self._edge_src
            dst_all = self._edge_dst
            active = (src_all < n) & (dst_all < n)
            if np.any(active):
                idx_active = np.nonzero(active)[0]
                src_a = src_all[idx_active]
                dst_a = dst_all[idx_active]
                enabled_pair = enabled[src_a] & enabled[dst_a]
                if np.any(~enabled_pair):
                    idx_active = idx_active[enabled_pair]
                    src_a = src_a[enabled_pair]
                    dst_a = dst_a[enabled_pair]

                is_on = adj_on[src_a, dst_a]

                # REMOVE: válido se existe E se não desconectar 0 -> N-1
                # (TRAVA: arestas-ponte do único caminho A->B ficam bloqueadas).
                idx_rem = idx_active[is_on]
                if idx_rem.size:
                    if not bool(self._has_path(0, n - 1, adj_on)):
                        # Já desconectado: não há caminho a preservar.
                        m_toggle[1 + E + idx_rem] = True
                    else:
                        src_rem = src_all[idx_rem]
                        dst_rem = dst_all[idx_rem]
                        keep = np.ones(int(idx_rem.shape[0]), dtype=bool)
                        adj_try = adj_on.copy()
                        for k in range(int(idx_rem.shape[0])):
                            u = int(src_rem[k]); v = int(dst_rem[k])
                            adj_try[u, v] = False
                            keep[k] = bool(self._has_path(0, n - 1, adj_try))
                            adj_try[u, v] = True
                        idx_keep = idx_rem[keep]
                        if idx_keep.size:
                            m_toggle[1 + E + idx_keep] = True

                # ADD: válido se NÃO existe e respeita regra hard de max_out
                add_ok = (~is_on) & (out_deg[src_a] < int(self.max_out_connections))
                idx_add = idx_active[add_ok]
                if idx_add.size:
                    # (mask-relax) não bloqueia left-link nem zona
                    m_toggle[1 + idx_add] = True

        # 4b) hard-mask (left-links): bloqueia ADD quando x_dst < x_src
        if self.hard_mask_left_links and idx_add.size:
            if pos_np is None:
                pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
            src_add = src_all[idx_add]
            dst_add = dst_all[idx_add]
            left = (pos_np[dst_add, 0] < pos_np[src_add, 0])
            if np.any(left):
                m_toggle[1 + idx_add[left]] = False

        # 4b.5) hard-mask (raio de vizinhança): bloqueia ADD de links longos demais
        radius_m = float(getattr(self, "link_neighbor_radius_m", 0.0))
        if radius_m > 0.0 and idx_add.size:
            if pos_np is None:
                pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
            src_add = src_all[idx_add]
            dst_add = dst_all[idx_add]
            dx_m = (pos_np[src_add, 0] - pos_np[dst_add, 0]) * float(self.square_size_m)
            dy_m = (pos_np[src_add, 1] - pos_np[dst_add, 1]) * float(self.square_size_m)
            dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
            too_far = dist_m > radius_m
            if np.any(too_far):
                m_toggle[1 + idx_add[too_far]] = False

        # 4c) hard-mask (ciclos): bloqueia ADD que criaria ciclo (dst alcança src)
        # BUGFIX: a variável era chamada `enabled`, SOMBREANDO o array de drones
        # ativos usado adiante pelos loops de ativar/desativar e por active_count
        # (que passavam a indexar a máscara de ops de ADD). Renomeada.
        if self.hard_mask_cycles and idx_add.size:
            add_enabled = m_toggle[1 + idx_add]
            if np.any(add_enabled):
                idx_chk = idx_add[add_enabled]
                src_chk = src_all[idx_chk]
                dst_chk = dst_all[idx_chk]
                reach = self._reachability_bits(adj_on)
                cyc = np.asarray([
                    ((int(reach[int(dst)]) >> int(src)) & 1) != 0
                    for src, dst in zip(src_chk.tolist(), dst_chk.tolist())
                ], dtype=bool)
                if np.any(cyc):
                    m_toggle[1 + idx_chk[cyc]] = False



        # 5) hard-mask (zona): bloqueia ADD de arestas que cruzam zona proibida
        if self.hard_mask_zone_edges and self.zonas_proibidas and idx_add.size:
            cross_mat = self._ensure_edge_cross_mat()
            if cross_mat is not None:
                src_add = src_all[idx_add]
                dst_add = dst_all[idx_add]
                cross = cross_mat[src_add, dst_add].astype(bool, copy=False)
                if np.any(cross):
                    m_toggle[1 + idx_add[cross]] = False

        activate_base = int(getattr(self, "_activate_op_base", 1 + 2 * E))
        deactivate_base = int(getattr(self, "_deactivate_op_base", 1 + 2 * E + self.max_drones))
        active_count = int(np.count_nonzero(enabled))
        # TETO DE ATIVAÇÃO: com folga suficiente sobre o mínimo analítico do
        # cenário (baseline + slack), ativar mais drones só piora vazão/latência
        # (redundância disjunta + contenção). Mascara ATIVAR nesse regime para a
        # exploração não cavar o buraco de que a poda precisaria escapar.
        try:
            act_cap = int(self._analytic_min_qos_drones()) + int(getattr(self, "activation_slack_over_min", 6))
        except Exception:
            act_cap = int(self.num_drones)
        if active_count < act_cap:
            for idx in range(1, max(1, n - 1)):
                if idx < enabled.size and (not bool(enabled[idx])):
                    op_idx = activate_base + int(idx)
                    if 0 <= op_idx < m_toggle.shape[0]:
                        m_toggle[op_idx] = True
        min_keep = 2
        prune_gate_ok = bool(getattr(self, "last_qos_target_met", 0.0) >= 1.0)
        if not prune_gate_ok:
            # Desbloqueia poda leve antes do QoS completo quando há caminho
            # e as métricas parciais já passaram dos limiares de poda.
            try:
                prune_gate_ok = bool(float(getattr(self, "last_qos_target_met", 0.0)) >= 1.0 or self._current_prune_gate())
            except Exception:
                prune_gate_ok = False
        if prune_gate_ok and active_count > min_keep:
            for idx in range(1, max(1, n - 1)):
                # TRAVA: só libera desativar drones cuja saída NÃO desconecta
                # o caminho 0 -> N-1.
                if idx < enabled.size and bool(enabled[idx]) and (not self._deactivate_disconnects(idx)):
                    op_idx = deactivate_base + int(idx)
                    if 0 <= op_idx < m_toggle.shape[0]:
                        m_toggle[op_idx] = True

        mask = np.concatenate([m_move, m_step, m_dir, m_toggle]).astype(bool, copy=False)
        self._mask_cache = mask
        self._mask_cache_version = self._mask_version
        return mask


# =========================
# Multi-agent wrapper
# =========================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-agent wrapper (turn-based) for your DroneEnv, designed for MARLlib/RLlib.

Goal:
- 2 policies:
  - "anchor" for the source anchor drone (index 0), responsible for its outgoing links.
  - "inter" shared across all intermediate drones (1..N-2), responsible for movement + outgoing links.

Key design choices:
- Turn-based MARL: at each env step, only ONE agent acts (round-robin over agents).
  This matches your original DroneEnv semantics (one move per step) and avoids a large joint action.
- Centralized reward: every agent receives the same global reward from DroneEnv.step().
- Action is flattened to a SINGLE Discrete so action masking is practical (toggle + movement-hard masks):
    inter_action_id = move_op * TOGGLE_LOCAL_N + toggle_local
    where:
      move_op in [0, MOVE_N-1], MOVE_N = 1 + STEP_CELLS_N*4
      toggle_local in [0, TOGGLE_LOCAL_N-1], TOGGLE_LOCAL_N = 1 + 2*MAX_DRONES
  Anchor action is just toggle_local (Discrete(TOGGLE_LOCAL_N)).

Observations:
- Dict with keys:
  - "obs":     augmented observation = [base_obs, onehot(drone_idx), role_bit]
  - "state":   base_obs (for CTDE / centralized critic)
  - "action_mask": 0/1 vector compatible with the Discrete action id of the current agent

Terminal-only metrics:
- We strip *_terminal fields from info on non-terminal steps to avoid "dirty start" contamination.
  On terminal steps, we keep (and thus log) fields like:
  reliability_terminal, right_links_ratio_terminal, zone_violation_terminal, success_terminal, etc.
"""

# (removed) from __future__ import annotations  # must be at file top

from typing import Dict, Tuple, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Compatibilidade sem Ray/RLlib:
# O trainer nativo PyTorch usa Drone5GHAPPOPyEnv e não precisa carregar Ray em cada worker.
# Mantemos uma classe mínima apenas para preservar os wrappers legados abaixo.
class MultiAgentEnv:  # type: ignore
    pass



def _decode_move_op(move_op: int) -> Tuple[int, int]:
    """
    move_op encoding:
      0 -> no move (step_code=0, dir_code=0)
      1.. -> (step_code, dir_code) where dir_code in {1:+x,2:-x,3:+y,4:-y}
    """
    if move_op <= 0:
        return 0, 0
    k = move_op - 1
    step_code = k // 4
    dir_code = 1 + (k % 4)
    return step_code, dir_code

def _dir_to_dxdy(dir_code: int) -> Tuple[int, int]:
    if dir_code == 1:
        return 1, 0
    if dir_code == 2:
        return -1, 0
    if dir_code == 3:
        return 0, 1
    if dir_code == 4:
        return 0, -1
    return 0, 0



class Drone5GMAEnv(MultiAgentEnv):
    """
    Turn-based MultiAgentEnv wrapper around DroneEnv.

    Agent ids (prefix-based, for MARLlib group policy sharing):
      - "anchor_0"          (drone index 0)
      - "inter_1"...        (drone indices 1..N-2)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        # RLlib MultiAgentEnv.__init__ NÃO aceita env_config como argumento.
        # Chamadas do tipo super().__init__(config) causam:
        #   TypeError: MultiAgentEnv.__init__() takes 1 positional argument but 2 were given
        try:
            super().__init__()
        except TypeError:
            try:
                MultiAgentEnv.__init__(self)  # type: ignore[misc]
            except Exception:
                pass

        config = dict(config or {})

        # Wrapper-only keys (o DroneEnv embutido não aceita)
        episode_horizon = config.pop("episode_horizon", None)
        self.mask_actions = bool(int(config.pop("mask_actions", 1)))

        # Converte zona_w/zona_h -> zona_size (DroneEnv espera zona_size=(w,h))
        if "zona_size" not in config and ("zona_w" in config or "zona_h" in config):
            zw = int(config.pop("zona_w", 15))
            zh = int(config.pop("zona_h", 15))
            config["zona_size"] = (zw, zh)

        # Underlying single-agent environment
        self.base_env = DroneEnv(**config)

        # Aplica horizonte no base env (Gymnasium-style)
        if episode_horizon is not None:
            try:
                self.base_env._max_episode_steps = int(episode_horizon)
            except Exception:
                pass

        # Initialize wrapper spaces/state regardless of the optional horizon.
        self.max_drones: int = int(getattr(self.base_env, "max_drones", config.get("max_drones", 13)))
        self._E: int = int(getattr(self.base_env, "_E", 0))
        self._edge_pairs: List[Tuple[int, int]] = list(getattr(self.base_env, "_edge_pairs", []))
        self._edge_pair_to_idx: Dict[Tuple[int, int], int] = {(s, d): i for i, (s, d) in enumerate(self._edge_pairs)}

        # Local toggle encoding (per-agent, only src fixed; agent chooses dst + add/remove)
        # 0: noop
        # 1..MAX: add src->dst (dst = a-1)
        # MAX+1..2*MAX: remove src->dst (dst = a-(MAX+1))
        self.ACTIVATE_LOCAL_BASE: int = 1 + 2 * self.max_drones
        self.DEACTIVATE_LOCAL_BASE: int = 1 + 3 * self.max_drones
        self.TOGGLE_LOCAL_N: int = 1 + 4 * self.max_drones

        # Movement encoding (per-agent)
        # 0: noop
        # 1..STEP_CELLS_N*4: (step_code in [0..STEP_CELLS_N-1], dir_code in {1..4})
        self.MOVE_N: int = 1 + int(STEP_CELLS_N) * 4

        # Flattened inter action space
        self.INTER_ACTION_N: int = self.MOVE_N * self.TOGGLE_LOCAL_N

        # --- Spaces ---
        base_obs_space = self.base_env.observation_space
        assert isinstance(base_obs_space, spaces.Box), "DroneEnv observation_space must be Box (vector)."
        base_obs_dim = int(np.prod(base_obs_space.shape))

        self._aug_obs_dim = base_obs_dim + self.max_drones + 1  # onehot + role bit

        self._anchor_obs_space = spaces.Dict(
            {
                "obs": spaces.Box(-np.inf, np.inf, shape=(self._aug_obs_dim,), dtype=np.float32),
                "state": spaces.Box(-np.inf, np.inf, shape=(base_obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(self.TOGGLE_LOCAL_N,), dtype=np.float32),
            }
        )
        self._inter_obs_space = spaces.Dict(
            {
                "obs": spaces.Box(-np.inf, np.inf, shape=(self._aug_obs_dim,), dtype=np.float32),
                "state": spaces.Box(-np.inf, np.inf, shape=(base_obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(self.INTER_ACTION_N,), dtype=np.float32),
            }
        )

        self._anchor_action_space = spaces.Discrete(self.TOGGLE_LOCAL_N)
        self._inter_action_space = spaces.Discrete(self.INTER_ACTION_N)

        # For RLlib's space introspection (define all possible agents at max size)
        self.possible_agents: List[str] = ["anchor_0", f"anchor_{self.max_drones - 1}"] + [f"inter_{i}" for i in range(1, self.max_drones - 1)]
        self.observation_space_dict: Dict[str, gym.Space] = {
            aid: (self._anchor_obs_space if aid.startswith("anchor_") else self._inter_obs_space)
            for aid in self.possible_agents
        }
        self.action_space_dict: Dict[str, gym.Space] = {
            aid: (self._anchor_action_space if aid.startswith("anchor_") else self._inter_action_space)
            for aid in self.possible_agents
        }

        # MARLlib policy mapping (group sharing):
        # 2 teams by prefix => 2 policies when share_policy="group"
        self.policy_mapping_dict = {
            "drone5g": {
                "description": "Drone5G: anchor vs intermediate drones (2 policies, CTDE).",
                "team_prefix": ("anchor_", "inter_"),
                "all_agents_one_policy": False,
                "one_agent_one_policy": False,
            }
        }

        # Runtime (per-episode) state
        self.agents: List[str] = []
        self._turn_idx: int = 0
        self._last_base_obs: np.ndarray | None = None
        # RLlib sometimes queries these
    @property
    def observation_space(self) -> gym.Space:
        return self._inter_obs_space

    @property
    def action_space(self) -> gym.Space:
        return self._inter_action_space

    def get_observation_space(self, agent_id: str) -> gym.Space:
        return self.observation_space_dict.get(agent_id, self._inter_obs_space)

    def get_action_space(self, agent_id: str) -> gym.Space:
        return self.action_space_dict.get(agent_id, self._inter_action_space)

    # --- Helpers ---

    def _active_agent_list(self) -> List[str]:
        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        n = max(2, min(n, self.max_drones))
        active_mask = getattr(self.base_env, "active_mask", None)
        if active_mask is not None:
            am = _torch_to_numpy_cpu_view(active_mask).astype(np.int32)[:n] > 0
        else:
            am = np.ones((n,), dtype=bool)
        agents = ["anchor_0", f"anchor_{n - 1}"]
        agents.extend([f"inter_{i}" for i in range(1, n - 1) if bool(am[i])])
        return agents

    def _strip_terminal_metrics_if_needed(self, info: Dict[str, Any], is_terminal: bool) -> Dict[str, Any]:
        """Avoid logging terminal metrics on every step (prevents 'dirty start' contamination)."""
        if is_terminal:
            return info
        # Remove terminal-only keys (keep normal step info)
        to_drop = [k for k in info.keys() if k.endswith("_terminal") or k.endswith("_init")]
        for k in to_drop:
            info.pop(k, None)
        return info

    def _global_toggle_mask(self) -> np.ndarray:
        """
        DroneEnv.action_masks() returns concatenated masks:
          [move_idx (max_drones), step_code (STEP_CELLS_N), dir_code (5), toggle_op (toggle_ops_n)]
        We need toggle_op mask.
        """
        full = np.asarray(self.base_env.action_masks(), dtype=np.int8)
        off = self.max_drones + int(STEP_CELLS_N) + 5
        return full[off:].astype(np.int8)

    def _local_toggle_mask(self, src: int) -> np.ndarray:
        """
        Build local toggle mask (length TOGGLE_LOCAL_N) for a fixed src drone.
        """
        mask = np.zeros((self.TOGGLE_LOCAL_N,), dtype=np.int8)
        mask[0] = 1  # noop

        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        gmask = self._global_toggle_mask()
        E = int(getattr(self.base_env, "_E", self._E))

        for dst in range(self.max_drones):
            add_a = 1 + dst
            rem_a = 1 + self.max_drones + dst

            if dst == src or dst >= n:
                continue

            pair_idx = self._edge_pair_to_idx.get((src, dst), None)
            if pair_idx is None:
                continue

            g_add = 1 + pair_idx
            g_rem = 1 + E + pair_idx

            if 0 <= g_add < len(gmask):
                mask[add_a] = int(gmask[g_add])
            if 0 <= g_rem < len(gmask):
                mask[rem_a] = int(gmask[g_rem])

        if src == 0 or src == max(0, n - 1):
            act_base = int(getattr(self, "ACTIVATE_LOCAL_BASE", 1 + 2 * self.max_drones))
            deact_base_local = int(getattr(self, "DEACTIVATE_LOCAL_BASE", 1 + 3 * self.max_drones))
            g_act_base = int(getattr(self.base_env, "_activate_op_base", 1 + 2 * E))
            g_deact_base = int(getattr(self.base_env, "_deactivate_op_base", 1 + 2 * E + self.max_drones))
            for idx in range(1, max(1, n - 1)):
                if idx >= self.max_drones:
                    continue
                if 0 <= act_base + idx < len(mask) and 0 <= g_act_base + idx < len(gmask):
                    mask[act_base + idx] = int(gmask[g_act_base + idx])
                if 0 <= deact_base_local + idx < len(mask) and 0 <= g_deact_base + idx < len(gmask):
                    mask[deact_base_local + idx] = int(gmask[g_deact_base + idx])
        return mask


    def _move_valid_mask(self, src: int) -> np.ndarray:
        """Binary mask of length MOVE_N for move_op validity of drone 'src' at current state.

        Blocks moves that would:
        - enter forbidden zone from outside (micro-step rule identical to DroneEnv.step),
        - create NEW left-links on EXISTING incident edges,
        - create NEW zone-crossing on EXISTING incident edges (if hard_block_move_cross_zone and zones enabled).
        """
        m = np.ones((self.MOVE_N,), dtype=np.int8)
        m[0] = 1  # noop always valid

        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        if not (0 <= src < n):
            m[:] = 0
            return m
        active_mask = getattr(self.base_env, "active_mask", None)
        if active_mask is not None:
            am = _torch_to_numpy_cpu_view(active_mask).astype(np.int32)[:n]
            if src >= am.shape[0] or int(am[src]) <= 0:
                m[:] = 0
                return m
        if src <= 0 or src >= (n - 1):
            m[1:] = 0
            return m

        # Current positions (ints)
        pos = np.asarray(self.base_env.drone_positions[:n, :2].detach().cpu().numpy(), dtype=np.float32)
        xi0 = int(round(float(pos[src, 0])))
        yi0 = int(round(float(pos[src, 1])))

        # Zone helpers
        zones = getattr(self.base_env, "zonas_proibidas", None)
        forb_grid = getattr(self.base_env, "_forbidden_grid", None)
        has_zone = bool(zones) and (forb_grid is not None or isinstance(zones, set))
        hard_cross = bool(getattr(self.base_env, "hard_block_move_cross_zone", True)) and has_zone

        _gmax = int(getattr(self.base_env, "grid_size", 0) or 0) - 1
        _fg = forb_grid
        if not has_zone:
            def cell_forbidden(xx: int, yy: int) -> bool:
                return False
        elif _fg is not None:
            def cell_forbidden(xx: int, yy: int) -> bool:
                if xx < 0:
                    xx = 0
                elif xx > _gmax:
                    xx = _gmax
                if yy < 0:
                    yy = 0
                elif yy > _gmax:
                    yy = _gmax
                return bool(_fg[xx, yy])
        else:
            def cell_forbidden(xx: int, yy: int) -> bool:
                if xx < 0:
                    xx = 0
                elif xx > _gmax:
                    xx = _gmax
                if yy < 0:
                    yy = 0
                elif yy > _gmax:
                    yy = _gmax
                return (int(xx), int(yy)) in zones

        grid_max = _gmax

        # Existing adjacency
        adj = np.asarray((self.base_env.adjacency_matrix[:n, :n].detach().cpu().numpy() > 0.5), dtype=np.bool_)
        out_js = np.nonzero(adj[src, :])[0].astype(np.int32, copy=False)
        in_ks = np.nonzero(adj[:, src])[0].astype(np.int32, copy=False)

        x_all = pos[:, 0]
        y_all = pos[:, 1]
        old_xi = float(x_all[src])
        old_yi = float(y_all[src])

        # Precompute OLD left-link flags on incident edges
        old_left_out = {int(j): (float(x_all[int(j)]) < old_xi) for j in out_js.tolist()}
        old_left_in = {int(k): (old_xi < float(x_all[int(k)])) for k in in_ks.tolist()}  # k->src left if x_src < x_k

        # Precompute OLD zone-cross flags on incident edges
        old_cross_out = {}
        old_cross_in = {}
        # OTIMIZAÇÃO: pré-computa, UMA vez, tudo que é invariante no loop de
        # move_op. Antes, int(round(float(x_all[j]))) e .tolist() eram refeitos
        # para cada vizinho EM CADA move_op (~75M chamadas a round() no profiling).
        # As posições dos vizinhos não mudam dentro do loop; só p_i_new varia.
        out_list = [int(j) for j in out_js.tolist()]
        in_list = [int(k) for k in in_ks.tolist()]
        # células (inteiras) dos vizinhos, calculadas uma única vez
        cell_j = {j: (int(round(float(x_all[j]))), int(round(float(y_all[j])))) for j in out_list}
        cell_k = {k: (int(round(float(x_all[k]))), int(round(float(y_all[k])))) for k in in_list}
        # coordenadas x dos vizinhos (para checagem de left-link), uma vez
        xj = {j: float(x_all[j]) for j in out_list}
        xk = {k: float(x_all[k]) for k in in_list}
        if hard_cross:
            p_i_old = (int(round(old_xi)), int(round(old_yi)))
            for j in out_list:
                old_cross_out[j] = bool(self.base_env._linha_passa_por_zona(p_i_old, cell_j[j], (src, j)))
            for k in in_list:
                old_cross_in[k] = bool(self.base_env._linha_passa_por_zona(cell_k[k], p_i_old, (k, src)))

        # Evaluate each move_op
        for move_op in range(1, self.MOVE_N):
            step_code, dir_code = _decode_move_op(move_op)
            if step_code < 0 or step_code >= int(STEP_CELLS_N):
                m[move_op] = 0
                continue

            step_cells = int(STEP_CELLS_CHOICES[step_code])
            dx, dy = _dir_to_dxdy(dir_code)
            if step_cells <= 0 or (dx == 0 and dy == 0):
                m[move_op] = 1
                continue

            # micro-step simulate (zone entry rule)
            x, y = xi0, yi0
            invalid = False
            for _ in range(step_cells):
                nx = max(0, min(grid_max, x + dx))
                ny = max(0, min(grid_max, y + dy))

                cur_in = cell_forbidden(x, y)
                nxt_in = cell_forbidden(nx, ny)
                if (not cur_in) and nxt_in:
                    invalid = True
                    break
                x, y = nx, ny

            if invalid:
                m[move_op] = 0
                continue

            new_xi = float(x)
            new_yi = float(y)

            # NEW left-links on existing incident edges => invalid
            for j in out_list:
                if (xj[j] < new_xi) and (not old_left_out.get(j, False)):
                    invalid = True
                    break
            if invalid:
                m[move_op] = 0
                continue

            for k in in_list:
                if (new_xi < xk[k]) and (not old_left_in.get(k, False)):
                    invalid = True
                    break
            if invalid:
                m[move_op] = 0
                continue

            # NEW zone-crossing on existing incident edges => invalid
            if hard_cross:
                p_i_new = (int(round(new_xi)), int(round(new_yi)))
                for j in out_list:
                    if old_cross_out.get(j, False):
                        continue
                    if bool(self.base_env._linha_passa_por_zona(p_i_new, cell_j[j], (src, j))):
                        invalid = True
                        break
                if invalid:
                    m[move_op] = 0
                    continue
                for k in in_list:
                    if old_cross_in.get(k, False):
                        continue
                    if bool(self.base_env._linha_passa_por_zona(cell_k[k], p_i_new, (k, src))):
                        invalid = True
                        break
                if invalid:
                    m[move_op] = 0
                    continue

            m[move_op] = 1

        return m

    def _decode_action_to_base(self, agent_id: str, action: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert agent action (Discrete) to DroneEnv action vector [move_idx, step_code, dir_code, toggle_op].
        """
        wrapper_info: Dict[str, Any] = {}
        if agent_id.startswith("anchor_"):
            src = 0
            move_idx = 0
            step_code, dir_code = 0, 0
            toggle_local = int(action)
        else:
            src = int(agent_id.split("_")[1])
            move_idx = src
            a = int(action)
            move_op = a // self.TOGGLE_LOCAL_N
            toggle_local = a % self.TOGGLE_LOCAL_N
            step_code, dir_code = _decode_move_op(move_op)

        # Map local toggle to global toggle_op for DroneEnv
        toggle_op = 0
        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        E = int(getattr(self.base_env, "_E", self._E))

        if toggle_local == 0:
            toggle_op = 0
        elif toggle_local == int(getattr(self, "DEACT_LOCAL_IDX", 1 + 2 * self.max_drones)):
            if agent_id.startswith("anchor_") or src <= 0 or src >= n - 1:
                wrapper_info["invalid_local_toggle"] = 1
                toggle_op = 0
            else:
                step_code, dir_code = 0, 0
                toggle_op = 1 + 2 * E + src
        elif 1 <= toggle_local <= self.max_drones:
            dst = toggle_local - 1
            if dst == src or dst >= n:
                wrapper_info["invalid_local_toggle"] = 1
                toggle_op = 0
            else:
                pair_idx = self._edge_pair_to_idx.get((src, dst), None)
                if pair_idx is None:
                    wrapper_info["invalid_local_toggle"] = 1
                    toggle_op = 0
                else:
                    toggle_op = 1 + pair_idx  # add
        else:
            dst = toggle_local - (self.max_drones + 1)
            if dst == src or dst >= n:
                wrapper_info["invalid_local_toggle"] = 1
                toggle_op = 0
            else:
                pair_idx = self._edge_pair_to_idx.get((src, dst), None)
                if pair_idx is None:
                    wrapper_info["invalid_local_toggle"] = 1
                    toggle_op = 0
                else:
                    toggle_op = 1 + E + pair_idx  # remove

        base_action = np.array([move_idx, step_code, dir_code, toggle_op], dtype=np.int64)
        return base_action, wrapper_info

    def _build_obs(self, agent_id: str, base_obs: np.ndarray) -> Dict[str, np.ndarray]:
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        onehot = np.zeros((self.max_drones,), dtype=np.float32)

        if agent_id.startswith("anchor_"):
            drone_idx = 0
            role = 1.0
        else:
            drone_idx = int(agent_id.split("_")[1])
            role = 0.0

        if 0 <= drone_idx < self.max_drones:
            onehot[drone_idx] = 1.0

        obs_aug = np.concatenate([base_obs, onehot, np.array([role], dtype=np.float32)], axis=0)

        # Action mask depends on agent
        local_toggle_mask = self._local_toggle_mask(src=drone_idx if role == 0.0 else 0).astype(np.float32)
        if agent_id.startswith("anchor_"):
            action_mask = local_toggle_mask
        else:
            move_mask = self._move_valid_mask(src=drone_idx).astype(np.float32)
            action_mask = (move_mask[:, None] * local_toggle_mask[None, :]).reshape(-1).astype(np.float32)

        return {"obs": obs_aug, "state": base_obs, "action_mask": action_mask}

    # --- Gymnasium / RLlib API ---

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        base_obs, base_info = self.base_env.reset(seed=seed, options=options)

        self.agents = self._active_agent_list()
        self._turn_idx = 0
        self._last_base_obs = np.asarray(base_obs, dtype=np.float32)

        first = self.agents[self._turn_idx]
        obs = {first: self._build_obs(first, self._last_base_obs)}
        base_info = dict(base_info or {})
        infos = {first: dict(base_info), "__common__": dict(base_info)}
        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        assert self.agents, "Call reset() before step()."

        acting = self.agents[self._turn_idx]
        a = action_dict.get(acting, 0)

        base_action, wrapper_info = self._decode_action_to_base(acting, int(a))

        _step_out = self.base_env.step(base_action)
        if isinstance(_step_out, tuple) and len(_step_out) == 5:
            base_obs, base_reward, base_terminated, base_truncated, base_info = _step_out
        elif isinstance(_step_out, tuple) and len(_step_out) == 4:
            base_obs, base_reward, base_done, base_info = _step_out
            base_terminated = bool(base_done)
            base_truncated = False
        else:
            raise ValueError(
                f"DroneEnv.step() retornou formato inesperado: "
                f"len={len(_step_out) if isinstance(_step_out, tuple) else type(_step_out).__name__}"
            )

        done = bool(base_terminated or base_truncated)
        base_info = dict(base_info or {})
        prev_agents = list(self.agents)
        new_agents = self._active_agent_list()
        if new_agents != prev_agents:
            self.agents = new_agents
            if self.agents:
                self._turn_idx = min(int(self._turn_idx), len(self.agents) - 1)
        base_info.update({f"ma/{k}": v for k, v in wrapper_info.items()})
        base_info["ma/acting_agent_idx"] = float(self._turn_idx)

        # ---------- métricas finais "limpas" (somente no terminal) ----------
        if done:
            # Garanta sempre pelo menos UMA métrica *_terminal (para o callback detectar)
            # e para evitar ficar "mudo" caso o bloco de métricas detalhadas falhe.
            try:
                rel_fallback = base_info.get(
                    "reliability_terminal",
                    base_info.get("reliability", getattr(self.base_env, "current_reliability", 0.0)),
                )
                base_info["reliability_terminal"] = float(rel_fallback)
            except Exception:
                base_info["reliability_terminal"] = 0.0

            try:
                # Matriz de adjacência e posições no estado FINAL
                adj_t = _torch_to_numpy_cpu_view(self.base_env.adjacency_matrix)
                pos_t = _torch_to_numpy_cpu_view(self.base_env.drone_positions)
                n = int(self.base_env.num_drones)

                adj = np.asarray(adj_t)[:n, :n]
                pos = np.asarray(pos_t)[:n, :]

                # drones ativos (se existir)
                active_mask = getattr(self.base_env, "active_mask", None)
                if active_mask is not None:
                    am = _torch_to_numpy_cpu_view(active_mask).astype(np.int32)[:n]
                    active = (am > 0)
                else:
                    active = np.ones(n, dtype=bool)

                ii, jj = np.nonzero(adj > 0.5)
                if ii.size > 0:
                    keep = active[ii] & active[jj]
                    ii = ii[keep]; jj = jj[keep]

                total_links = int(ii.size)
                if total_links > 0:
                    left_links = int(np.sum(pos[jj, 0] < pos[ii, 0]))
                    right_ratio = float((total_links - left_links) / max(1, total_links))
                else:
                    left_links = 0
                    right_ratio = 1.0

                # DAG (detecção de ciclo em grafo direcionado)
                def _is_dag(num_nodes: int, edges_i: np.ndarray, edges_j: np.ndarray) -> bool:
                    g = [[] for _ in range(num_nodes)]
                    for a, b in zip(edges_i.tolist(), edges_j.tolist()):
                        g[a].append(b)
                    color = [0] * num_nodes  # 0=unvisited, 1=visiting, 2=done
                    def dfs(u: int) -> bool:
                        color[u] = 1
                        for v in g[u]:
                            if color[v] == 1:
                                return False
                            if color[v] == 0 and not dfs(v):
                                return False
                        color[u] = 2
                        return True
                    for u in range(num_nodes):
                        if color[u] == 0 and not dfs(u):
                            return False
                    return True

                is_dag = bool(_is_dag(n, ii, jj))

                # path exists 0 -> n-1
                def _path_exists(num_nodes: int, edges_i: np.ndarray, edges_j: np.ndarray) -> bool:
                    g = [[] for _ in range(num_nodes)]
                    for a, b in zip(edges_i.tolist(), edges_j.tolist()):
                        g[a].append(b)
                    src, dst = 0, num_nodes - 1
                    q = [src]
                    vis = [False] * num_nodes
                    vis[src] = True
                    for u in q:
                        if u == dst:
                            return True
                        for v in g[u]:
                            if not vis[v]:
                                vis[v] = True
                                q.append(v)
                    return vis[dst]

                path_ok = bool(_path_exists(n, ii, jj))

                # violações de zona (se já vierem no info do DroneEnv, melhor)
                drones_in_zone = int(base_info.get("drones_in_zone", 0))
                links_cross_zone = int(base_info.get("links_cross_zone", 0))
                viol = int(drones_in_zone + links_cross_zone)

                # métricas finais (latência considera a rota redundante ativa com mais saltos)
                abreu_enabled_terminal = bool(getattr(self.base_env, "abreu_enabled", False))
                bootstrap_path_success = bool(path_ok and not abreu_enabled_terminal)
                rel_final = float(base_info.get("reliability_terminal", getattr(self.base_env, "current_reliability", 0.0)))
                rel_block_final = float(base_info.get("reliability_block_terminal", getattr(self.base_env, "current_reliability_block", rel_final)))
                delivery_final = float(base_info.get("delivery_ratio_terminal", getattr(self.base_env, "current_delivery_ratio", 0.0)))
                rel_hybrid_final = float(base_info.get("reliability_hybrid_terminal", getattr(self.base_env, "current_reliability_hybrid", rel_final)))
                lat_final = float(base_info.get("latency_terminal_ms", getattr(self.base_env, "current_latency_ms", 0.0)))
                tp_final = float(base_info.get("throughput_terminal", getattr(self.base_env, "current_throughput", 0.0)))
                if (not path_ok) or (not abreu_enabled_terminal):
                    # Sem caminho ou bootstrap: não tratar caminho simplificado como QoS física.
                    rel_final = 0.0
                    rel_block_final = 0.0
                    delivery_final = 0.0
                    rel_hybrid_final = 0.0
                    lat_final = 0.0
                    tp_final = 0.0
                lat_limit = float(base_info.get("success_max_latency_ms_terminal", getattr(self.base_env, "max_delay_ms", MAX_DELAY_MS_DEFAULT)))
                target_goodput, target_rel, target_delay = self.base_env._min_drone_targets()
                tp_limit = float(getattr(self.base_env, "throughput_max_mbps", THROUGHPUT_MAX_MBPS))
                qos_terminal_ok = bool(
                    path_ok
                    and (rel_block_final + 1e-12 >= float(target_rel))
                    and (delivery_final + 1e-12 >= float(getattr(self.base_env, "delivery_ratio_target", 0.999)))
                    and (tp_final + 1e-9 >= float(self.base_env._effective_goodput_target()))
                    and (tp_final <= tp_limit + 1e-9)
                    and (lat_final <= float(target_delay) + 1e-9)
                )

                # sucesso estrito no FINAL
                success_strict = bool(qos_terminal_ok)

                base_info["left_links_count_terminal"] = int(left_links)
                base_info["right_links_ratio_terminal"] = float(right_ratio)
                base_info["links_total_terminal"] = int(total_links)
                base_info["active_drones_terminal"] = int(base_info.get("active_drones", n))
                base_info["drone_excess_terminal"] = float(base_info.get("drone_excess", 0.0))
                base_info["drone_efficiency_terminal"] = float(base_info.get("drone_efficiency", 1.0))
                base_info["qos_target_met_terminal"] = float(1.0 if qos_terminal_ok else 0.0)
                base_info["pruned_drones_terminal"] = int(base_info.get("pruned_drones", 0))
                base_info["links_removed_by_pruning_terminal"] = int(base_info.get("links_removed_by_pruning", 0))
                base_info["is_dag_terminal"] = int(1 if is_dag else 0)
                base_info["path_exists_terminal"] = int(1 if path_ok else 0)
                base_info["bootstrap_path_success_terminal"] = int(1 if bootstrap_path_success else 0)
                base_info["abreu_enabled_terminal"] = int(1 if abreu_enabled_terminal else 0)
                base_info["success_terminal"] = int(1 if success_strict else 0)
                base_info["zone_violations_terminal"] = int(viol)
                base_info["reliability_terminal"] = float(rel_final)
                base_info["reliability_block_terminal"] = float(rel_block_final)
                base_info["delivery_ratio_terminal"] = float(delivery_final)
                base_info["reliability_hybrid_terminal"] = float(rel_hybrid_final)
                base_info["latency_terminal_ms"] = float(lat_final)
                base_info["throughput_terminal"] = float(tp_final)
                base_info["success_max_latency_ms_terminal"] = float(lat_limit)
            except Exception as e:
                # Não derrube o episódio por erro de métrica; mas registre um rastro pequeno
                # para facilitar debug quando os logs não aparecem.
                base_info["ma/terminal_metrics_exc"] = type(e).__name__

                # Keep terminal metrics only at the end
        base_info = self._strip_terminal_metrics_if_needed(base_info, is_terminal=done)
        if done:
            # Evita carregar arrays grandes no pipeline do RLlib/Tune/TensorBoard.
            # Mantém apenas scalars/strings para logging; snapshots completos ficam no DroneEnv (single-agent).
            for k in ("terminal_adj01", "terminal_pos_cells"):
                if k in base_info:
                    base_info.pop(k, None)

        # Team reward (cooperative)
        rewards = {aid: float(base_reward) for aid in self.agents}

        terminateds = {aid: bool(base_terminated) for aid in self.agents}
        truncateds = {aid: bool(base_truncated) for aid in self.agents}
        terminateds["__all__"] = bool(base_terminated)
        truncateds["__all__"] = bool(base_truncated)

        if done:
            # RLlib requirement: infos keys must be a subset of obs keys (plus optional '__common__').
            # To keep per-agent terminal metrics (used by TerminalMetricsCallbacks), we return a final
            # observation for every agent, along with matching infos.
            last = np.asarray(base_obs, dtype=np.float32)
            obs = {aid: self._build_obs(aid, last) for aid in self.agents}
            infos = {aid: dict(base_info) for aid in obs.keys()}
            infos["__common__"] = dict(base_info)
            self._last_base_obs = None
            return obs, rewards, terminateds, truncateds, infos

        # Next agent's turn
        self._last_base_obs = np.asarray(base_obs, dtype=np.float32)
        self._turn_idx = (self._turn_idx + 1) % len(self.agents)
        nxt = self.agents[self._turn_idx]

        obs = {nxt: self._build_obs(nxt, self._last_base_obs)}
        infos = {nxt: dict(base_info), "__common__": dict(base_info)}
        return obs, rewards, terminateds, truncateds, infos


# ============================
# MARLlib / HAPPO wrapper legado
# ============================
# Ray/RLlib não é importado aqui para evitar travamento/carga pesada nos workers
# do trainer nativo PyTorch. MultiAgentEnv já foi definido como stub acima.


class Drone5GMARLlibHAPPOEnv(MultiAgentEnv):
    """Turn-based custom env for MARLlib HAPPO using the same base DroneEnv."""

    def __init__(self, env_config=None):
        super().__init__()
        cfg = dict(env_config or {})
        self.map_name = str(cfg.pop("map_name", "Drone5G-HAPPO"))
        episode_limit = int(cfg.pop("episode_horizon", cfg.pop("episode_limit", 2000)))
        self.mask_actions = bool(int(cfg.pop("mask_actions", cfg.pop("mask_flag", 1))))
        cfg.pop("global_state_flag", None)
        cfg.pop("opp_action_in_cc", None)
        cfg.pop("agent_level_batch_update", None)

        if "zona_size" not in cfg and ("zona_w" in cfg or "zona_h" in cfg):
            zw = int(cfg.pop("zona_w", 15))
            zh = int(cfg.pop("zona_h", 15))
            cfg["zona_size"] = (zw, zh)

        accepted = set(inspect.signature(DroneEnv.__init__).parameters.keys()) - {"self"}
        dropped = {k: v for k, v in cfg.items() if k not in accepted}
        cfg = {k: v for k, v in cfg.items() if k in accepted}

        self.base_env = DroneEnv(**cfg)
        self._dropped_env_config = dropped
        try:
            self.base_env._max_episode_steps = int(episode_limit)
        except Exception:
            pass

        self.max_drones = int(getattr(self.base_env, "max_drones", cfg.get("max_drones", 50)))
        self._E = int(getattr(self.base_env, "_E", 0))
        self._edge_pairs = list(getattr(self.base_env, "_edge_pairs", []))
        self._edge_pair_to_idx = {(int(s), int(d)): i for i, (s, d) in enumerate(self._edge_pairs)}

        self.ACTIVATE_LOCAL_BASE = 1 + 2 * self.max_drones
        self.DEACTIVATE_LOCAL_BASE = 1 + 3 * self.max_drones
        self.TOGGLE_LOCAL_N = 1 + 4 * self.max_drones
        self.MOVE_N = 1 + int(STEP_CELLS_N) * 4
        self.INTER_ACTION_N = self.MOVE_N * self.TOGGLE_LOCAL_N

        base_obs_space = self.base_env.observation_space
        base_obs_dim = int(np.prod(base_obs_space.shape))
        self._aug_obs_dim = base_obs_dim + self.max_drones + 1
        self._homogeneous_obs_space = spaces.Dict({
            "obs": spaces.Box(-np.inf, np.inf, shape=(self._aug_obs_dim,), dtype=np.float32),
            "state": spaces.Box(-np.inf, np.inf, shape=(base_obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(self.INTER_ACTION_N,), dtype=np.float32),
        })
        self._homogeneous_action_space = spaces.Discrete(self.INTER_ACTION_N)

        self.possible_agents = [f"drone_{i}" for i in range(self.max_drones)]
        self.agents = list(self.possible_agents)
        self.active_agents = []
        self._turn_idx = 0
        self._last_base_obs = None

        policy_mapping = {
            "description": "Drone5G HAPPO with one policy per acting drone.",
            "team_prefix": ("drone_",),
            "all_agents_one_policy": False,
            "one_agent_one_policy": True,
        }
        self.policy_mapping_dict = {"all_scenario": policy_mapping, self.map_name: policy_mapping}
        self.observation_space = self._homogeneous_obs_space
        self.action_space = self._homogeneous_action_space

    def _current_acting_agent_ids(self):
        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        n = max(2, min(n, self.max_drones))
        active_mask = getattr(self.base_env, "active_mask", None)
        if active_mask is not None:
            am = _torch_to_numpy_cpu_view(active_mask).astype(np.int32)[:n] > 0
        else:
            am = np.ones((n,), dtype=bool)
        return [f"drone_{i}" for i in range(n) if (i == 0) or (i == n - 1) or bool(am[i])]

    def _global_toggle_mask(self):
        full = np.asarray(self.base_env.action_masks(), dtype=np.int8)
        off = self.max_drones + int(STEP_CELLS_N) + 5
        return full[off:].astype(np.int8)

    def _local_toggle_mask(self, src: int):
        mask = np.zeros((self.TOGGLE_LOCAL_N,), dtype=np.int8)
        mask[0] = 1
        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        gmask = self._global_toggle_mask()
        E = int(getattr(self.base_env, "_E", self._E))
        for dst in range(self.max_drones):
            if dst == src or dst >= n:
                continue
            pair_idx = self._edge_pair_to_idx.get((src, dst), None)
            if pair_idx is None:
                continue
            add_a = 1 + dst
            rem_a = 1 + self.max_drones + dst
            g_add = 1 + pair_idx
            g_rem = 1 + E + pair_idx
            if 0 <= g_add < len(gmask):
                mask[add_a] = int(gmask[g_add])
            if 0 <= g_rem < len(gmask):
                mask[rem_a] = int(gmask[g_rem])
        if src == 0 or src == max(0, n - 1):
            act_base = int(getattr(self, "ACTIVATE_LOCAL_BASE", 1 + 2 * self.max_drones))
            deact_base_local = int(getattr(self, "DEACTIVATE_LOCAL_BASE", 1 + 3 * self.max_drones))
            g_act_base = int(getattr(self.base_env, "_activate_op_base", 1 + 2 * E))
            g_deact_base = int(getattr(self.base_env, "_deactivate_op_base", 1 + 2 * E + self.max_drones))
            for idx in range(1, max(1, n - 1)):
                if idx >= self.max_drones:
                    continue
                if 0 <= act_base + idx < len(mask) and 0 <= g_act_base + idx < len(gmask):
                    mask[act_base + idx] = int(gmask[g_act_base + idx])
                if 0 <= deact_base_local + idx < len(mask) and 0 <= g_deact_base + idx < len(gmask):
                    mask[deact_base_local + idx] = int(gmask[g_deact_base + idx])
        return mask

    @staticmethod
    def _dir_to_dxdy(dir_code: int):
        return {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}.get(dir_code, (0, 0))

    def _move_valid_mask(self, src: int):
        m = np.ones((self.MOVE_N,), dtype=np.int8)
        m[0] = 1
        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        if not (0 <= src < n):
            m[:] = 0
            return m
        active_mask = getattr(self.base_env, "active_mask", None)
        if active_mask is not None:
            am = _torch_to_numpy_cpu_view(active_mask).astype(np.int32)[:n]
            if src >= am.shape[0] or int(am[src]) <= 0:
                m[:] = 0
                return m
        if src <= 0 or src >= (n - 1):
            m[1:] = 0
            return m

        pos = np.asarray(self.base_env.drone_positions[:n, :2].detach().cpu().numpy(), dtype=np.float32)
        xi0 = int(round(float(pos[src, 0])))
        yi0 = int(round(float(pos[src, 1])))
        zones = getattr(self.base_env, "zonas_proibidas", None)
        forb_grid = getattr(self.base_env, "_forbidden_grid", None)
        has_zone = bool(zones) and (forb_grid is not None or isinstance(zones, set))
        hard_cross = bool(getattr(self.base_env, "hard_block_move_cross_zone", True)) and has_zone

        # cell_forbidden é chamado ~16M vezes; limites pré-computados fora do
        # caminho por-chamada (evita getattr/recomputo de grid_size).
        _gmax = int(getattr(self.base_env, "grid_size", 0) or 0) - 1
        _fg = forb_grid
        if not has_zone:
            def cell_forbidden(xx: int, yy: int) -> bool:
                return False
        elif _fg is not None:
            def cell_forbidden(xx: int, yy: int) -> bool:
                if xx < 0:
                    xx = 0
                elif xx > _gmax:
                    xx = _gmax
                if yy < 0:
                    yy = 0
                elif yy > _gmax:
                    yy = _gmax
                return bool(_fg[xx, yy])
        else:
            def cell_forbidden(xx: int, yy: int) -> bool:
                if xx < 0:
                    xx = 0
                elif xx > _gmax:
                    xx = _gmax
                if yy < 0:
                    yy = 0
                elif yy > _gmax:
                    yy = _gmax
                return (int(xx), int(yy)) in zones

        grid_max = _gmax
        adj = np.asarray((self.base_env.adjacency_matrix[:n, :n].detach().cpu().numpy() > 0.5), dtype=np.bool_)
        out_js = np.nonzero(adj[src, :])[0].astype(np.int32, copy=False)
        in_ks = np.nonzero(adj[:, src])[0].astype(np.int32, copy=False)
        x_all = pos[:, 0]

        for move_op in range(1, self.MOVE_N):
            k = move_op - 1
            step_code = k // 4
            dir_code = 1 + (k % 4)
            step_cells = int(step_code) + 1
            dx, dy = self._dir_to_dxdy(dir_code)
            xi = xi0
            yi = yi0
            valid = True
            for _ in range(step_cells):
                xi = max(0, min(grid_max, xi + dx))
                yi = max(0, min(grid_max, yi + dy))
                if cell_forbidden(xi, yi):
                    valid = False
                    break
            if valid and bool(getattr(self.base_env, "hard_block_move_left_links", True)):
                new_x = float(xi)
                if out_js.size > 0 and np.any(x_all[out_js] < new_x):
                    valid = False
                if valid and in_ks.size > 0 and np.any(new_x < x_all[in_ks]):
                    valid = False
            if valid and hard_cross:
                try:
                    new_xy = np.asarray([float(xi), float(yi)], dtype=np.float32)
                    pos2 = pos.copy()
                    pos2[src, 0] = new_xy[0]
                    pos2[src, 1] = new_xy[1]
                    for j in out_js.tolist():
                        if bool(self.base_env._linha_passa_por_zona(new_xy, pos2[j, :2])):
                            valid = False
                            break
                    if valid:
                        for k_in in in_ks.tolist():
                            if bool(self.base_env._linha_passa_por_zona(pos2[k_in, :2], new_xy)):
                                valid = False
                                break
                except Exception:
                    pass
            m[move_op] = 1 if valid else 0
        return m

    def _decode_action_to_base(self, agent_id: str, action: int):
        wrapper_info = {}
        n = int(getattr(self.base_env, "num_drones", self.max_drones))
        E = int(getattr(self.base_env, "_E", self._E))
        src = int(agent_id.split("_")[1])
        a = int(action)
        if a < 0 or a >= self.INTER_ACTION_N:
            wrapper_info["invalid_action"] = 1
            a = 0
        move_local = a // self.TOGGLE_LOCAL_N
        toggle_local = a % self.TOGGLE_LOCAL_N
        if src == 0 or src == n - 1:
            if move_local != 0:
                wrapper_info["anchor_move_forced_noop"] = 1
            move_idx = int(src)
            step_code = dir_code = 0
        else:
            move_idx = int(src)
            if move_local == 0:
                step_code = dir_code = 0
            else:
                step_code = int((move_local - 1) // 4)
                dir_code = 1 + int((move_local - 1) % 4)
        toggle_op = 0
        if toggle_local == 0:
            toggle_op = 0
        elif int(getattr(self, "ACTIVATE_LOCAL_BASE", 1 + 2 * self.max_drones)) <= toggle_local < int(getattr(self, "ACTIVATE_LOCAL_BASE", 1 + 2 * self.max_drones)) + self.max_drones:
            dst = toggle_local - int(getattr(self, "ACTIVATE_LOCAL_BASE", 1 + 2 * self.max_drones))
            if not (src == 0 or src == n - 1) or dst <= 0 or dst >= n - 1:
                wrapper_info["invalid_local_toggle"] = 1
            else:
                step_code = dir_code = 0
                toggle_op = int(getattr(self.base_env, "_activate_op_base", 1 + 2 * E)) + int(dst)
        elif int(getattr(self, "DEACTIVATE_LOCAL_BASE", 1 + 3 * self.max_drones)) <= toggle_local < int(getattr(self, "DEACTIVATE_LOCAL_BASE", 1 + 3 * self.max_drones)) + self.max_drones:
            dst = toggle_local - int(getattr(self, "DEACTIVATE_LOCAL_BASE", 1 + 3 * self.max_drones))
            if not (src == 0 or src == n - 1) or dst <= 0 or dst >= n - 1:
                wrapper_info["invalid_local_toggle"] = 1
            else:
                step_code = dir_code = 0
                toggle_op = int(getattr(self.base_env, "_deactivate_op_base", 1 + 2 * E + self.max_drones)) + int(dst)
        elif 1 <= toggle_local <= self.max_drones:
            dst = toggle_local - 1
            pair_idx = None if (dst == src or dst >= n) else self._edge_pair_to_idx.get((src, dst), None)
            if pair_idx is None:
                wrapper_info["invalid_local_toggle"] = 1
            else:
                toggle_op = 1 + pair_idx
        else:
            dst = toggle_local - (self.max_drones + 1)
            pair_idx = None if (dst == src or dst >= n) else self._edge_pair_to_idx.get((src, dst), None)
            if pair_idx is None:
                wrapper_info["invalid_local_toggle"] = 1
            else:
                toggle_op = 1 + E + pair_idx
        base_action = np.array([move_idx, step_code, dir_code, toggle_op], dtype=np.int64)
        return base_action, wrapper_info

    def _build_obs(self, agent_id: str, base_obs: np.ndarray):
        base_obs = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        onehot = np.zeros((self.max_drones,), dtype=np.float32)
        drone_idx = int(agent_id.split("_")[1])
        is_anchor = 1.0 if (drone_idx == 0 or drone_idx == self.max_drones - 1) else 0.0
        if 0 <= drone_idx < self.max_drones:
            onehot[drone_idx] = 1.0
        obs_aug = np.concatenate([base_obs, onehot, np.array([is_anchor], dtype=np.float32)], axis=0)
        local_toggle_mask = self._local_toggle_mask(src=drone_idx).astype(np.float32)
        move_mask = self._move_valid_mask(src=drone_idx).astype(np.float32)
        action_mask = (move_mask[:, None] * local_toggle_mask[None, :]).reshape(-1).astype(np.float32)
        return {"obs": obs_aug, "state": base_obs, "action_mask": action_mask}

    def reset(self, seed=None):
        out = self.base_env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            base_obs, _base_info = out
        else:
            base_obs = out
        self.active_agents = self._current_acting_agent_ids()
        self._turn_idx = 0
        self._last_base_obs = np.asarray(base_obs, dtype=np.float32)
        first = self.active_agents[self._turn_idx]
        return {first: self._build_obs(first, self._last_base_obs)}

    def step(self, action_dict):
        if not self.active_agents:
            raise RuntimeError("Call reset() before step().")
        acting = self.active_agents[self._turn_idx]
        a = action_dict.get(acting, 0)
        base_action, wrapper_info = self._decode_action_to_base(acting, int(a))
        step_out = self.base_env.step(base_action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            base_obs, base_reward, base_done, base_trunc, base_info = step_out
            done = bool(base_done or base_trunc)
        else:
            base_obs, base_reward, base_done, base_info = step_out
            done = bool(base_done)
        base_info = dict(base_info or {})
        base_info.update({f"ma/{k}": v for k, v in wrapper_info.items()})
        base_info["ma/acting_agent"] = acting
        self.active_agents = self._current_acting_agent_ids()
        rewards = {aid: float(base_reward) for aid in self.active_agents}
        dones = {aid: bool(done) for aid in self.active_agents}
        dones["__all__"] = bool(done)
        if done:
            last = np.asarray(base_obs, dtype=np.float32)
            obs = {aid: self._build_obs(aid, last) for aid in self.active_agents}
            infos = {aid: dict(base_info) for aid in self.active_agents}
            self._last_base_obs = None
            return obs, rewards, dones, infos
        self._last_base_obs = np.asarray(base_obs, dtype=np.float32)
        self._turn_idx = (self._turn_idx + 1) % len(self.active_agents) if self.active_agents else 0
        nxt = self.active_agents[self._turn_idx]
        obs = {nxt: self._build_obs(nxt, self._last_base_obs)}
        infos = {nxt: dict(base_info)}
        return obs, rewards, dones, infos

    def close(self):
        try:
            return self.base_env.close()
        except Exception:
            return None

    def render(self, mode=None):
        try:
            return self.base_env.render(mode=mode)
        except Exception:
            return None

    def get_env_info(self):
        return {
            "space_obs": self._homogeneous_obs_space,
            "space_act": self._homogeneous_action_space,
            "num_agents": len(self.possible_agents),
            "episode_limit": int(getattr(self.base_env, "_max_episode_steps", 2000)),
            "policy_mapping_info": self.policy_mapping_dict,
        }


__all__ = [name for name in globals() if name in {"DroneEnv", "Drone5GMARLlibHAPPOEnv", "STEP_CELLS_N", "_torch_to_numpy_cpu_view"}]


# =========================
# Standard Gym env for native PyTorch HAPPO training
# =========================

class Drone5GHAPPOPyEnv(gym.Env):
    """Standard Gymnasium-style env for native PyTorch HAPPO.

    Each env step corresponds to the action of the *current* acting drone in the
    turn-based DroneEnv wrapper. The observation exposes all possible agents
    simultaneously, including:
      - per-agent local observation vector,
      - shared state,
      - per-agent action mask,
      - alive mask (agent exists in current topology),
      - turn mask (agent that acts on this env step).

    This makes the environment easy to use with vectorized on-policy trainers
    while keeping the original base environment logic intact.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config=None):
        super().__init__()
        self._env_config = dict(env_config or {})
        self._tb = Drone5GMARLlibHAPPOEnv(env_config=self._env_config)
        self.base_env = self._tb.base_env
        self._configured_episode_horizon = int(self._env_config.get("episode_horizon", self._env_config.get("episode_limit", getattr(self.base_env, "_max_episode_steps", 2000))))
        try:
            self.base_env._max_episode_steps = int(self._configured_episode_horizon)
        except Exception:
            pass
        self.max_agents = len(self._tb.possible_agents)
        self.act_dim = int(self._tb.INTER_ACTION_N)
        self.obs_dim = int(self._tb._aug_obs_dim)
        self.state_dim = int(np.prod(self.base_env.observation_space.shape))

        self.action_space = spaces.MultiDiscrete(np.full((self.max_agents,), self.act_dim, dtype=np.int64))
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(-np.inf, np.inf, shape=(self.max_agents, self.obs_dim), dtype=np.float32),
            "state": spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(self.max_agents, self.act_dim), dtype=np.float32),
            "alive_mask": spaces.Box(0.0, 1.0, shape=(self.max_agents, 1), dtype=np.float32),
            "turn_mask": spaces.Box(0.0, 1.0, shape=(self.max_agents, 1), dtype=np.float32),
        })
        self._last_info = {}

    def _agent_idx(self, agent_id: str) -> int:
        return int(str(agent_id).split('_')[1])

    def _assemble_obs(self, base_state: np.ndarray | None = None):
        if base_state is None:
            if self._tb._last_base_obs is not None:
                base_state = np.asarray(self._tb._last_base_obs, dtype=np.float32)
            else:
                base_state = np.zeros((self.state_dim,), dtype=np.float32)
        else:
            base_state = np.asarray(base_state, dtype=np.float32).reshape(-1)

        obs = np.zeros((self.max_agents, self.obs_dim), dtype=np.float32)
        action_mask = np.zeros((self.max_agents, self.act_dim), dtype=np.float32)
        alive_mask = np.zeros((self.max_agents, 1), dtype=np.float32)
        turn_mask = np.zeros((self.max_agents, 1), dtype=np.float32)

        current_agents = list(getattr(self._tb, 'active_agents', []) or self._tb._current_acting_agent_ids())
        current_set = set(current_agents)
        acting_id = None
        if current_agents:
            acting_id = current_agents[int(getattr(self._tb, '_turn_idx', 0)) % len(current_agents)]

        for aid in self._tb.possible_agents:
            idx = self._agent_idx(aid)
            built = self._tb._build_obs(aid, base_state)
            obs[idx] = np.asarray(built['obs'], dtype=np.float32)
            action_mask[idx] = np.asarray(built['action_mask'], dtype=np.float32)
            if aid in current_set:
                alive_mask[idx, 0] = 1.0
            else:
                # keep a valid no-op action for inactive slots
                action_mask[idx, :] = 0.0
                action_mask[idx, 0] = 1.0
            if aid == acting_id:
                turn_mask[idx, 0] = 1.0

        return {
            'obs': obs,
            'state': base_state.astype(np.float32),
            'action_mask': action_mask,
            'alive_mask': alive_mask,
            'turn_mask': turn_mask,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        out = self._tb.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            _obs_dict, infos = out
            info = dict(infos.get('__common__', {})) if isinstance(infos, dict) else {}
        else:
            info = {}
        info.setdefault("episode_step", int(getattr(self.base_env, "_episode_steps", 0)))
        info.setdefault("episode_horizon", int(getattr(self.base_env, "_max_episode_steps", self._configured_episode_horizon)))
        full_obs = self._assemble_obs()
        self._last_info = dict(info)
        return full_obs, dict(info)

    def step(self, action):
        if not getattr(self._tb, 'active_agents', None):
            raise RuntimeError('Call reset() before step().')
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        acting_id = self._tb.active_agents[int(self._tb._turn_idx) % len(self._tb.active_agents)]
        acting_idx = self._agent_idx(acting_id)
        act = int(action[acting_idx]) if 0 <= acting_idx < action.shape[0] else 0

        _tb_step_out = self._tb.step({acting_id: act})
        if isinstance(_tb_step_out, tuple) and len(_tb_step_out) == 5:
            obs_dict, rewards, terminateds, truncateds, infos = _tb_step_out
        elif isinstance(_tb_step_out, tuple) and len(_tb_step_out) == 4:
            obs_dict, rewards, dones, infos = _tb_step_out
            if isinstance(dones, dict):
                terminateds = dict(dones)
                truncateds = {k: False for k in dones.keys()}
                if "__all__" in dones:
                    truncateds["__all__"] = False
            else:
                d = bool(dones)
                terminateds = {acting_id: d, "__all__": d}
                truncateds = {acting_id: False, "__all__": False}
        else:
            raise ValueError(f"Wrapper _tb.step() retornou formato inesperado: {type(_tb_step_out).__name__}")
        terminated = bool(terminateds.get('__all__', False))
        truncated = bool(truncateds.get('__all__', False))
        base_steps = int(getattr(self.base_env, "_episode_steps", 0))
        base_horizon = int(getattr(self.base_env, "_max_episode_steps", self._configured_episode_horizon))
        timeout_fallback = (base_horizon > 0) and (base_steps >= base_horizon)
        if timeout_fallback and not (terminated or truncated):
            truncated = True
            truncateds = {k: bool(v) for k, v in truncateds.items()}
            truncateds[acting_id] = True
            truncateds["__all__"] = True
        done = bool(terminated or truncated)

        info = {}
        if isinstance(infos, dict):
            if '__common__' in infos and isinstance(infos['__common__'], dict):
                info = dict(infos['__common__'])
            elif infos:
                first_key = next(iter(infos.keys()))
                info = dict(infos[first_key]) if isinstance(infos[first_key], dict) else {}
        info.setdefault("episode_step", int(base_steps))
        info.setdefault("episode_horizon", int(base_horizon))
        if timeout_fallback:
            info.setdefault("done_reason", "timeout_fallback")

        reward = 0.0
        if isinstance(rewards, dict) and rewards:
            try:
                reward = float(next(iter(rewards.values())))
            except Exception:
                reward = 0.0
        else:
            reward = float(rewards or 0.0)

        if done:
            base_state = None
            if isinstance(obs_dict, dict) and obs_dict:
                first_val = next(iter(obs_dict.values()))
                if isinstance(first_val, dict) and 'state' in first_val:
                    base_state = np.asarray(first_val['state'], dtype=np.float32)
            full_obs = self._assemble_obs(base_state=base_state)
        else:
            full_obs = self._assemble_obs()

        self._last_info = dict(info)
        return full_obs, reward, terminated, truncated, dict(info)

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


__all__ = sorted(set(list(globals().get('__all__', [])) + ['Drone5GHAPPOPyEnv']))


# ============================================================================
# NOTA DE TREINO (correções de convergência)
# ============================================================================
# As principais mudanças nesta versão:
#
# 1) _evaluate_graph_metrics(use_abreu=False) agora modela MULTICAMINHOS com
#    redundância por duplicação de pacotes:
#        reliability = 1 - Π_k (1 - rel_k)
#    com latência da cópia redundante mais lenta e throughput pelo gargalo.
#    Antes a recompensa por step usava UM caminho BFS e throughput constante,
#    portanto NÃO recompensava a redundância exigida no teste terminal de Abreu.
#    Esse descasamento era a causa principal de não-convergência.
#
# 2) FIT_WEIGHTS_DEFAULT rebalanceado para faixa compacta (~[-3,+5]); a
#    PRIORIDADE vem da estrutura lexicográfica, não da magnitude bruta.
#
# 3) Custo de drone pré-QoS só é aplicado quando a confiabilidade já tem folga
#    (>=1.05x do alvo), evitando penalizar a redundância antes de ela render
#    recompensa (conflito de gradiente eliminado).
#
# RECOMENDAÇÕES NO TRAINER (5GMLMA_MA.py), se ainda não estiverem ativas:
#   - VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
#   - gamma >= 0.99, gae_lambda ~0.95
#   - ent_coef pequeno mas > 0 (ex.: 0.005) para exploração de topologias
#   - manter o currículo de start.sh (delivery_ratio_target 0.90 -> 0.98),
#     pois 0.999 só é atingível com redundância suficiente.
# ============================================================================
