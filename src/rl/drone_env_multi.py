#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""drone_env_multi.py (standalone)

SELF-CONTAINED:
- Embeds the single-agent DroneEnv in this file.
- Provides the multi-agent wrapper (Drone5GMAEnv) with real action masking (toggle + movement)
  and cleanup of *_terminal metrics on non-terminal steps.

No dependency on an external drone_env.py file."""


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
from scipy.sparse.csgraph import floyd_warshall

try:
    from calculaProbLink_otimizado import calcula_confiabilidade_iterativa, prob_link_literature, RadioParams
except Exception:
    try:
        from calculaProbLinkOtimizado import calcula_confiabilidade_iterativa, prob_link_literature, RadioParams
    except Exception:
        from calculaProbLink import calcula_confiabilidade_iterativa, prob_link_literature, RadioParams

# ---- Safe RadioParams construction (handles dataclass signatures) ----
_RADIO_DEFAULTS = dict(
    pt_dbm=19.54242509439325,
    gt_db=2.0,
    gr_db=2.0,
    freq_hz=2.4e9,
    n=2.3,
    d0_m=1.0,
    pmin_dbm=-83.0,
    sigma_db=4.0,
)

def _make_radio_params():
    """Create RadioParams robustly across different implementations."""
    try:
        return RadioParams()
    except Exception:
        pass
    try:
        return RadioParams(**_RADIO_DEFAULTS)
    except Exception:
        pass
    # last resort: allocate object and set attrs
    try:
        obj = RadioParams.__new__(RadioParams)
        for k, v in _RADIO_DEFAULTS.items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj
    except Exception:
        return None

def _torch_to_numpy_cpu_view(t: "torch.Tensor") -> np.ndarray:
    """Convert a Torch tensor to NumPy with the lowest possible overhead.

- CPU: returns a view via .detach().numpy() (no copy).
- Non-CPU: uses .detach().cpu().numpy() (unavoidable copy).

Note: this project runs on CPU by default (DEVICE=cpu), so this removes the
cost of .cpu() and reduces overhead in hot paths."""
    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    if t.device.type == "cpu":
        return t.detach().numpy()
    return t.detach().cpu().numpy()

# =========================
# Constants/Parameters
# =========================

DEVICE = torch.device("cpu")

MAX_DRONES = 30
GRID_SIZE_DEFAULT = 100
SQUARE_SIZE_M_DEFAULT = 20.0

# Movement: discrete code (0..3) -> step in cells {1,2,3,4}.
# Note: dir_code==0 means 'do not move' (the step is ignored).
STEP_CELLS_CHOICES = [1, 2, 3, 4, 8, 16]
STEP_CELLS_N = len(STEP_CELLS_CHOICES)

# Targets/thresholds
RELIABILITY_TARGET = 0.99999
RELIABILITY_BASE = 0.4
# Gate (nearly hard) to unlock secondary terms:
# - 0.0 at RELIABILITY_UNLOCK_START
# - 1.0 at RELIABILITY_UNLOCK_END (>=0.9999 unlocks everything)
RELIABILITY_UNLOCK_START = 0.0
RELIABILITY_UNLOCK_END = 0.99999
THROUGHPUT_ORIGEM_MBPS = 100.0
THROUGHPUT_MAX_MBPS = 9600.0
HOPS_TARGET = 10
# ----------------------------
# Use only FIT_WEIGHTS_DEFAULT to adjust fitness weights.

# Simplified "physical" latency
MAX_LATENCY_MS = 100.0
LATENCY_PER_HOP_MS = 1.0
SPEED_M_PER_MS = 0.2  # 0.2 m/ms = 200 m/s (for cost only)

# =========================
# EFFECTIVE fitness weights (what actually contributes to the fitness)
# Edit here (or override via DroneEnv(fit_weights={...}))
#
# Warning: several weights at the top of the file are historical/legacy.
# The fitness function uses the weights below.
# =========================
FIT_WEIGHTS_DEFAULT = {
    # Main reliability: rel_score = REL_MAIN*(2*rel_log - 1)
    # where rel_log = clamp((log10(reliability)+9)/9, 0..1).
    # Therefore: rel_score ∈ [-REL_MAIN, +REL_MAIN]
    "REL_MAIN": 25_000.0,
    "REL_STEP": 0.0,  # 0 => large reliability term only at TERMINAL (without dense per-step shaping)
    "TERM_REL": 20_000.0,  # terminal bonus/penalty based on final reliability (large scale to reflect the test)
    "TERM_SUCCESS_BONUS": 3_000.0,  # terminal bonus when reaching RELIABILITY_TARGET (0.99999) with a clean zone
    "TERM_ZONE_PEN": 10_000.0,  # terminal penalty for zone violations (per drone inside + per crossing link)

    # Hard penalty if no path exists 0->N-1
    "PATH_MISSING": 3_000.0,

    # Hard penalty for zone violations (multiplies total_violacoes = drones_in_zone + links_cross_zone)
    "ZONE_HARD": 3_000.0,

    # zone (penalties proportional to each occurrence)
    # (keeps the previous effective value: 20*100=2000 and 20*100.5=2010)
    "ZONE_DRONE_COEF": 2000.0,  # per drone inside the zone
    "ZONE_LINK_COEF": 2010.0,  # per link crossing the zone

    # Average connection quality (small coefficient)
    # "CONN_QUALITY_COEF": 0.0,
    "CONN_QUALITY_COEF": 400.0,

    # Hops (with gate): hops_score = HOPS_WEIGHT*hops_gate*hop_diff
    "HOPS_WEIGHT": 50.0,

    # Path layout (when path_exists)
    "LAYOUT_ALIGN_W": 200.0,
    "LAYOUT_STRAIGHT_W": 160.0,
    "LAYOUT_REDUND_W": 450.0,

    # Spread mixture: layout_terms = LAYOUT_MIX*(nn_reward + uniform_reward) + layout_score
    "LAYOUT_MIX": 2500.0,

    # 1D uniformity: number of bins along the segment 0->N-1
    "UNIFORM1D_BINS": 8.0,
    # "UNIFORM1D_BINS": 0.0,

    # Pulls intermediate drones toward the line (drone0 -> droneN-1)
    # line_align_term = LINE_ALIGN_W * gate_zone * (1 - mean_perp_dist_norm)
    # where mean_perp_dist_norm ∈ [0,1]
    # "LINE_ALIGN_W": 0.0,
    "LINE_ALIGN_W": 800.0,

    # Penalty for isolated intermediate nodes on the path (per node)
    "ISOLATED_NODE_PENALTY": 4000.0,

    # Penalty for nodes (except node 0) that are NOT reachable from drone 0 (directly or indirectly)
    "UNREACHABLE_FROM_FIRST_PENALTY": 0.0,  # (LEGACY) penalizes by count; prefer REACH_FROM_FIRST_W
    "REACH_FROM_FIRST_W": 6500.0,  # penalizes (1 - reach_from_first_frac) ∈ [0,1]

    # Terms computed in the environment, but DISABLED in the fitness by default (weight 0.0).
    # To enable, set values > 0 via fit_weights.
    "LEFT_LINK_COEF": 1000.0,  # penalty for a "backward" link (x_dst < x_src)

    # Strong penalty if there is a cycle (loop) in the link graph
    "CYCLE_PENALTY": 10_000.0,
    "THROUGHPUT_PENALTY_MAX": 500.0,  # maximum penalty when throughput gets too close to the maximum (ratio >= 0.85)
    "DISJOINT_PATH_BONUS": 6000.0,  # bonus per additional disjoint path (k-1)
    "DISJOINT_PARTIAL_FACTOR": 0.5,  # fraction of the bonus for PARTIAL parallel routes (>=2)
    "DISJOINT_PARTIAL_MIN_PROGRESS": 0.5,
    # minimum progress (0..1) along the segment 0->N-1 to count a partial target
    "DISJOINT_MIN_P": 0.5,  # minimum link probability to count toward the disjoint-path bonus (0 disables)
    "DIRECT_0N1_MIN_P": 0.7,  # minimum probability to allow a direct link 0<->N-1 (0 disables)

    "ZONE_FIX_LINK_BONUS": 600.0,  # bonus for a link that STOPS crossing a zone (delta per step)
    "ZONE_FIX_DRONE_BONUS": 600.0, # bonus for a drone that LEAVES the zone (delta per step)

    "ZONE_CLEAN_ONCE_BONUS": 2500.0, # one-time bonus when zone violations reach zero in the episode
    "INVALID_ACTION_PENALTY": 10.0,  # per-step penalty when a link/move action is invalid
    # ---- Bottleneck shaping (improves the worst link on the path 0->N-1) ----
    "BOTTLENECK_W": 3000.0,  # bottleneck-term weight (scale ~ thousands)
    "BOTTLENECK_TARGET_P": 0.70,  # minimum-probability target on the path (>= this yields a bonus)
    # ---- Disjoint gate (activates parallel-route bonuses early, based on useful connectivity) ----
    "DISJOINT_GATE_RFF_START": 0.85,  # starts activating when reach_from_first_frac >= this value
    "DISJOINT_GATE_RF_START": 0.85,  # starts activating when reach_frac >= this value
}


class DroneEnv(gym.Env):
    """Drone environment for PPO (Gymnasium). The observation contains:
- normalized (x, y) positions for each drone,
- a flattened adjacency matrix,
- an active-drone mask (always 1.0 here for N drones).

Actions (MultiDiscrete):
[move_idx (0..MAX_DRONES-1), step_code (0..3)->{1,2,3,4}, dir_code (0..4), toggle_op (0..(1+2*E-1))]"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            grid_size=GRID_SIZE_DEFAULT,
            num_drones=13,
            max_out_connections=10,
            square_size_m=SQUARE_SIZE_M_DEFAULT,
            zonas_proibidas=None,
            p_sem_zona=0.0,
            variable_drones=True,
            min_drones=None,
            max_drones=None,
            eval_mode=False,
            # parameterized zones
            zonas_count=None,
            zona_size=None,  # (w, h)
            zona_single_fixed=False,
            # fitness weights (effective) / detailed fitness logging
            fit_weights=None,
            log_fit_terms=False,
            # ---- Extra observation (optional) ----
            obs_zone_map=0,          # adds a map (downsample) of forbidden zones to the observation
            zone_map_scale=4,        # downsampling factor (ex.: 4 => 100x100 -> 25x25)
            obs_edge_cross=0,        # adds a matrix NxN indicating whether edge (i,j) crosses a zone
            # ---- Hard masks ----
            hard_mask_zone_edges=1,  # blocks ADD operations for edges crossing zones (REMOVE remains allowed)
            hard_mask_left_links=1,  # blocks ADD operations for leftward links (x_dst < x_src)
            hard_mask_cycles=1,      # blocks ADD operations for links that would create cycles (without loops)
            hard_block_move_cross_zone=1,  # HARD: reverts movement if it INCREASES zone-crossing links
            # ---- Curriculum: "dirty reset" (starts dirty and ends 100% dirty) ----
            dirty_reset_prob_start=0.20,
            dirty_reset_prob_end=1.00,
            dirty_force_drone_in_zone_prob_start=0.20,
            dirty_force_drone_in_zone_prob_end=1.00,
            dirty_anneal_episodes=300,
    ):
        super().__init__()

        # ---------- basic parameters ----------
        self.grid_size = int(grid_size)
        self.square_size_m = float(square_size_m)
        self._radio_params = _make_radio_params()  # used to filter links in the disjoint-path bonus

        # Local cache of p(link) by distance (same semantics; avoids recomputing erf/log10 repeatedly)
        self._prob_cache = {}
        self.max_out_connections = int(max_out_connections)

        self.variable_drones = bool(variable_drones)
        self.min_drones = int(min_drones) if min_drones is not None else None
        self.max_drones = int(max_drones) if max_drones is not None else None
        self.num_drones = int(num_drones)

        # Defines max_drones effective value used to size a observation_space
        if self.max_drones is None:
            self.max_drones = self.num_drones
        # Ensures limits [1, MAX_DRONES]
        self.max_drones = max(1, min(int(self.max_drones), MAX_DRONES))
        # Ensures that num_drones does not exceed max_drones
        if self.num_drones > self.max_drones:
            self.num_drones = self.max_drones

        self.eval_mode = bool(eval_mode)

        # zones
        self.zonas_proibidas_param = zonas_proibidas
        self.zonas_proibidas = set()
        # cache: boolean grid for fast lookup of forbidden zones (x,y)
        self._forbidden_grid = None
        self._zone_sig = 0  # deterministic signature of forbidden grid
        self.p_sem_zona = float(p_sem_zona)
        self.zonas_count = int(zonas_count) if zonas_count is not None else None
        self.zona_size = tuple(zona_size) if (zona_size and zona_size[0] and zona_size[1]) else None
        self.zona_single_fixed = bool(zona_single_fixed)

        # ---- Extra observation / hard masks ----
        self.obs_zone_map = bool(obs_zone_map)
        self.zone_map_scale = max(1, int(zone_map_scale))
        self.obs_edge_cross = bool(obs_edge_cross)
        self.hard_mask_zone_edges = bool(hard_mask_zone_edges)
        self.hard_mask_left_links = bool(hard_mask_left_links)
        self.hard_mask_cycles = bool(hard_mask_cycles)
        self.hard_block_move_cross_zone = bool(hard_block_move_cross_zone)

        # ---- Curriculum: dirty reset ----
        self.dirty_reset_prob_start = float(dirty_reset_prob_start)
        self.dirty_reset_prob_end = float(dirty_reset_prob_end)
        self.dirty_force_drone_in_zone_prob_start = float(dirty_force_drone_in_zone_prob_start)
        self.dirty_force_drone_in_zone_prob_end = float(dirty_force_drone_in_zone_prob_end)
        self.dirty_anneal_episodes = max(1, int(dirty_anneal_episodes))
        self._reset_count = 0
        self._dirty_episode = False
        self._force_drone_in_zone = False

        # cache: zone_map (downsample) and edge_cross (matrix NxN)
        self._zone_map_down = None
        self._zone_map_down_sig = None
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None


        # current metrics (for external logging)
        self.current_reliability = 0.0
        self.current_latency_penalty = 0.0
        self.current_throughput = 0.0
        # ---------- spaces ----------
        # action: 4 dimensions (moves 1 drone per step + flattened link operation)
        # [move_idx, step_code, dir_code, toggle_op]
        #
        # toggle_op:
        # 0 -> does not change links
        # 1..and -> ADD (src,dst) for each directed pair src!=dst (fixed order at self._edge_pairs)
        # and+1..2E -> REMOVE (src,dst)
        #
        # This enables action masking REAL (without invalid combinations src/dst).
        self._edge_pairs = [(i, j) for i in range(self.max_drones) for j in range(self.max_drones) if i != j]
        self._edge_pair_to_idx = {p: k for k, p in enumerate(self._edge_pairs)}
        self._E = len(self._edge_pairs)
        self._toggle_ops_n = 1 + 2 * self._E

        # Precomputed vectors for vectorized indexing in action_masks()
        # (avoids a Python loop per pair (src,dst) at each step)
        self._edge_src = np.fromiter((p[0] for p in self._edge_pairs), dtype=np.int16, count=self._E)
        self._edge_dst = np.fromiter((p[1] for p in self._edge_pairs), dtype=np.int16, count=self._E)

        self.action_space = spaces.MultiDiscrete(
            np.array([self.max_drones, STEP_CELLS_N, 5, self._toggle_ops_n], dtype=np.int64)
        )

        # simple cache for action_masks (avoids recomputing when the state does not change)
        self._mask_version = 0
        self._mask_cache_version = -1
        self._mask_cache = None
        # observation: FIXED base with padding up to max_drones.
        # Parts:
        # - pos (max_drones x 2) normalized at [-1, 1]
        #   - adj (max_drones x max_drones) in {0,1}
        # - mask (max_drones) at {0,1} indicating active drones
        # Optional extras (via arguments):
        # - zone_map (downsample of the forbidden-zone grid)
        # - edge_cross (matrix NxN: 1 if the segment (i,j) crosses a zone)
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

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim_pos + obs_dim_adj + obs_dim_mask + obs_dim_zone + obs_dim_edge_cross,),
            dtype=np.float32
        )

        # ---------- state ----------
        self._rng = np.random.RandomState()
        self._dist_matrix = None
        self._dist_pos_sig = None
        self._bonus_sucesso_dado = False
        # ---------- EFFECTIVE fitness weights ----------
        self.fit_w = dict(FIT_WEIGHTS_DEFAULT)
        if fit_weights:
            # partial override, ex: {"REL_MAIN": 800.0}
            self.fit_w.update({k: float(v) for k, v in fit_weights.items()})
        # REL_STEP: by default is 10% of REL_MAIN (reduces dense penalty/reward for per-step reliability)
        if self.fit_w.get("REL_STEP", None) is None:
            self.fit_w["REL_STEP"] = 0.1 * float(self.fit_w.get("REL_MAIN", 0.0))

        self.log_fit_terms = bool(log_fit_terms)

        self._reset_zonas()
        self._init_state()

        self._last_reward = 0.0
        # Tracking for zone shaping (violation delta and one-time bonus when clean)
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._zone_clean_once = False

        self._last_reward = 0.0
        self._episode_steps = 0
        self._max_episode_steps = 2000

        self._success_streak = 0  # consecutive successful steps
        self._success_need = 1

    # =========================
    # Forbidden zones
    # =========================
    def _reset_zonas(self):
        """(Re)generate forbidden zones and rebuild the boolean grid used for fast lookup.

Important: preserves the original semantics:
- If zonas_proibidas_param is provided, use exactly that set.
- Otherwise, sample procedurally (p_sem_zona, count, size, etc.)."""
        self.zonas_proibidas.clear()

        # Case 1: fixed zones provided externally
        if self.zonas_proibidas_param is not None:
            for (x, y) in self.zonas_proibidas_param:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.zonas_proibidas.add((x, y))
            self._rebuild_forbidden_grid()
            return

        # Case 2: chance of having no zone (procedural mode)
        if self.p_sem_zona > 0.0 and self._rng.rand() < self.p_sem_zona:
            self._rebuild_forbidden_grid()
            return

        # count and size
        count = self.zonas_count if self.zonas_count is not None else self._rng.randint(0, 3)
        if count <= 0:
            self._rebuild_forbidden_grid()
            return

        if self.zona_size:
            w, h = map(int, self.zona_size)
        else:
            w, h = max(2, self.grid_size // 12), max(2, self.grid_size // 12)

        if self.zona_single_fixed:
            # single fixed central zone
            cx, cy = self.grid_size // 2, self.grid_size // 2
            x0 = max(0, cx - w // 2)
            y0 = max(0, cy - h // 2)
            for xx in range(x0, min(self.grid_size, x0 + w)):
                for yy in range(y0, min(self.grid_size, y0 + h)):
                    self.zonas_proibidas.add((xx, yy))
            self._rebuild_forbidden_grid()
            return

        # multiple random zones
        for _ in range(count):
            x0 = self._rng.randint(0, max(1, self.grid_size - w))
            y0 = self._rng.randint(0, max(1, self.grid_size - h))
            for xx in range(x0, min(self.grid_size, x0 + w)):
                for yy in range(y0, min(self.grid_size, y0 + h)):
                    self.zonas_proibidas.add((xx, yy))

        self._rebuild_forbidden_grid()

    def _rebuild_forbidden_grid(self):
        """Rebuild a boolean [grid_size x grid_size] mask with True in forbidden cells.

- Preserves the original set representation (self.zonas_proibidas).
- Only speeds up repeated lookups in hot paths (step/fitness)."""
        if not self.zonas_proibidas:
            self._forbidden_grid = None
            self._zone_sig = 0
            # keeps observation shapes consistent even without zones
            self._rebuild_zone_map_cache()
            self._invalidate_edge_cross_cache()
            return
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.bool_)
        for (x, y) in self.zonas_proibidas:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                g[int(x), int(y)] = True
        self._forbidden_grid = g
        self._zone_sig = int(zlib.crc32(g.tobytes()))
        # updates zone-dependent caches
        self._rebuild_zone_map_cache()
        self._invalidate_edge_cross_cache()


    # =========================
    # Auxiliary caches: zone_map / edge_cross
    # =========================

    def _rebuild_zone_map_cache(self):
        """Recompute the downsampled forbidden-zone map used in obs_zone_map."""
        if not getattr(self, "obs_zone_map", False):
            self._zone_map_down = None
            self._zone_map_down_sig = None
            return

        ds = int(getattr(self, "_zone_map_ds", 0) or 0)
        if ds <= 0:
            self._zone_map_down = np.zeros((0,), dtype=np.float32)
            self._zone_map_down_sig = None
            return

        # If there is no zone in this episode, we still return a vector
        # with FIXED size (ds*ds) filled with zeros to keep the observation consistent.
        if self._forbidden_grid is None:
            self._zone_map_down = np.zeros((ds * ds,), dtype=np.float32)
            self._zone_map_down_sig = (int(getattr(self, "_zone_sig", 0)) ^ (int(self.zone_map_scale) << 16) ^ (ds << 1)) & 0xFFFFFFFF
            return

        scale = max(1, int(self.zone_map_scale))
        g = self._forbidden_grid.astype(bool, copy=False)  # (grid,grid) in [x,y]

        # padding to a multiple of scale
        target = ds * scale
        pad_x = max(0, target - g.shape[0])
        pad_y = max(0, target - g.shape[1])
        if pad_x or pad_y:
            g = np.pad(g, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=False)

        # reshape -> (ds, scale, ds, scale) and performs OR per block
        try:
            g4 = g.reshape(ds, scale, ds, scale)
            down = g4.any(axis=(1, 3)).astype(np.float32, copy=False)
        except Exception:
            # safe fallback
            down = np.zeros((ds, ds), dtype=np.float32)
            for ix in range(ds):
                for iy in range(ds):
                    x0 = ix * scale
                    y0 = iy * scale
                    block = g[x0:x0 + scale, y0:y0 + scale]
                    down[ix, iy] = 1.0 if np.any(block) else 0.0

        self._zone_map_down = down.reshape(-1).astype(np.float32, copy=False)
        # signature: zone + scale
        self._zone_map_down_sig = (int(getattr(self, "_zone_sig", 0)) ^ (scale << 16) ^ (ds << 1)) & 0xFFFFFFFF

    def _invalidate_edge_cross_cache(self):
        """Invalidate caches that depend on zones or positions (edge_cross)."""
        self._edge_cross_mat = None
        self._edge_cross_n = None
        self._edge_cross_zone_sig = None

    def _ensure_edge_cross_mat(self):
        """Ensure self._edge_cross_mat (NxN) exists for the current number of drones.

edge_cross_mat[i,j] = 1 if the segment between i and j crosses a forbidden zone."""
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

            # computes only the upper triangle and mirrors it (symmetric)
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
        """Incrementally update the edge_cross matrix when one drone moves."""
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

        # if the cache is invalid (n changed or the zone changed), recompute everything
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




    # =========================
    # Initial state

    # =========================
    # Initial state
    # =========================
    def _init_state(self):
        """Initialize the state:
- Drone 0 is in column 0 (x=0), on the center row (y≈grid/2) — source anchor.
- The last drone is in the last column (x=grid-1), on the center row — sink anchor.
- Intermediate drones are placed at random non-colliding positions.
- The initial adjacency matrix contains only strictly X-monotonic edges
  (x_dst > x_src), without considering forbidden-zone crossing or occupancy."""
        center_y = self.grid_size // 2
        last_col = self.grid_size - 1

        dirty_episode = bool(getattr(self, "_dirty_episode", False))

        pos = np.zeros((self.num_drones, 2), dtype=np.float32)

        # drone 0: left column (x=0), center row
        pos[0, 0] = 0.0
        pos[0, 1] = float(center_y)

        # drone N-1: right column (x=grid-1), center row
        if self.num_drones > 1:
            pos[self.num_drones - 1, 0] = float(last_col)
            pos[self.num_drones - 1, 1] = float(center_y)

        # Intermediate drones: random non-colliding positions.
        # The position may or may not fall inside a forbidden zone; this does not affect reset/init.
        for i in range(1, self.num_drones - 1):
            for _ in range(1000):
                rx = int(self._rng.randint(0, self.grid_size))
                ry = int(self._rng.randint(0, self.grid_size))

                collision = False
                for j in range(i):
                    if int(pos[j, 0]) == rx and int(pos[j, 1]) == ry:
                        collision = True
                        break
                if collision:
                    continue

                pos[i, 0] = float(rx)
                pos[i, 1] = float(ry)
                break
            else:
                pos[i, 0] = pos[0, 0]
                pos[i, 1] = pos[0, 1]

        # Keeps the logic of the "dirty" positioning curriculum,
        # but initial route construction completely ignores forbidden zones.
        if dirty_episode and getattr(self, "_force_drone_in_zone", False) and self.zonas_proibidas and self.num_drones > 2:
            k = int(self._rng.randint(1, self.num_drones - 1))
            zone_cells = list(self.zonas_proibidas)
            self._rng.shuffle(zone_cells)
            chosen = None
            for (zx, zy) in zone_cells:
                ok = True
                for j in range(self.num_drones):
                    if j == k:
                        continue
                    if int(round(float(pos[j, 0]))) == int(zx) and int(round(float(pos[j, 1]))) == int(zy):
                        ok = False
                        break
                if ok:
                    chosen = (zx, zy)
                    break
            if chosen is None and zone_cells:
                chosen = zone_cells[0]
            if chosen is not None:
                pos[k, 0] = float(chosen[0])
                pos[k, 1] = float(chosen[1])

        # Stores positions in the tensor
        self.drone_positions = torch.tensor(pos, dtype=torch.float32, device=DEVICE)

        # ---------- Adjacency-matrix construction ----------
        n = self.num_drones
        self.adjacency_matrix = torch.zeros(
            (n, n),
            dtype=torch.float32,
            device=DEVICE,
        )

        if n > 1:
            pos_np = _torch_to_numpy_cpu_view(self.drone_positions)

            def _edge_ok_monotonic(i: int, j: int) -> bool:
                if i == j:
                    return False
                # strictly X-monotonic route
                if float(pos_np[j, 0]) <= float(pos_np[i, 0]):
                    return False
                # by default, avoids, o direct link 0->N-1 when it is physically poor
                p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                if p_min_direct > 0.0 and n > 2:
                    if (i == 0 and j == n - 1) or (i == n - 1 and j == 0):
                        d_m = float(np.linalg.norm((pos_np[j] - pos_np[i]) * self.square_size_m))
                        p_dir = float(self._prob_link(d_m))
                        if p_dir < p_min_direct:
                            return False
                return True

            # Precomputes valid edges based only on X monotonicity
            valid_edges = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if _edge_ok_monotonic(i, j):
                        valid_edges[i][j] = True

            path_nodes = None

            # Attempts to build a monotonic route 0->...->N-1 using BFS
            prev = [-1] * n
            q = deque([0])
            prev[0] = 0
            while q:
                u = q.popleft()
                if u == n - 1:
                    break
                for v in range(n):
                    if valid_edges[u][v] and prev[v] == -1:
                        prev[v] = u
                        q.append(v)

            if prev[n - 1] != -1:
                rev_path = [n - 1]
                while rev_path[-1] != 0:
                    rev_path.append(prev[rev_path[-1]])
                path_nodes = list(reversed(rev_path))

            # Fallback: builds an X-monotonic route by choosing the closest neighbor to the right
            if path_nodes is None:
                def _nearest_right_neighbor(i: int) -> int:
                    xi = float(pos_np[i, 0])
                    cand = [j for j in range(n) if j != i and float(pos_np[j, 0]) > xi]
                    if not cand:
                        return n - 1
                    dists = [float(np.linalg.norm(pos_np[j] - pos_np[i])) for j in cand]
                    return cand[int(np.argmin(dists))]

                cur = 0
                visited = {cur}
                path_nodes = [cur]
                while cur != n - 1:
                    nxt = _nearest_right_neighbor(cur)
                    if nxt in visited or float(pos_np[nxt, 0]) <= float(pos_np[cur, 0]):
                        nxt = n - 1
                    path_nodes.append(nxt)
                    visited.add(nxt)
                    cur = nxt

            # Creates links for the main monotonic route
            for a, b in zip(path_nodes[:-1], path_nodes[1:]):
                if a != b and float(pos_np[b, 0]) > float(pos_np[a, 0]):
                    out_deg = int((self.adjacency_matrix[a, :] > 0).sum().item())
                    if out_deg < self.max_out_connections:
                        self.adjacency_matrix[a, b] = 1.0

            # ---------- Seed redundant monotonic links at reset ----------
            adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix)
            pos_np = _torch_to_numpy_cpu_view(self.drone_positions)

            def _reachable(adj: np.ndarray, src: int, dst: int) -> bool:
                if src == dst:
                    return True
                visited = {src}
                stack = [src]
                while stack:
                    u = stack.pop()
                    nbrs = np.where(adj[u] > 0.0)[0]
                    for v in nbrs:
                        v = int(v)
                        if v == dst:
                            return True
                        if v not in visited:
                            visited.add(v)
                            stack.append(v)
                return False

            def _can_add_edge(src: int, dst: int) -> bool:
                if src == dst:
                    return False
                if adj_np[src, dst] > 0.0:
                    return False
                if float(pos_np[dst, 0]) <= float(pos_np[src, 0]):
                    return False
                out_deg = int(np.sum(adj_np[src, :] > 0.0))
                if out_deg >= int(self.max_out_connections):
                    return False

                p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                if p_min_direct > 0.0 and n > 2:
                    if (src == 0 and dst == n - 1) or (src == n - 1 and dst == 0):
                        d_m = float(np.linalg.norm((pos_np[dst] - pos_np[src]) * self.square_size_m))
                        p_dir = float(self._prob_link(d_m))
                        if p_dir < p_min_direct:
                            return False

                # avoids cycles
                if _reachable(adj_np, dst, src):
                    return False
                return True

            # extra monotonic links; forbidden zones are not part of the criterion
            max_good_edges = max(1, n)
            added_good = 0
            for _ in range(max_good_edges * 10):
                if added_good >= max_good_edges:
                    break
                src = int(self._rng.randint(0, n - 1))
                cand = [j for j in range(n) if j != src and float(pos_np[j, 0]) > float(pos_np[src, 0])]
                self._rng.shuffle(cand)
                cand = cand[: min(16, len(cand))]
                cand.sort(
                    key=lambda j: ((float(pos_np[src, 0]) - float(pos_np[j, 0])) * self.square_size_m) ** 2
                    + ((float(pos_np[src, 1]) - float(pos_np[j, 1])) * self.square_size_m) ** 2
                )
                for dst in cand:
                    dst = int(dst)
                    if _can_add_edge(src, dst):
                        adj_np[src, dst] = 1.0
                        added_good += 1
                        break

            self.adjacency_matrix = torch.tensor(adj_np, dtype=torch.float32, device=DEVICE)

        self._bonus_sucesso_dado = False
        self._dist_matrix = None
        self._dist_pos_sig = None

        # action_masks cache invalid after reset
        if hasattr(self, "_mask_version"):
            self._mask_version += 1
            self._mask_cache = None
            self._mask_cache_version = -1

    # =========================
    # Helpers
    # =========================
    def _calc_distance_matrix(self):
        # cache: avoids recomputing O(N²) when positions have not changed
        n = int(self.num_drones)
        pos = np.ascontiguousarray(_torch_to_numpy_cpu_view(self.drone_positions[:n, :2]), dtype=np.float32)
        # robust signature to avoid collisions (the sum of coordinates may collide)
        sig = (int(zlib.crc32(pos.tobytes())), int(pos.shape[0]))
        if self._dist_pos_sig == sig and self._dist_matrix is not None:
            return self._dist_matrix

        # vectorized version (same semantics, much faster than a Python loop)
        px = (pos[:, 0] * float(self.square_size_m)).astype(np.float32, copy=False)
        py = (pos[:, 1] * float(self.square_size_m)).astype(np.float32, copy=False)
        dx = px[:, None] - px[None, :]
        dy = py[:, None] - py[None, :]
        d = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)

        self._dist_matrix = d
        self._dist_pos_sig = sig
        return d

    def _prob_link(self, d_m: float) -> float:
        """P(link) with a distance-based cache.

Does NOT change semantics: it only avoids repeatedly recomputing
prob_link_literature for the same distances (hot path in fitness/masking)."""
        d = float(d_m)
        if d <= 0.0:
            d = 1e-6
        pc = self._prob_cache.get(d)
        if pc is None:
            pc = float(prob_link_literature(d, self._radio_params))
            self._prob_cache[d] = pc
            # simple protection against pathological growth
            if len(self._prob_cache) > 200000:
                self._prob_cache.clear()
        return float(pc)

    def _has_path(self, src, dst, adj, out_neighbors=None):
        """BFS used to check whether a path src->dst exists.

out_neighbors (optional): list of outgoing neighbors for each node.
When provided, avoids scanning range(n), making the check faster without changing the logic."""
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

        # version using an adjacency list
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
        """Return a route (list of nodes) from src to dst using BFS over directed edges.
Return an empty list if no path exists.

out_neighbors (optional): list of outgoing neighbors for each node."""
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


    def _floyd_warshall(self, dist):
        return floyd_warshall(dist, directed=True)

    def _linha_passa_por_zona(self, p1, p2, link=None):
        """Check whether the segment p1->p2 passes through any forbidden cell.

Preserves the original semantics (discrete sampling with round), but performs
the check in vectorized form and, when available, uses the boolean grid
(self._forbidden_grid) to speed it up."""
        if not self.zonas_proibidas:
            return False
        x1, y1 = p1
        x2, y2 = p2
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) + 1)
        if steps <= 1:
            return False

        # identical sampling (round/banker's rounding)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        xs = np.rint(x1 + t * (x2 - x1)).astype(np.int32, copy=False)
        ys = np.rint(y1 + t * (y2 - y1)).astype(np.int32, copy=False)

        # bounds protection
        xs = np.clip(xs, 0, self.grid_size - 1)
        ys = np.clip(ys, 0, self.grid_size - 1)

        if self._forbidden_grid is not None:
            return bool(self._forbidden_grid[xs, ys].any())

        # fallback (rare case): uses the original set
        for x, y in zip(xs.tolist(), ys.tolist()):
            if (int(x), int(y)) in self.zonas_proibidas:
                return True
        return False


    # =========================
    # Gym API

    # =========================
    # Gym API
    # =========================
    def _filter_adj_by_disjoint_quality(self, adj_bin: np.ndarray, dist_m: np.ndarray, p_min: float) -> np.ndarray:
        """Filter edges used by the disjoint-path bonus.
Remove edges with prob_link(d) < p_min. A p_min <= 0 disables the filter.

Optimization (without changing semantics): iterate only over existing edges (O(E))
and use a local probability cache."""
        if p_min <= 0.0:
            return adj_bin

        out = adj_bin.copy()
        ei, ej = np.nonzero(out > 0.0)
        if ei.size == 0:
            return out

        # iterates only over existing edges
        for i, j in zip(ei.tolist(), ej.tolist()):
            d = float(dist_m[int(i), int(j)])
            if d <= 0.0:
                continue
            if self._prob_link(d) < float(p_min):
                out[int(i), int(j)] = 0.0
        return out

    def _max_edge_disjoint_paths_cap(self, cap: np.ndarray, src: int, dst: int) -> int:
        """Unit-capacity max-flow used to count edge-disjoint paths.

Replaces Edmonds-Karp with Dinic (same logical result, much faster on denser graphs).
`cap` does NOT need to be preserved; for compatibility, treat it as input-only."""
        n = int(cap.shape[0])
        src = int(src)
        dst = int(dst)
        if n <= 1 or src == dst:
            return 0

        # Dinic (small graphs: simple Python implementation)
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

        # builds the graph from the capacity matrix
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

    def _max_edge_disjoint_paths_no_zone_links(
        self,
        adj_np: np.ndarray,
        pos_np: np.ndarray,
        src: int,
        dst: int,
        pos_round: np.ndarray = None,
        zone_edge_cache: dict = None,
    ) -> int:
        """Count edge-disjoint paths ONLY if NO link crosses a forbidden zone.
Requested rule: if at least one link crosses a zone, return 0.

Optimizations (without changing semantics):
- use precomputed `pos_round` when provided;
- use a local endpoint cache (x1,y1,x2,y2) to avoid recomputing _linha_passa_por_zona."""
        n = int(adj_np.shape[0])
        if n <= 1:
            return 0
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0

        # If there are no zones, compute directly
        if not self.zonas_proibidas:
            cap = (adj_np > 0.0).astype(np.int32)
            return self._max_edge_disjoint_paths_cap(cap, src, dst)

        if pos_round is None:
            pos_round = np.rint(pos_np[:, :2]).astype(np.int32, copy=False)
        if zone_edge_cache is None:
            zone_edge_cache = {}

        def edge_crosses(i: int, j: int) -> bool:
            x1 = int(pos_round[i, 0]); y1 = int(pos_round[i, 1])
            x2 = int(pos_round[j, 0]); y2 = int(pos_round[j, 1])
            k = (x1, y1, x2, y2)
            v = zone_edge_cache.get(k)
            if v is None:
                v = bool(self._linha_passa_por_zona((x1, y1), (x2, y2), (i, j)))
                zone_edge_cache[k] = v
            return v

        # If ANY link exists that crosses a zone, disqualify it
        ei, ej = np.nonzero(adj_np > 0.0)
        for i, j in zip(ei.tolist(), ej.tolist()):
            if edge_crosses(int(i), int(j)):
                return 0

        cap = (adj_np > 0.0).astype(np.int32)
        return self._max_edge_disjoint_paths_cap(cap, src, dst)

    def _max_edge_disjoint_partial_paths(self, adj_np: np.ndarray, pos_np: np.ndarray, src: int, dst: int,
                                         min_progress: float = 0.5) -> int:
        """Count the maximum number of *partial* edge-disjoint paths from src toward dst.

Idea:
- Define a set of "partial targets": nodes reachable from src that have progressed
  by at least `min_progress` (0..1) along the src->dst segment (vector projection).
- Connect each partial target to a super-sink with capacity 1 and compute max-flow
  with unit capacities.
- The flow value is the number of parallel edge-disjoint routes that progress toward
  the destination, even if they do not reach dst.

Note:
- Forbidden zones are not filtered here; zone penalties handle them separately."""
        n = int(adj_np.shape[0])
        if n <= 1:
            return 0
        src = int(src)
        dst = int(dst)
        if src == dst:
            return 0

        # vector src->dst
        p0 = np.array([float(pos_np[src, 0]), float(pos_np[src, 1])], dtype=np.float64)
        pL = np.array([float(pos_np[dst, 0]), float(pos_np[dst, 1])], dtype=np.float64)
        v = pL - p0
        den = float(v[0] * v[0] + v[1] * v[1])
        if den <= 1e-12:
            return 0

        cap0 = (adj_np > 0.0).astype(np.int32)

        # reachable from src
        reachable = [False] * n
        reachable[src] = True
        q = deque([src])
        while q:
            u = q.popleft()
            for vv in range(n):
                if cap0[u, vv] > 0 and not reachable[vv]:
                    reachable[vv] = True
                    q.append(vv)

        # collects partial targets by projection onto the segment src->dst
        min_progress = float(max(0.0, min(1.0, min_progress)))
        targets = []
        for i in range(n):
            if i == src:
                continue
            if not reachable[i]:
                continue
            pi = np.array([float(pos_np[i, 0]), float(pos_np[i, 1])], dtype=np.float64)
            t = float(((pi - p0) @ v) / den)  # normalized projection
            if t >= min_progress:
                targets.append(i)

        if not targets:
            return 0

        # super-sink
        sink = n
        cap = np.zeros((n + 1, n + 1), dtype=np.int32)
        cap[:n, :n] = cap0
        for i in targets:
            cap[int(i), sink] = 1

        return int(self._max_edge_disjoint_paths_cap(cap, src, sink))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            # Seeds the environment's internal RNG
            self._rng.seed(int(seed))

        if self.variable_drones and self.min_drones and self.max_drones:
            # Samples num_drones within [min_drones, max_drones]
            self.num_drones = int(
                self._rng.randint(int(self.min_drones), int(self.max_drones) + 1)
            )
        else:
            # Ensures that num_drones does not exceed max_drones (capacity)
            if self.num_drones > self.max_drones:
                self.num_drones = self.max_drones

        self._reset_zonas()

        # -------------------------
        # Curriculum: dirty reset
        # -------------------------
        # Idea:
        # - at the beginning, only some episodes start "dirty" (with drones/edges in zones)
        # - at the end, 100% of episodes are dirty (and there is always at least one drone in a zone),
        # forcing the agent to learn to "clean" by removing links and relocating drones.
        self._reset_count += 1
        if self.zonas_proibidas:
            prog = min(1.0, float(self._reset_count) / float(self.dirty_anneal_episodes))
            p_dirty = self.dirty_reset_prob_start + (self.dirty_reset_prob_end - self.dirty_reset_prob_start) * prog
            p_force = self.dirty_force_drone_in_zone_prob_start + (
                self.dirty_force_drone_in_zone_prob_end - self.dirty_force_drone_in_zone_prob_start
            ) * prog

            self._dirty_episode = (float(self._rng.rand()) < float(p_dirty))
            # forces a drone inside the zone only when "dirty"
            self._force_drone_in_zone = bool(self._dirty_episode) and (float(self._rng.rand()) < float(p_force))
        else:
            self._dirty_episode = False
            self._force_drone_in_zone = False

        # Recomputes the initial state according to the curriculum
        self._init_state()
        # reset zone tracking
        self._prev_links_cross_zone = None
        self._prev_drones_in_zone = None
        self._zone_clean_once = False
        self._episode_steps = 0
        # Success reset (prevents carrying streak/bonus across episodes)
        self._success_streak = 0
        self._bonus_sucesso_dado = False
        obs = self._get_obs()
        if _GYMNASIUM:
            return obs, {"reset": True}
        return obs

    def step(self, action):
        """Execute one environment step.

Masking adjustment: apply toggle_op (ADD/REMOVE) BEFORE movement.
This keeps the action mask computed in the pre-action state consistent with link
validation and avoids many artificial `invalid_link_action` events when an action
contains both movement and a link toggle in the same step.

action: [move_idx, step_code, dir_code, toggle_op]"""

        # ---------- decode action ----------
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

        # Compatibility: if someone provides the old action 6D, convert it to toggle_op.
        # old action: [move_idx, step_code, dir_code, toggle_flag, src, dst]
        if len(action) >= 6:
            old_toggle_flag = int(action[3])
            old_src = int(action[4])
            old_dst = int(action[5])
            if old_toggle_flag == 1 and old_src != old_dst:
                edge_idx = self._edge_pair_to_idx.get((old_src, old_dst), None)
                if edge_idx is not None:
                    # in the old format it was "toggle" (remove if it exists, otherwise add)
                    if (old_src < self.num_drones and old_dst < self.num_drones and
                            self.adjacency_matrix[old_src, old_dst].item() > 0.5):
                        toggle_op = 1 + self._E + edge_idx  # REMOVE
                    else:
                        toggle_op = 1 + edge_idx  # ADD

        # ---------- 1) apply toggle BEFORE movement ----------
        invalid_link_action = False
        link_changed = False

        if toggle_op != 0:
            op = int(toggle_op) - 1
            is_remove = (op >= self._E)
            edge_idx = (op - self._E) if is_remove else op

            if 0 <= edge_idx < self._E:
                src, dst = self._edge_pairs[edge_idx]

                # drones out of range / inactive
                if src >= self.num_drones or dst >= self.num_drones:
                    invalid_link_action = True
                elif src == dst:
                    invalid_link_action = True
                else:
                    is_on = (self.adjacency_matrix[src, dst].item() > 0.5)

                    if is_remove:
                        # Removal: is valid only if the edge exists.
                        if is_on:
                            self.adjacency_matrix[src, dst] = 0.0
                            link_changed = True
                        else:
                            invalid_link_action = True
                    else:
                        # Addition: is valid only if the edge does NOT exist and respects the rules.
                        if is_on:
                            invalid_link_action = True
                        else:
                            # 1) out-degree limit
                            out_deg_src = int(torch.sum(self.adjacency_matrix[src] > 0.5).item())
                            if out_deg_src >= self.max_out_connections:
                                invalid_link_action = True

                            # 2) avoids creating a direct link 0<->N-1 when it is physically poor (p below the threshold).
                            if (not invalid_link_action):
                                p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                                if p_min_direct > 0.0 and self.num_drones > 2:
                                    if (src == 0 and dst == self.num_drones - 1) or (
                                            src == self.num_drones - 1 and dst == 0):
                                        dx_m = (float(self.drone_positions[src, 0].item()) - float(
                                            self.drone_positions[dst, 0].item())) * self.square_size_m
                                        dy_m = (float(self.drone_positions[src, 1].item()) - float(
                                            self.drone_positions[dst, 1].item())) * self.square_size_m
                                        d_m = float(math.hypot(dx_m, dy_m))
                                        p_dir = float(self._prob_link(d_m))
                                        if p_dir < p_min_direct:
                                            invalid_link_action = True

                            # 3) Hard mask: if enabled, blocks ADD operations for edges crossing forbidden zones.
                            if (not invalid_link_action) and self.hard_mask_zone_edges and self.zonas_proibidas:
                                p1 = (int(round(float(self.drone_positions[src, 0].item()))), int(round(float(self.drone_positions[src, 1].item()))))
                                p2 = (int(round(float(self.drone_positions[dst, 0].item()))), int(round(float(self.drone_positions[dst, 1].item()))))
                                if self._linha_passa_por_zona(p1, p2, (src, dst)):
                                    invalid_link_action = True

                            # 3b) Blocks cycle creation: if a path already exists dst -> src,
                            # then adding src->dst closes a directed cycle.
                            if not invalid_link_action:
                                n = int(self.num_drones)
                                if self._has_path(dst, src, self.adjacency_matrix[:n, :n]):
                                    invalid_link_action = True

                            # 4) (mask-relax) does NOT block based on left-link. This constraint is learned through penalties.
                            if not invalid_link_action:
                                self.adjacency_matrix[src, dst] = 1.0
                                link_changed = True
            else:
                invalid_link_action = True

        if link_changed:
            self._mask_version += 1

        # ---------- 2) Movement (1 drone by step) ----------
        old_pos = self.drone_positions.clone()
        invalid_move_count = 0

        # HARD: store how many EXISTING links cross zones BEFORE movement.
        # This allows blocking movements that "push" a link into a forbidden zone.
        prev_links_cross_zone = None
        if (getattr(self, "hard_block_move_cross_zone", True)
                and dir_code != 0 and step_cells > 0
                and getattr(self, "zonas_proibidas", None)):
            # edge_cross is cached; ensures it exists for the current n
            if self.obs_edge_cross or self.hard_mask_zone_edges:
                self._ensure_edge_cross_mat()
                if self._edge_cross_mat is not None:
                    n = int(self.num_drones)
                    if n > 1:
                        adj_np = _torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]).astype(np.uint8, copy=False)
                        cross_np = self._edge_cross_mat[:n, :n]
                        prev_links_cross_zone = int(((adj_np > 0) & (cross_np > 0)).sum())

        # IMPORTANT: drones 0 and (num_drones-1) are fixed (anchors).
        # Only ONE intermediate drone (move_idx) may move per step.
        if dir_code != 0 and step_cells > 0:
            if move_idx <= 0 or move_idx >= self.num_drones - 1:
                # attempt to move an anchor or invalid index
                invalid_move_count += 1
            else:
                i = move_idx
                x0 = int(self.drone_positions[i, 0].item())
                y0 = int(self.drone_positions[i, 1].item())
                x, y = x0, y0
                invalid_zone_move = False

                # HARD invalid: if the drone is OUTSIDE the zone and the movement (at any micro-step)
                # would attempt to ENTER the forbidden zone, then the action is invalid and does NOT move.
                # Note: if the drone is already inside (dirty episode), we allow movement to EXIT.
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

            # Updates edge_cross (depends on position) and invalidates masks if necessary
            if self.obs_edge_cross or self.hard_mask_zone_edges:
                self._update_edge_cross_after_move(int(move_idx))

            # HARD post-movement: if the movement INCREASED the number of links crossing zones,
            # then undo the movement and mark the action as invalid.
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

                        # Reverts position
                        self.drone_positions[mi, 0] = old_pos[mi, 0]
                        self.drone_positions[mi, 1] = old_pos[mi, 1]

                        # Updates edge_cross back to the previous position
                        if self.obs_edge_cross or self.hard_mask_zone_edges:
                            self._update_edge_cross_after_move(mi)

                        move_reverted = True
                        moved = False

            # If there is a position-dependent hard mask (zone), the set of valid ADD actions changes with movement
            if (not move_reverted) and self.hard_mask_zone_edges:
                self._mask_version += 1
                self._mask_cache = None
                self._mask_cache_version = -1

        # ---------- fit/reward ----------
        reward, info = self._calculate_fit()

        # Penalizes invalid actions to help the policy learn faster
        if invalid_move_count > 0 or invalid_link_action:
            reward -= float(self.fit_w.get("INVALID_ACTION_PENALTY", 0.0)) * (float(self.num_drones) / float(max(1, self.max_drones)))

        # ----- success criterion -----
        success_now = bool(
            info.get("path_exists", False)
            and self.current_reliability >= RELIABILITY_TARGET
            and info.get("reach_frac", 0.0) >= 0.85
            and info.get("drones_in_zone", 0) == 0
            and info.get("links_cross_zone", 0) == 0
            and info.get("left_links_count", 0) == 0
            and info.get("is_dag", 1) == 1
        )

        if success_now:
            self._success_streak += 1
        else:
            self._success_streak = 0

        success_reached = self._success_streak >= self._success_need
        if success_reached and not self._bonus_sucesso_dado:
            reward += 500.0
            self._bonus_sucesso_dado = True

        # ----- episode termination -----
        self._last_reward = float(reward)
        self._episode_steps += 1

        done = False
        if self._episode_steps >= self._max_episode_steps or success_reached:
            done = True

        truncation = (self._episode_steps >= self._max_episode_steps)
        obs = self._get_obs()
        info["reliability"] = float(self.current_reliability)
        info["invalid_action"] = bool(invalid_move_count > 0 or invalid_link_action)
        info["invalid_action_penalty_scale"] = float(self.num_drones) / float(max(1, self.max_drones))
        info["invalid_move_count"] = int(invalid_move_count)
        info["success"] = bool(success_reached)
        if success_reached:
            info["success_at_step"] = int(self._episode_steps)

        # Terminal shaping: aligns training with the TEST metric (reliability of the episode's FINAL state).
        if done:
            info["is_success"] = bool(success_reached)
            # Recomputes reliability in the FINAL state to avoid any divergence (cache/stale)
            try:
                n = int(self.num_drones)
                adj_t = _torch_to_numpy_cpu_view(self.adjacency_matrix)
                pos_t = _torch_to_numpy_cpu_view(self.drone_positions)
                adj_np = (adj_t[:n, :n] > 0).astype(np.float32)
                pos_np = np.asarray(pos_t[:n, :2], dtype=np.float32)

                # Snapshot terminal: avoids divergence during post-processing when VecEnv performs auto-reset.
                # (The TesteMultiDrone.txt test prefers these fields.)
                info["terminal_adj01"] = adj_np
                info["terminal_pos_cells"] = pos_np
                info["terminal_num_drones"] = int(n)
                info["terminal_square_size_m"] = float(self.square_size_m)
                path_final = bool(self._has_path(0, n - 1, adj_np)) if n > 1 else False
                if path_final:
                    self.current_reliability = float(
                        calcula_confiabilidade_iterativa(adj_np, pos_np, self.square_size_m, params=self._radio_params)
                    )
                else:
                    self.current_reliability = 0.0
            except Exception:
                pass

            info["reliability_terminal"] = float(self.current_reliability)
            info["done_reason"] = ("success" if success_reached else ("timeout" if truncation else "done"))

            viol = int(info.get("drones_in_zone", 0) + info.get("links_cross_zone", 0))
            info["zone_violations_terminal"] = int(viol)

            # Terminal throughput (final throughput, in Mbps)
            try:
                info["throughput_terminal"] = float(info.get("throughput", getattr(self, "current_throughput", 0.0)))
            except Exception:
                info["throughput_terminal"] = 0.0


            # Terminal bonus/penalty based on final reliability
            term_rel_w = float(self.fit_w.get("TERM_REL", 0.0))
            if term_rel_w != 0.0:
                # Uses the SAME metric of _calculate_fit(): -log10(1-rel) mapped between REL_BASE and TARGET
                eps = 1e-12
                rel_clipped = max(eps, min(1.0 - eps, float(self.current_reliability)))
                fail = max(eps, 1.0 - rel_clipped)
                fail_b = max(eps, 1.0 - float(RELIABILITY_BASE))
                fail_t = max(eps, 1.0 - float(RELIABILITY_TARGET))
                a = -math.log10(fail_b)
                b = -math.log10(fail_t)
                cur = -math.log10(fail)
                if b <= a + 1e-12:
                    rel_term = 1.0 if rel_clipped >= float(RELIABILITY_TARGET) else 0.0
                else:
                    rel_term = (cur - a) / (b - a)
                rel_term = max(0.0, min(1.0, float(rel_term)))
                term_rel_score = term_rel_w * (2.0 * rel_term - 1.0)
                reward += term_rel_score
                info["term_rel_term"] = float(rel_term)
                info["term_rel_score"] = float(term_rel_score)

            # terminal success bonus (only if the zone is clean)
            if (self.current_reliability >= RELIABILITY_TARGET and viol == 0 and
                    int(info.get("left_links_count", 0)) == 0 and int(info.get("is_dag", 1)) == 1):
                reward += float(self.fit_w.get("TERM_SUCCESS_BONUS", 0.0))

            # terminal penalty for zone violations
            if viol > 0:
                reward -= float(self.fit_w.get("TERM_ZONE_PEN", 0.0))

            self._last_reward = float(reward)

        # Return compatible with Gymnasium vs Gym
        if _GYMNASIUM:
            return obs, float(reward), bool(done), bool(truncation), info
        return obs, float(reward), bool(done or truncation), info
    def _get_obs(self):
        """Observation padded up to max_drones with an active-drone mask.
- The first self.num_drones entries represent active drones.
- Remaining indices up to max_drones are zeroed and have mask=0."""
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
        mask[:num] = 1.0

        obs_parts = [pos_full.reshape(-1), adj_full.reshape(-1), mask]

                # (optional) downsampled forbidden-zone map
        if self.obs_zone_map:
            # Ensures an updated cache and, most importantly, a FIXED size
            # even when there is no zone (p_sem_zona>0).
            expected = int(self._zone_map_ds * self._zone_map_ds) if int(getattr(self, '_zone_map_ds', 0) or 0) > 0 else 0
            sig = (int(getattr(self, '_zone_sig', 0)) ^ (int(self.zone_map_scale) << 16) ^ (int(self._zone_map_ds) << 1)) & 0xFFFFFFFF
            if (self._zone_map_down is None) or (self._zone_map_down_sig != sig) or (expected and (self._zone_map_down.shape[0] != expected)):
                self._rebuild_zone_map_cache()
            if expected > 0 and (self._zone_map_down is None or self._zone_map_down.shape[0] != expected):
                # ultra-safe fallback: FIXED size
                self._zone_map_down = np.zeros((expected,), dtype=np.float32)
                self._zone_map_down_sig = sig
            obs_parts.append(self._zone_map_down if self._zone_map_down is not None else np.zeros((expected,), dtype=np.float32))

        # (optional) NxN matrix indicating zone crossing per edge (i,j)
        if self.obs_edge_cross:
            cross_full = np.zeros((max_n, max_n), dtype=np.float32)
            cross_mat = self._ensure_edge_cross_mat()
            if cross_mat is not None and num > 0:
                # cross_mat is (num,num)
                cross_full[:num, :num] = cross_mat.astype(np.float32, copy=False)
            obs_parts.append(cross_full.reshape(-1))

        return np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)

    # =========================
    # Fitness / Reward
    # =========================

    def _emax_scales(self, n=None):
        """Compute normalization scales for variable number of active drones.

        Emax(n) = n * max_out_connections (upper bound on directed edges).
        Scales are defined so that when n == self.max_drones the scale is ~1.0.
        """
        if n is None:
            n = int(self.num_drones)
        else:
            n = int(n)
        n_ref = int(self.max_drones)
        max_out = int(self.max_out_connections)

        emax_n = max(1, n * max_out)
        emax_ref = max(1, n_ref * max_out)

        link_scale = float(emax_ref) / float(emax_n)              # scale for link-count terms
        drone_scale = float(n_ref) / float(max(1, n))             # scale for drone-count terms
        drone_scale_wo0 = float(max(1, n_ref - 1)) / float(max(1, n - 1))  # excludes drone 0
        relay_scale = float(max(1, n_ref - 2)) / float(max(1, n - 2))      # excludes anchors 0 and n-1

        return {
            "Emax_n": int(emax_n),
            "Emax_ref": int(emax_ref),
            "link_scale": float(link_scale),
            "drone_scale": float(drone_scale),
            "drone_scale_wo0": float(drone_scale_wo0),
            "relay_scale": float(relay_scale),
        }


    def _calculate_fit(self):
        """Compute the fitness of the current state.

Objective priority:
1. A path from node 0 to node N-1 must exist.
2. High end-to-end reliability (RELIABILITY_TARGET).
3. No drone or link may occupy/cross a forbidden zone.
4. Throughput must remain below the maximum cap.
5. Keep the number of hops reasonable, but only after the objectives above are good.
6. Encourage a well-formed layout between source and destination with a VERY small weight."""
        w = self.fit_w  # effective fitness weights (centralized)

        # Basic conversions (always considering ONLY active drones: 0..num_drones-1)
        n = int(self.num_drones)
        scales = self._emax_scales(n)
        Emax_n = int(scales["Emax_n"])
        Emax_ref = int(scales["Emax_ref"])
        link_scale = float(scales["link_scale"])
        drone_scale = float(scales["drone_scale"])
        drone_scale_wo0 = float(scales["drone_scale_wo0"])
        relay_scale = float(scales["relay_scale"])
        pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2])
        adj_np = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5).astype(np.float32)
        # Precomputes edges and adjacency lists (hot-path): avoids scans O(n²) repeated
        ei, ej = np.nonzero(adj_np > 0.0)
        out_neighbors = [[] for _ in range(n)]
        in_neighbors = [[] for _ in range(n)]
        if ei.size > 0:
            for i, j in zip(ei.tolist(), ej.tolist()):
                out_neighbors[int(i)].append(int(j))
                in_neighbors[int(j)].append(int(i))
        out_deg = np.bincount(ei.astype(np.int32, copy=False), minlength=n) if ei.size > 0 else np.zeros(n, dtype=np.int32)
        in_deg = np.bincount(ej.astype(np.int32, copy=False), minlength=n) if ej.size > 0 else np.zeros(n, dtype=np.int32)
        pos_round = np.rint(pos_np[:, :2]).astype(np.int32, copy=False)
        dist_full = self._calc_distance_matrix()

        # slices (dist_full is already n x n)
        dist_m = dist_full[:n, :n]
        # path_exists must be defined before it is used (local scope in Python)
        path_exists = bool(self._has_path(0, n - 1, adj_np, out_neighbors=out_neighbors)) if n > 1 else False
        # -------------------------
        # No loops: detects cycles in the graph (a DAG is expected)
        # -------------------------
        is_dag = True
        if n > 1 and ei.size > 0:
            indeg_tmp = in_deg.astype(np.int32, copy=True)
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

        cycle_penalty = 0.0
        if not is_dag:
            cycle_penalty = -float(w.get("CYCLE_PENALTY", 0.0)) * float(link_scale)

        # -------------------------
        # Bottleneck shaping: improves the worst link on the path 0->(n-1)
        # - finds a path (BFS) and computes p_min along the path.
        # - rewards if p_min exceeds BOTTLENECK_TARGET_P; penalizes if below.
        # -------------------------
        bottleneck_min_p = 0.0
        bottleneck_max_dist_m = 0.0
        bottleneck_term = 0.0
        if path_exists:
            path_nodes_b = self._get_any_path_nodes(0, n - 1, adj_np, out_neighbors=out_neighbors)
            if len(path_nodes_b) >= 2:
                p_min = 1.0
                d_max = 0.0
                for a, b in zip(path_nodes_b[:-1], path_nodes_b[1:]):
                    d = float(dist_m[a, b])
                    d_max = max(d_max, d)
                    p = float(self._prob_link(d))
                    p_min = min(p_min, p)
                bottleneck_min_p = float(p_min)
                bottleneck_max_dist_m = float(d_max)
                bt_w = float(w.get("BOTTLENECK_W", 0.0))
                bt_t = float(w.get("BOTTLENECK_TARGET_P", 0.7))
                if bt_w != 0.0:
                    bottleneck_term = bt_w * (float(bottleneck_min_p) - bt_t)
                    bottleneck_term = max(-bt_w, min(bt_w, bottleneck_term))

        # -------------------------
        # Connection quality (shorter links are better) - small weight
        # connection_quality ∈ [0, 1]
        # -------------------------
        num_links = int(ei.size)
        # Emax-based normalization can overweight link penalties when graphs are sparse (few links).
        # To keep training stable with variable n, soften link scaling by link density (num_links / Emax_n).
        link_density = float(num_links) / float(max(1, Emax_n)) if num_links > 0 else 0.0
        link_scale_eff = float(link_scale) * float(np.sqrt(max(1e-6, link_density)))
        if num_links > 0:
            mean_dist = float(dist_m[ei, ej].mean())
            norm_mean_dist = mean_dist / max(1.0, self.grid_size * self.square_size_m)
            connection_quality = 1.0 / (1.0 + norm_mean_dist)
        else:
            connection_quality = 0.0

        conn_quality_term = w["CONN_QUALITY_COEF"] * float(connection_quality)

        # -------------------------
        # Reachability to the last drone (weight=1 in the fitness)
        # reach_last_penalty = -(1 - reach_frac) ∈ [-1, 0]
        # -------------------------
        reach_count = 0
        reach_frac = 0.0
        if n > 1:
            # A single BFS on the reverse graph starting from the last drone
            can_reach_last = np.zeros(n, dtype=bool)
            dst = n - 1
            can_reach_last[dst] = True
            qrev = deque([dst])
            while qrev:
                u = qrev.popleft()
                for p in in_neighbors[u]:
                    if not can_reach_last[p]:
                        can_reach_last[p] = True
                        qrev.append(p)
            # counts only src at [0..n-2] (same semantics as the old code)
            reach_count = int(np.sum(can_reach_last[: max(0, n - 1)]))
            reach_frac = reach_count / max(1, n - 1)
        reach_last_penalty = -(1.0 - float(reach_frac))

        # -------------------------
        # Reachability FROM the first drone (0) – penalizes components disconnected from node 0
        # reach_from_first_frac = fraction of nodes (1..N-1) reachable via directed edges from node 0
        # unreachable_from_first_penalty = -UNREACHABLE_FROM_FIRST_PENALTY * (#unreachable nodes)
        # -------------------------
        reachable = np.zeros(n, dtype=bool)
        reachable[0] = True
        q0 = deque([0])
        while q0:
            u = q0.popleft()
            # follows outgoing edges u -> v (via the adjacency list)
            for v in out_neighbors[u]:
                if not reachable[v]:
                    reachable[v] = True
                    q0.append(v)

        reachable_count = int(np.sum(reachable[1:]))  # excludes node 0 itself
        reach_from_first_frac = reachable_count / max(1, n - 1)
        unreachable_from_first_count = int((n - 1) - reachable_count)
        unreachable_from_first_penalty = -float(w.get("UNREACHABLE_FROM_FIRST_PENALTY", 0.0)) * float(unreachable_from_first_count) * float(drone_scale_wo0)
        reach_from_first_penalty = -float(w.get("REACH_FROM_FIRST_W", 0.0)) * float(1.0 - reach_from_first_frac)

        # -------------------------
        # Relay structure (penalties only; weight=1 in the fitness)
        # - ISOLATED_NODE_PENALTY is included in invalid_relay_penalty
        # - other terms use internal penalties (0.2 * diff etc.)
        # -------------------------
        invalid_relay_penalty = 0.0
        blocked_relay_penalty = 0.0
        imbalance_penalty = 0.0
        dead_end_penalty = 0.0

        for node in range(1, n - 1):
            in_deg_i = int(in_deg[node])
            out_deg_i = int(out_deg[node])

            # completely isolated node (does not act as a relay)
            if in_deg_i == 0 and out_deg_i == 0:
                invalid_relay_penalty -= float(w["ISOLATED_NODE_PENALTY"])
                continue

            # Nodes with only incoming or only outgoing edges (undesired "terminal" structures)
            if in_deg_i == 0 and out_deg_i > 0:
                blocked_relay_penalty -= 0.2

            if in_deg_i > 0 and out_deg_i == 0 and node != n - 1:
                blocked_relay_penalty -= 0.2
                dead_end_penalty -= 0.2

            # Imbalance between in-degree and out-degree
            if (in_deg_i + out_deg_i) > 0:
                diff = abs(in_deg_i - out_deg_i)
                imbalance_penalty -= 0.2 * diff

        # -------------------------
        # Normalizes structural penalties by number of relays (variable n)
        invalid_relay_penalty *= float(relay_scale)
        blocked_relay_penalty *= float(relay_scale)
        imbalance_penalty *= float(relay_scale)
        dead_end_penalty *= float(relay_scale)

        # "Backward" links (x_dst < x_src) – computed but NOT included in the current fitness
        # -------------------------
        if num_links > 0:
            left_links_count = int(np.sum(pos_np[ej, 0] < pos_np[ei, 0]))
        else:
            left_links_count = 0
        left_links_penalty = 0.0  # gated: applied only when path 0->(n-1) exists

        # -------------------------
        # Hops and latency: longest path 0->N-1
        # latency_penalty ≤ 0, latency_reward ≥ 0
        # -------------------------
        path_exists = False
        max_hops = 0
        avg_hops = 0.0
        hops_score = 0.0
        latency_penalty = 0.0
        latency_reward = 0.0

        longest_path_dist = 0.0

        if num_links > 0 and n > 1:
            visited = np.zeros(n, dtype=bool)
            dst = n - 1
            state = {"path_exists": False, "max_hops": 0, "longest_path_dist": 0.0}

            def dfs(u, hops, dist_acc):
                if u == dst:
                    state["path_exists"] = True
                    if hops > state["max_hops"]:
                        state["max_hops"] = hops
                        state["longest_path_dist"] = dist_acc
                    return
                visited[u] = True
                for v in out_neighbors[u]:
                    if not visited[v]:
                        dfs(v, hops + 1, dist_acc + float(dist_m[u, v]))
                visited[u] = False

            dfs(0, 0, 0.0)

            path_exists = bool(state["path_exists"])
            max_hops = int(state["max_hops"])
            longest_path_dist = float(state["longest_path_dist"])

            if path_exists and max_hops > 0:
                avg_hops = float(max_hops)
                hops = max_hops
                latency_ms = (
                        hops * LATENCY_PER_HOP_MS
                        + longest_path_dist / max(1e-6, SPEED_M_PER_MS)
                )
                if latency_ms > MAX_LATENCY_MS:
                    latency_penalty -= 0.005 * (latency_ms - MAX_LATENCY_MS) / MAX_LATENCY_MS
                else:
                    latency_reward += 0.005 * (MAX_LATENCY_MS - latency_ms) / MAX_LATENCY_MS
            else:
                max_hops -= 15.0


        # Gate left-link penalty: allow exploration until a 0->(n-1) path exists
        if path_exists and left_links_count > 0:
            left_links_penalty = -float(w["LEFT_LINK_COEF"]) * float(left_links_count) * float(link_scale_eff)
            left_links_gate = 1
        else:
            left_links_penalty = 0.0
            left_links_gate = 0

        self.current_latency_penalty = float(latency_penalty)

        # -------------------------
        # Throughput (simplified) – computed but NOT included in the current fitness
        # Normalizes by Emax(n): keeps approximately the same scale when n varies (9..13)
        # -------------------------
        throughput = 0.0
        num_links_eff = int(max(1, int(round(float(num_links) * float(link_scale_eff))))) if num_links > 0 else 0
        if path_exists:
            throughput = THROUGHPUT_ORIGEM_MBPS
            throughput = min(throughput * max(1, num_links_eff), THROUGHPUT_MAX_MBPS)
        self.current_throughput = float(throughput)

        throughput_penalty = 0.0
        if throughput > 0.0:
            ratio_tp = throughput / THROUGHPUT_MAX_MBPS
            if ratio_tp >= 0.85:
                # progressive penalty starting at 85% of the cap, reaching the maximum at 100%
                excess = (ratio_tp - 0.85) / 0.15
                excess = max(0.0, min(1.0, float(excess)))
                throughput_penalty -= float(w["THROUGHPUT_PENALTY_MAX"]) * (excess ** 2)

        # -------------------------
        # Reliability 0->N-1 (physical calculation)
        # -------------------------
        if path_exists:
            try:
                reliability = float(
                    calcula_confiabilidade_iterativa(adj_np, pos_np, self.square_size_m, params=self._radio_params)
                )
            except Exception:
                reliability = 0.0
        else:
            reliability = 0.0

        self.current_reliability = float(reliability)

        # -------------------------
        # Spread terms (layout)
        # nn_reward, uniform_reward ~ [0,1] when reliability > 0
        # (in the current code, If reliability==0: nn_reward gets -100)
        # -------------------------
        nn_reward = 0.0
        uniform_reward = 0.0
        nn_mean_dist = 0.0

        if reliability > 0.0:
            if n > 1:
                # NN-spacing
                nn_dists = []
                for i in range(n):
                    dists_i = dist_m[i, :n].copy()
                    dists_i[i] = np.inf
                    nn = float(np.min(dists_i))
                    if np.isfinite(nn):
                        nn_dists.append(nn)
                if nn_dists:
                    nn_mean_dist = float(np.mean(nn_dists))
                    max_possible = math.sqrt(2.0) * self.grid_size * self.square_size_m
                    nn_norm = min(1.0, nn_mean_dist / max(1.0, max_possible))
                    nn_reward = nn_norm

                # 1D uniformity along the segment (drone0 -> droneN-1)
                # Helps spread drones from start to end (avoids clustering), without encouraging 2D border filling.
                p0 = pos_np[0, :2]
                p1 = pos_np[n - 1, :2]
                v = p1 - p0
                vv = float(v[0] * v[0] + v[1] * v[1])
                bins = int(w.get("UNIFORM1D_BINS", 8))

                if vv > 1e-9 and n > 2 and bins > 0:
                    counts = np.zeros((bins,), dtype=np.float32)

                    # uses only intermediate drones (does not include anchors 0 and N-1)
                    for i in range(1, n - 1):
                        p = pos_np[i, :2]
                        t = float(((p - p0) @ v) / vv)
                        t = max(0.0, min(1.0, t))
                        b = min(bins - 1, int(t * bins))
                        counts[b] += 1.0

                    # optional: ignores bins whose center point lies in a forbidden zone
                    valid = np.ones((bins,), dtype=bool)
                    if self.zonas_proibidas:
                        for b in range(bins):
                            tmid = (b + 0.5) / bins
                            pmid = p0 + tmid * v
                            cx = int(round(pmid[0]))
                            cy = int(round(pmid[1]))
                            if (cx, cy) in self.zonas_proibidas:
                                valid[b] = False

                    valid_counts = counts[valid]
                    if valid_counts.size > 0:
                        mean_count = float(valid_counts.mean())
                        var = float(((valid_counts - mean_count) ** 2).mean())
                        denom = max(1.0, float((n - 2) ** 2))  # order of magnitude of the number of intermediate drones
                        var_norm = var / denom
                        uniform_score = 1.0 / (1.0 + var_norm)
                        uniform_reward = float(uniform_score)
        else:
            nn_reward -= 100.0

        # -------------------------
        # Penalty for drones/links in forbidden zones (proportional)
        # -------------------------
        drones_in_zone = 0
        links_cross_zone = 0
        zone_fix_bonus = 0.0
        zone_clean_once_bonus = 0.0
        # Local endpoint cache (x1,y1,x2,y2) to speed up zone-crossing checks
        zone_edge_cache = {}
        if self.zonas_proibidas:
            # counts drones in the zone via the boolean grid when available
            if self._forbidden_grid is not None:
                drones_in_zone = int(self._forbidden_grid[pos_round[:n, 0], pos_round[:n, 1]].sum())
            else:
                for i in range(n):
                    if (int(pos_round[i, 0]), int(pos_round[i, 1])) in self.zonas_proibidas:
                        drones_in_zone += 1

            # counts links crossing zones iterating only over existing edges
            if num_links > 0:
                for i, j in zip(ei.tolist(), ej.tolist()):
                    x1 = int(pos_round[i, 0]); y1 = int(pos_round[i, 1])
                    x2 = int(pos_round[j, 0]); y2 = int(pos_round[j, 1])
                    k = (x1, y1, x2, y2)
                    v = zone_edge_cache.get(k)
                    if v is None:
                        v = bool(self._linha_passa_por_zona((x1, y1), (x2, y2), (int(i), int(j))))
                        zone_edge_cache[k] = v
                    if v:
                        links_cross_zone += 1

        zone_penalty = 0.0
        # zone_penalty = -ZONE_DRONE_COEF*drones_in_zone - ZONE_LINK_COEF*links_cross_zone
        if drones_in_zone > 0:
            zone_penalty -= float(w["ZONE_DRONE_COEF"]) * float(drones_in_zone) * float(drone_scale)
        if links_cross_zone > 0:
            zone_penalty -= float(w["ZONE_LINK_COEF"]) * float(links_cross_zone) * float(link_scale_eff)

        # -------------------------
        # Shaping of zone: reward for REMOVING violations (delta) + one-time bonus when clean
        # -------------------------
        zone_fix_bonus = 0.0
        prev_links = self._prev_links_cross_zone
        prev_drones = self._prev_drones_in_zone
        if prev_links is None:
            prev_links = links_cross_zone
        if prev_drones is None:
            prev_drones = drones_in_zone

        d_links = int(prev_links) - int(links_cross_zone)
        d_drones = int(prev_drones) - int(drones_in_zone)
        if d_links > 0:
            zone_fix_bonus += float(w.get("ZONE_FIX_LINK_BONUS", 0.0)) * float(d_links) * float(link_scale_eff)
        if d_drones > 0:
            zone_fix_bonus += float(w.get("ZONE_FIX_DRONE_BONUS", 0.0)) * float(d_drones) * float(drone_scale)

        zone_clean_once_bonus = 0.0
        if (drones_in_zone == 0 and links_cross_zone == 0):
            if not getattr(self, "_zone_clean_once", False):
                zone_clean_once_bonus = float(w.get("ZONE_CLEAN_ONCE_BONUS", 0.0))
                self._zone_clean_once = True
        else:
            self._zone_clean_once = False

        self._prev_links_cross_zone = int(links_cross_zone)
        self._prev_drones_in_zone = int(drones_in_zone)
        # -------------------------
        # Line alignment (drone0 -> droneN-1): pulls intermediate drones close to the line
        # line_align_score ∈ [0,1] (1 = well aligned)
        # line_align_term = LINE_ALIGN_W * gate_zone * line_align_score
        # -------------------------
        line_align_score = 0.0
        line_align_term = 0.0
        LINE_ALIGN_W = float(w.get("LINE_ALIGN_W", 0.0))
        if LINE_ALIGN_W > 0.0 and n > 2:
            p0_line = pos_np[0, :2]
            p1_line = pos_np[n - 1, :2]
            v_line = p1_line - p0_line
            vv_line = float(v_line[0] * v_line[0] + v_line[1] * v_line[1])
            if vv_line > 1e-9:
                dperp = []
                for i in range(1, n - 1):
                    p = pos_np[i, :2]
                    t = float(((p - p0_line) @ v_line) / vv_line)
                    t = max(0.0, min(1.0, t))
                    proj = p0_line + t * v_line
                    dp = float(np.linalg.norm(p - proj))  # in "cells"
                    dperp.append(dp)

                if dperp:
                    mean_dp = float(np.mean(dperp))
                    # normalizes by approximately half the grid width (order of magnitude)
                    mean_dp_norm = min(1.0, mean_dp / max(1.0, self.grid_size * 0.5))
                    line_align_score = 1.0 - mean_dp_norm
                    # does not compete with the huge zone penalty: only rewards when there are no violations
                    gate_zone = 1.0 if (drones_in_zone == 0 and links_cross_zone == 0) else 0.0
                    line_align_term = LINE_ALIGN_W * gate_zone * float(line_align_score)

        # -------------------------
        # Main reliability reward (log-scale at (1-rel))
        # - Uses -log10(1-rel) to provide high sensitivity near 1.0
        # - Makes the agent "feel" improvement starting from RELIABILITY_BASE
        # - rel_term=1.0 at RELIABILITY_TARGET
        # -------------------------
        eps = 1e-12
        rel_clipped = max(eps, min(1.0 - eps, float(reliability)))
        fail = max(eps, 1.0 - rel_clipped)
        fail_b = max(eps, 1.0 - float(RELIABILITY_BASE))
        fail_t = max(eps, 1.0 - float(RELIABILITY_TARGET))

        a = -math.log10(fail_b)
        b = -math.log10(fail_t)
        cur = -math.log10(fail)
        if b <= a + 1e-12:
            rel_term = 1.0 if rel_clipped >= float(RELIABILITY_TARGET) else 0.0
        else:
            rel_term = (cur - a) / (b - a)
        rel_term = max(0.0, min(1.0, float(rel_term)))

        # rel_score (per step) uses REL_STEP (default 0 => large reliability term only at TERMINAL)
        rel_step_w = float(w.get("REL_STEP", 0.0))
        rel_score = rel_step_w * (2.0 * rel_term - 1.0)

        # For gates/reports
        rel_frac = float(rel_term)

        # Soft gate (nearly hard) for secondary terms
        if reliability >= float(RELIABILITY_UNLOCK_END):
            rel_unlock_gate = 1.0
        elif reliability <= float(RELIABILITY_UNLOCK_START):
            rel_unlock_gate = 0.0
        else:
            rel_unlock_gate = (float(reliability) - float(RELIABILITY_UNLOCK_START)) / max(1e-12, (
                        float(RELIABILITY_UNLOCK_END) - float(RELIABILITY_UNLOCK_START)))
            rel_unlock_gate = max(0.0, min(1.0, rel_unlock_gate))

        # Penalty for missing path 0->N-1
        path_penalty = 0.0
        if not path_exists:
            path_penalty -= float(w["PATH_MISSING"])

        # Hard penalty proportional to zone violations
        # GATE: when no path 0->N-1 exists, do NOT apply the hard penalty
        # (allows the agent to explore links/positions to create connectivity and only then "clean" the zone)
        # gate ~0 up to reach_frac~0.6, rises smoothly and reaches 1 near 0.9
        zone_hard_gate = np.clip((reach_frac - 0.6) / (0.9 - 0.6), 0.0, 1.0)

        zone_hard_penalty = 0.0
        if zone_hard_gate > 0.0 and (drones_in_zone > 0 or links_cross_zone > 0):
            total_violacoes_zona = float(drones_in_zone) * float(drone_scale) + float(links_cross_zone) * float(link_scale_eff)
            zone_hard_penalty -= float(w["ZONE_HARD"]) * total_violacoes_zona * float(zone_hard_gate)

        # -------------------------
        # Hops: count only AFTER the main objectives
        # -------------------------
        comp_rel = rel_frac
        comp_path = 1.0 if path_exists else 0.0
        comp_zone = 1.0 if (drones_in_zone == 0 and links_cross_zone == 0) else 0.0
        comp_left = 1.0 if (left_links_count == 0) else 0.0
        comp_dag = 1.0 if is_dag else 0.0

        # Gate of hops: optimizes hops only after the primary objectives are good
        primary_ok = (comp_rel + comp_path + comp_zone + comp_left + comp_dag) / 5.0
        hops_gate = primary_ok ** 2

        # Shortcut: we use the minimum number of hops (shortest path in hops) to avoid penalizing redundancy.
        # max_hops continues to be logged/measured separately.
        min_hops = 0
        if path_exists:
            parent_min = [-1] * n
            q_min = deque([0])
            parent_min[0] = 0
            while q_min:
                u = q_min.popleft()
                if u == n - 1:
                    break
                for v in range(n):
                    if adj_np[u, v] > 0.0 and parent_min[v] == -1:
                        parent_min[v] = u
                        q_min.append(v)
            if parent_min[n - 1] != -1:
                cur = n - 1
                while cur != 0:
                    cur = parent_min[cur]
                    min_hops += 1

        if path_exists and min_hops > 0:
            # Directly optimizes the MINIMUM number of hops (1 is the best possible)
            hop_norm = (float(min_hops) - 1.0) / max(1.0, float(n - 1))
            hop_norm = max(0.0, min(1.0, hop_norm))
            hop_term = 1.0 - 2.0 * hop_norm  # ∈ [-1, +1]
            hops_score = float(w["HOPS_WEIGHT"]) * float(hops_gate) * float(hop_term)
        # -------------------------
        # Layout along the path 0->N-1 – VERY small bonus
        # layout_score uses weights w["LAYOUT_*"]
        # -------------------------
        layout_score = 0.0
        layout_gate = primary_ok ** 3

        if path_exists:
            parent = [-1] * n
            q = deque([0])
            parent[0] = 0
            while q:
                u = q.popleft()
                if u == n - 1:
                    break
                for v in range(n):
                    if adj_np[u, v] > 0.0 and parent[v] == -1:
                        parent[v] = u
                        q.append(v)

            if parent[n - 1] != -1:
                path_nodes = []
                cur = n - 1
                while True:
                    path_nodes.append(cur)
                    if cur == 0:
                        break
                    cur = parent[cur]
                path_nodes = path_nodes[::-1]

                center_x = self.grid_size / 2.0
                x_coords = pos_np[path_nodes, 0]
                y_coords = pos_np[path_nodes, 1]

                dx_center = np.abs(x_coords - center_x)
                max_dx = self.grid_size / 2.0
                align_score = 1.0 - np.clip(dx_center.mean() / max_dx, 0.0, 1.0)

                if len(path_nodes) >= 2:
                    y0 = y_coords[0]
                    yN = y_coords[-1]
                    t = np.linspace(0.0, 1.0, len(path_nodes))
                    y_line = y0 + t * (yN - y0)
                    dev_y = np.abs(y_coords - y_line).mean()
                    max_dev_y = max(1.0, self.grid_size / 3.0)
                    straight_score = 1.0 - np.clip(dev_y / max_dev_y, 0.0, 1.0)
                else:
                    straight_score = 1.0

                out_degrees_path = adj_np.sum(axis=1)[path_nodes]
                redundant_frac = float(np.mean(out_degrees_path >= 2.0))

                layout_score = float(layout_gate) * (
                    float(w["LAYOUT_ALIGN_W"]) * float(align_score)
                    + float(w["LAYOUT_STRAIGHT_W"]) * float(straight_score)
                    + float(w["LAYOUT_REDUND_W"]) * float(redundant_frac)
                )

        # -------------------------
        # -------------------------
        # Disjoint paths (edge-disjoint):
        # - FULL: complete edge-disjoint paths 0->(n-1) (only if there are NO links crossing zones)
        # - PARTIAL: partial parallel routes (edge-disjoint) starting from node 0 toward (n-1),
        # even if they do not yet reach the destination.
        # Bonus:
        #   disjoint_bonus = DISJOINT_PATH_BONUS * [(full-1) + DISJOINT_PARTIAL_FACTOR*(partial-1)]
        # -------------------------
        disjoint_paths_full = 0
        disjoint_paths_partial = 0
        disjoint_bonus = 0.0
        # Filters edges used no Bonus of Disjoint paths by link quality (minimum probability).
        # This prevents the bonus from encouraging very long/poor edges (ex.: direct link 0->N-1).
        disjoint_p_min = float(w.get("DISJOINT_MIN_P", 0.0))
        dist_m = self._calc_distance_matrix()
        adj_disjoint = self._filter_adj_by_disjoint_quality(adj_np, dist_m, disjoint_p_min)

        # FULL (keeps the old rule)
        if path_exists:
            disjoint_paths_full = self._max_edge_disjoint_paths_no_zone_links(adj_disjoint, pos_np, 0, n - 1, pos_round=pos_round, zone_edge_cache=zone_edge_cache)
            if disjoint_paths_full > 1:
                disjoint_bonus += float(w.get("DISJOINT_PATH_BONUS", 0.0)) * min(float(disjoint_paths_full - 1), 10.0)

        # PARTIAL (new)
        partial_factor = float(w.get("DISJOINT_PARTIAL_FACTOR", 0.5))
        min_prog = float(w.get("DISJOINT_PARTIAL_MIN_PROGRESS", 0.5))
        disjoint_paths_partial = self._max_edge_disjoint_partial_paths(adj_disjoint, pos_np, 0, n - 1,
                                                                       min_progress=min_prog)
        if disjoint_paths_partial > 1 and partial_factor > 0.0:
            disjoint_bonus += float(w.get("DISJOINT_PATH_BONUS", 0.0)) * float(partial_factor) * min(
                float(disjoint_paths_partial - 1), 10.0)

        # Compatibility: metric "disjoint_paths" now reflects the better of the two (FULL vs PARTIAL)
        disjoint_paths = int(max(disjoint_paths_full, disjoint_paths_partial))
        # Total fitness
        # -------------------------
        layout_terms = float(w["LAYOUT_MIX"]) * (float(nn_reward) + float(uniform_reward)) + float(layout_score)

        # Comments preceding each term: value/range and associated weight.
        # -------------------------
        # Final gate (nearly hard) for secondary terms
        # Unlocks only when: Reliability >= ~0.9999 (soft), a path exists and the zone is clean.
        # -------------------------
        secondary_gate = float(rel_unlock_gate)
        if not path_exists:
            secondary_gate = 0.0
        if (drones_in_zone > 0) or (links_cross_zone > 0):
            secondary_gate = 0.0

        hops_score *= secondary_gate
        latency_reward *= secondary_gate
        layout_terms *= secondary_gate
        line_align_term *= secondary_gate
        # Redundancy bonus (Disjoint paths) must not depend on the secondary-objective gate,
        # because it helps push reliability into the ultra-high region.
        # Unlocks gradually starting from 0.90 up to 0.9999 (fine-tune as needed).
        # Parallel-route bonus gate: early activation based on useful connectivity,
        # avoiding dependence on being close to 5-nines.
        disjoint_gate = 0.0
        if path_exists and disjoint_bonus != 0.0:
            rff0 = float(w.get("DISJOINT_GATE_RFF_START", 0.85))
            rf0 = float(w.get("DISJOINT_GATE_RF_START", 0.85))
            gate_rff = 0.0 if reach_from_first_frac <= rff0 else (float(reach_from_first_frac) - rff0) / max(1e-9, (
                        1.0 - rff0))
            gate_rf = 0.0 if reach_frac <= rf0 else (float(reach_frac) - rf0) / max(1e-9, (1.0 - rf0))
            disjoint_gate = max(0.0, min(1.0, gate_rff)) * max(0.0, min(1.0, gate_rf))

        disjoint_bonus_raw = float(disjoint_bonus)
        disjoint_bonus = disjoint_bonus_raw * float(disjoint_gate)

        fit = (
            # rel_score = REL_MAIN * (2*rel_log - 1), rel_log ∈ [0,1] => rel_score ∈ [-REL_MAIN, +REL_MAIN]
                rel_score
                # path_penalty = -PATH_MISSING if there is no path, otherwise 0
                + path_penalty
                # zone_hard_penalty = -ZONE_HARD * (drones_in_zone + links_cross_zone)
                + zone_hard_penalty
                # cycle_penalty = -CYCLE_PENALTY if there is a cycle in the graph
                + cycle_penalty
                # conn_quality_term = CONN_QUALITY_COEF * connection_quality, connection_quality ∈ [0,1]
                + conn_quality_term
                # reach_last_penalty = -(1 - reach_frac) ∈ [-1, 0] (peso=1)
                + reach_last_penalty
                # unreachable_from_first_penalty = -UNREACHABLE_FROM_FIRST_PENALTY * (#unreachable nodes of the 0)
                + unreachable_from_first_penalty
                # reach_from_first_penalty = -REACH_FROM_FIRST_W * (1 - reach_from_first_frac)
                + reach_from_first_penalty
                # invalid_relay_penalty: sum of -ISOLATED_NODE_PENALTY per node isolated (weight=1)
                + invalid_relay_penalty
                # blocked_relay_penalty: internal penalties (0.2 per occurrence) (weight=1)
                + blocked_relay_penalty * 100
                # imbalance_penalty: -0.2 * |in_deg - out_deg| accumulated (weight=1)
                + imbalance_penalty * 100
                # dead_end_penalty: -0.2 by dead-end accumulated (weight=1)
                + dead_end_penalty * 100
                # hops_score = HOPS_WEIGHT * hops_gate * hop_diff, hop_diff ∈ [-1,1], hops_gate ∈ [0,1]
                + hops_score
                # latency_penalty: ≤ 0, proportional to the amount above MAX_LATENCY_MS (weight=1)
                + latency_penalty
                # latency_reward: ≥ 0, proportional to the margin below MAX_LATENCY_MS (weight=1)
                + latency_reward
                # layout_terms = LAYOUT_MIX*(nn_reward + uniform_reward) + layout_score
                + layout_terms
                # line_align_term = LINE_ALIGN_W*gate_zone*line_align_score (pulls toward the line 0->N-1)
                + line_align_term
                # zone_penalty = -ZONE_DRONE_COEF*drones_in_zone - ZONE_LINK_COEF*links_cross_zone
                + zone_penalty
                + zone_fix_bonus
                + zone_clean_once_bonus
                # left_links_penalty = -LEFT_LINK_COEF * left_links_count (disabled by default: LEFT_LINK_COEF=0.0)
                + left_links_penalty
                # throughput_penalty = -THROUGHPUT_PENALTY_MAX * excess, excess∈[0,1] when throughput/THROUGHPUT_MAX_MBPS >= 0.85 (disabled by default)
                + throughput_penalty
                # bottleneck_term: focuses on improving the worst link on path 0->N-1
                + bottleneck_term
                # disjoint_bonus (gated)
                + disjoint_bonus
        )
        info = {
            "reliability": float(reliability),
            "bottleneck_min_p": float(bottleneck_min_p),
            "bottleneck_max_dist_m": float(bottleneck_max_dist_m),
            "throughput": float(throughput),
            "num_links": int(num_links),
            "num_links_eff": int(num_links_eff) if 'num_links_eff' in locals() else int(num_links),
            "Emax_n": int(Emax_n),
            "Emax_ref": int(Emax_ref),
            "link_scale": float(link_scale),
            "link_density": float(link_density),
            "link_scale_eff": float(link_scale_eff),
            "drone_scale": float(drone_scale),
            "relay_scale": float(relay_scale),
            "connection_quality": float(connection_quality),
            "reach_frac": float(reach_frac),
            "reach_from_first_frac": float(reach_from_first_frac),
            "unreachable_from_first_count": int(unreachable_from_first_count),
            "reach_from_first_penalty": float(reach_from_first_penalty),
            "path_exists": bool(path_exists),
            "is_dag": int(is_dag),
            "cycle_penalty": float(cycle_penalty),
            "zone_hard_gate": float(zone_hard_gate),
            "max_hops": int(max_hops),
            "avg_hops": float(avg_hops),
            "nn_mean_dist": float(nn_mean_dist),
            "spread_nn_score": float(nn_reward),
            "spread_uniform_score": float(uniform_reward),
            "line_align_score": float(line_align_score),
            "line_align_term": float(line_align_term),
            "drones_in_zone": int(drones_in_zone),
            "links_cross_zone": int(links_cross_zone),
            "zone_fix_bonus": float(zone_fix_bonus),
            "zone_clean_once_bonus": float(zone_clean_once_bonus),
            "left_links_count": int(left_links_count),
            "left_links_penalty": float(left_links_penalty),
            "left_links_gate": int(left_links_gate),
            "throughput_penalty": float(throughput_penalty),
            "num_drones": int(self.num_drones),
            "max_drones": int(self.max_drones),
            "disjoint_paths": int(disjoint_paths),
            "disjoint_paths_full": int(disjoint_paths_full),
            "disjoint_paths_partial": int(disjoint_paths_partial),
            "disjoint_bonus_raw": float(disjoint_bonus_raw),
            "disjoint_bonus": float(disjoint_bonus),
            "disjoint_gate": float(disjoint_gate),
        }

        # Optional: exposes actual fitness contributions in info (useful for TensorBoard/debug)
        if self.log_fit_terms:
            info.update({
                "fit_rel_score": float(rel_score),
                "fit_rel_unlock_gate": float(rel_unlock_gate),
                "fit_secondary_gate": float(secondary_gate),
                "fit_path_penalty": float(path_penalty),
                "fit_zone_hard_penalty": float(zone_hard_penalty),
                "fit_cycle_penalty": float(cycle_penalty),
                "is_dag": int(is_dag),
                "fit_zone_hard_gate": float(zone_hard_gate),
                "fit_conn_quality": float(conn_quality_term),
                "fit_reach_last_penalty": float(reach_last_penalty),
                "fit_unreachable_from_first_penalty": float(unreachable_from_first_penalty),
                "fit_reach_from_first_penalty": float(reach_from_first_penalty),
                "fit_invalid_relay_penalty": float(invalid_relay_penalty),
                "fit_blocked_relay_penalty": float(blocked_relay_penalty),
                "fit_imbalance_penalty": float(imbalance_penalty),
                "fit_dead_end_penalty": float(dead_end_penalty),
                "fit_hops_score": float(hops_score),
                "fit_latency_penalty": float(latency_penalty),
                "fit_latency_reward": float(latency_reward),
                "fit_layout_terms": float(layout_terms),
                "fit_line_align_term": float(line_align_term),
                "fit_zone_penalty": float(zone_penalty),
                "fit_left_links_penalty": float(left_links_penalty),
                "fit_left_links_gate": int(left_links_gate),
                "fit_throughput_penalty": float(throughput_penalty),
                "fit_total": float(fit),
            })

        return float(fit), info

    @staticmethod
    def _reachability_bits(adj_on: np.ndarray) -> np.ndarray:
        """Compute transitive closure (reachability) using bitsets.
Return reach[i] as uint32 with bit j=1 when j is reachable from i.
Assumes n<=32 (n is small in this environment)."""
        n = int(adj_on.shape[0])
        reach = np.zeros(n, dtype=np.uint32)
        # Initial: direct neighbors + self
        for i in range(n):
            bits = np.uint32(1 << i)
            js = np.nonzero(adj_on[i])[0]
            for j in js.tolist():
                bits |= np.uint32(1 << int(j))
            reach[i] = bits
        # Warshall with bitsets
        for k in range(n):
            kbit = np.uint32(1 << k)
            rk = reach[k]
            for i in range(n):
                if (reach[i] & kbit) != 0:
                    reach[i] |= rk
        return reach

    # =========================
    # Action masking (sb3-contrib)
    # =========================
    def action_masks(self) -> np.ndarray:
        """Boolean flattened mask compatible with MultiDiscrete.

Order: concatenate the mask of each MultiDiscrete dimension.
- move_idx: prevents moving inactive drones and anchors (0 and N-1);
- step_code/dir_code: no fine-grained mask (always True);
- toggle_op: allows valid ADD/REMOVE operations in the current state and may apply
  hard masks (zone, left-link, and cycles) to enforce hard constraints."""
        if self._mask_cache is not None and self._mask_cache_version == self._mask_version:
            return self._mask_cache

        nvec = np.asarray(self.action_space.nvec, dtype=np.int64)
        n = int(self.num_drones)

        # --- move_idx ---
        m_move = np.zeros(int(nvec[0]), dtype=bool)
        if n > 0:
            m_move[:min(n, int(nvec[0]))] = True
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

        # Predefinitions to avoid NameError when n<2
        src_all = self._edge_src
        dst_all = self._edge_dst
        idx_add = np.array([], dtype=np.int64)
        pos_np = None

        if n >= 2:
            # CPU->NumPy views (without copying) + vectorized logic: avoids a Python loop per edge.
            adj_on = (_torch_to_numpy_cpu_view(self.adjacency_matrix[:n, :n]) > 0.5)
            out_deg = adj_on.sum(axis=1).astype(np.int32, copy=False)

            src_all = self._edge_src
            dst_all = self._edge_dst
            active = (src_all < n) & (dst_all < n)
            if np.any(active):
                idx_active = np.nonzero(active)[0]
                src_a = src_all[idx_active]
                dst_a = dst_all[idx_active]

                is_on = adj_on[src_a, dst_a]

                # REMOVE: valid if it exists
                idx_rem = idx_active[is_on]
                if idx_rem.size:
                    m_toggle[1 + E + idx_rem] = True

                # ADD: valid if it does NOT exist and respects the hard max_out rule
                add_ok = (~is_on) & (out_deg[src_a] < int(self.max_out_connections))
                idx_add = idx_active[add_ok]
                if idx_add.size:
                    # (mask-relax) does not block left-links or zones
                    m_toggle[1 + idx_add] = True

                    # direct link 0<->N-1 only if p_dir >= threshold (when threshold > 0)
                    p_min_direct = float(self.fit_w.get("DIRECT_0N1_MIN_P", 0.0))
                    if p_min_direct > 0.0 and n > 2:
                        src_add = src_all[idx_add]
                        dst_add = dst_all[idx_add]
                        direct_mask = (
                            ((src_add == 0) & (dst_add == (n - 1))) |
                            ((src_add == (n - 1)) & (dst_add == 0))
                        )
                        if np.any(direct_mask):
                            pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
                            for edge_idx in idx_add[direct_mask].tolist():
                                s = int(src_all[edge_idx])
                                d = int(dst_all[edge_idx])
                                dx_m = (float(pos_np[d, 0]) - float(pos_np[s, 0])) * float(self.square_size_m)
                                dy_m = (float(pos_np[d, 1]) - float(pos_np[s, 1])) * float(self.square_size_m)
                                d_m = float(math.hypot(dx_m, dy_m))
                                p_dir = float(self._prob_link(d_m))
                                if p_dir < p_min_direct:
                                    m_toggle[1 + edge_idx] = False
        # 4b) hard-mask (left-links): blocks ADD when x_dst < x_src
        if self.hard_mask_left_links and idx_add.size:
            if pos_np is None:
                pos_np = _torch_to_numpy_cpu_view(self.drone_positions[:n, :2]).astype(np.float32, copy=False)
            src_add = src_all[idx_add]
            dst_add = dst_all[idx_add]
            left = (pos_np[dst_add, 0] < pos_np[src_add, 0])
            if np.any(left):
                m_toggle[1 + idx_add[left]] = False

        # 4c) hard-mask (cycles): blocks ADD operations that would create a cycle (dst reaches src)
        if self.hard_mask_cycles and idx_add.size:
            enabled = m_toggle[1 + idx_add]
            if np.any(enabled):
                idx_chk = idx_add[enabled]
                src_chk = src_all[idx_chk]
                dst_chk = dst_all[idx_chk]
                reach = self._reachability_bits(adj_on)
                reach_dst = reach[dst_chk]
                cyc = ((reach_dst >> src_chk) & np.uint32(1)).astype(bool)
                if np.any(cyc):
                    m_toggle[1 + idx_chk[cyc]] = False



        # 5) hard mask (zone): blocks ADD operations for edges crossing forbidden zones
        if self.hard_mask_zone_edges and self.zonas_proibidas and idx_add.size:
            cross_mat = self._ensure_edge_cross_mat()
            if cross_mat is not None:
                src_add = src_all[idx_add]
                dst_add = dst_all[idx_add]
                cross = cross_mat[src_add, dst_add].astype(bool, copy=False)
                if np.any(cross):
                    m_toggle[1 + idx_add[cross]] = False

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

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# RLlib MultiAgentEnv (MARLlib is built on Ray/RLlib)
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception as e:  # pragma: no cover
    MultiAgentEnv = object  # type: ignore



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
 - "anchor_0" (drone index 0)
 - "inter_1"... (drone indices 1..N-2)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        # RLlib MultiAgentEnv.__init__ does NOT accept env_config as an argument.
        # Calls such as super().__init__(config) cause:
        #   TypeError: MultiAgentEnv.__init__() takes 1 positional argument but 2 were given
        try:
            super().__init__()
        except TypeError:
            try:
                MultiAgentEnv.__init__(self)  # type: ignore[misc]
            except Exception:
                pass

        config = dict(config or {})

        # Wrapper-only keys (the embedded DroneEnv does not accept)
        episode_horizon = config.pop("episode_horizon", None)
        self.mask_actions = bool(int(config.pop("mask_actions", 1)))

        # Converts zona_w/zona_h -> zona_size (DroneEnv expects zona_size=(w,h))
        if "zona_size" not in config and ("zona_w" in config or "zona_h" in config):
            zw = int(config.pop("zona_w", 15))
            zh = int(config.pop("zona_h", 15))
            config["zona_size"] = (zw, zh)

        # Underlying single-agent environment
        self.base_env = DroneEnv(**config)

        # Applies the horizon to the base environment (Gymnasium-style)
        if episode_horizon is not None:
            try:
                self.base_env._max_episode_steps = int(episode_horizon)
            except Exception:
                pass

            self.max_drones: int = int(getattr(self.base_env, "max_drones", config.get("max_drones", 13)))
            self._E: int = int(getattr(self.base_env, "_E", 0))
            self._edge_pairs: List[Tuple[int, int]] = list(getattr(self.base_env, "_edge_pairs", []))
            self._edge_pair_to_idx: Dict[Tuple[int, int], int] = {(s, d): i for i, (s, d) in enumerate(self._edge_pairs)}

            # Local toggle encoding (per-agent, only src fixed; agent chooses dst + add/remove)
            # 0: noop
            # 1..MAX: add src->dst (dst = a-1)
            # MAX+1..2*MAX: remove src->dst (dst = a-(MAX+1))
            self.TOGGLE_LOCAL_N: int = 1 + 2 * self.max_drones

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

            # For RLlib's space introspection (Defines all possible agents at max size)
            self.possible_agents: List[str] = ["anchor_0"] + [f"inter_{i}" for i in range(1, self.max_drones - 1)]
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
        # Only the source anchor + intermediates act. Sink (N-1) has no outgoing links to manage.
        return ["anchor_0"] + [f"inter_{i}" for i in range(1, n - 1)]

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

        def cell_forbidden(xx: int, yy: int) -> bool:
            if not has_zone:
                return False
            gx = int(getattr(self.base_env, "grid_size", 0) or 0)
            xx = max(0, min(gx - 1, int(xx)))
            yy = max(0, min(gx - 1, int(yy)))
            if forb_grid is not None:
                return bool(forb_grid[xx, yy])
            return (int(xx), int(yy)) in zones

        grid_max = int(getattr(self.base_env, "grid_size", 0) or 0) - 1

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
        if hard_cross:
            p_i_old = (int(round(old_xi)), int(round(old_yi)))
            for j in out_js.tolist():
                j = int(j)
                p_j = (int(round(float(x_all[j]))), int(round(float(y_all[j]))))
                old_cross_out[j] = bool(self.base_env._linha_passa_por_zona(p_i_old, p_j, (src, j)))
            for k in in_ks.tolist():
                k = int(k)
                p_k = (int(round(float(x_all[k]))), int(round(float(y_all[k]))))
                old_cross_in[k] = bool(self.base_env._linha_passa_por_zona(p_k, p_i_old, (k, src)))

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
            for j in out_js.tolist():
                j = int(j)
                new_left = float(x_all[j]) < new_xi
                if new_left and (not old_left_out.get(j, False)):
                    invalid = True
                    break
            if invalid:
                m[move_op] = 0
                continue

            for k in in_ks.tolist():
                k = int(k)
                new_left = new_xi < float(x_all[k])
                if new_left and (not old_left_in.get(k, False)):
                    invalid = True
                    break
            if invalid:
                m[move_op] = 0
                continue

            # NEW zone-crossing on existing incident edges => invalid
            if hard_cross:
                p_i_new = (int(round(new_xi)), int(round(new_yi)))
                for j in out_js.tolist():
                    j = int(j)
                    if old_cross_out.get(j, False):
                        continue
                    p_j = (int(round(float(x_all[j]))), int(round(float(y_all[j]))))
                    if bool(self.base_env._linha_passa_por_zona(p_i_new, p_j, (src, j))):
                        invalid = True
                        break
                if invalid:
                    m[move_op] = 0
                    continue
                for k in in_ks.tolist():
                    k = int(k)
                    if old_cross_in.get(k, False):
                        continue
                    p_k = (int(round(float(x_all[k]))), int(round(float(y_all[k]))))
                    if bool(self.base_env._linha_passa_por_zona(p_k, p_i_new, (k, src))):
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

        base_obs, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(base_action)

        done = bool(base_terminated or base_truncated)
        base_info = dict(base_info or {})
        base_info.update({f"ma/{k}": v for k, v in wrapper_info.items()})
        base_info["ma/acting_agent_idx"] = float(self._turn_idx)

        # ---------- "clean" final metrics (only at terminal) ----------
        if done:
            # Always ensure at least ONE metric *_terminal (for the callback to detect)
            # and to avoid going "silent" if the detailed metrics block fails.
            try:
                rel_fallback = base_info.get(
                    "reliability_terminal",
                    base_info.get("reliability", getattr(self.base_env, "current_reliability", 0.0)),
                )
                base_info["reliability_terminal"] = float(rel_fallback)
            except Exception:
                base_info["reliability_terminal"] = 0.0

            try:
                # Adjacency matrix and positions in the FINAL state
                adj_t = _torch_to_numpy_cpu_view(self.base_env.adjacency_matrix)
                pos_t = _torch_to_numpy_cpu_view(self.base_env.drone_positions)
                n = int(self.base_env.num_drones)

                adj = np.asarray(adj_t)[:n, :n]
                pos = np.asarray(pos_t)[:n, :]

                # active drones (if available)
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

                # DAG (cycle detection in a directed graph)
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

                # zone violations (prefer values already present in DroneEnv info, when available)
                drones_in_zone = int(base_info.get("drones_in_zone", 0))
                links_cross_zone = int(base_info.get("links_cross_zone", 0))
                viol = int(drones_in_zone + links_cross_zone)

                # final reliability (already ensured above)
                rel_final = float(base_info.get("reliability_terminal", getattr(self.base_env, "current_reliability", 0.0)))

                # strict success in the final state (as defined)
                success_strict = bool((viol == 0) and (left_links == 0) and is_dag and path_ok and (rel_final >= float(RELIABILITY_TARGET)))

                base_info["left_links_count_terminal"] = int(left_links)
                base_info["right_links_ratio_terminal"] = float(right_ratio)
                base_info["links_total_terminal"] = int(total_links)
                base_info["is_dag_terminal"] = int(1 if is_dag else 0)
                base_info["path_exists_terminal"] = int(1 if path_ok else 0)
                base_info["success_terminal"] = int(1 if success_strict else 0)
                base_info["zone_violations_terminal"] = int(viol)
                base_info["reliability_terminal"] = float(rel_final)
            except Exception as e:
                # Do not crash the episode due to a metric error; but record a small trace
                # to simplify debugging when logs do not appear.
                base_info["ma/terminal_metrics_exc"] = type(e).__name__

                # Keep terminal metrics only at the end
        base_info = self._strip_terminal_metrics_if_needed(base_info, is_terminal=done)
        if done:
            # Avoids carrying large arrays through the RLlib/Tune/TensorBoard pipeline.
            # Keeps only scalars/strings for logging; full snapshots remain in DroneEnv (single-agent).
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
