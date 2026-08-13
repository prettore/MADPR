# ==== calculaProbLink_otimizado.py ("literature accurate" version) ============
from dataclasses import dataclass
from typing import Optional, Sequence
from functools import lru_cache
import numpy as np
from math import erf, sqrt, log10, pi

# -----------------------------
# Radio parameters (reasonable defaults)
# -----------------------------
@dataclass
class RadioParams:
    pt_dbm: float = 19.54242509439325   # 90 mW - Tx power [dBm]
    gt_db: float = 2.0          # Tx gain [dBi]
    gr_db: float = 2.0          # Rx gain [dBi]
    freq_hz: float = 2.4e9      # frequency [Hz]
    n: float = 2.3              # path-loss exponent
    d0_m: float = 1.0          # reference distance [m]
    pmin_dbm: float = -83.0     # Rx sensitivity [dBm]
    sigma_db: float = 4.0       # log-normal standard deviation [dB]

# -----------------------------
# Path-loss utilities
# -----------------------------
def wavelength_m(freq_hz: float) -> float:
    c = 299_792_458.0
    return c / float(freq_hz)

def pl_d0_db(params: RadioParams) -> float:
    """Path loss in dB at the reference distance (Friis)."""
    lam = wavelength_m(params.freq_hz)
    return 20.0 * log10(4.0 * pi * params.d0_m / lam)

def rx_mean_dbm_at_distance(d_m: float, params: RadioParams) -> float:
    """Mean received power in dBm from the log-distance model."""
    if d_m <= 0:
        d_m = 1e-6
    return (
        params.pt_dbm
        + params.gt_db
        + params.gr_db
        - (pl_d0_db(params) + 10.0 * params.n * log10(d_m / params.d0_m))
    )

# -----------------------------
# p(link) — canonical form (literature)
# -----------------------------
def _prob_link_literature_uncached(d_m: float, params: RadioParams) -> float:
    """P_link(d) = 0.5 * [1 + erf( (mu_rx(d) - Pmin) / (sigma * sqrt(2)) )]"""
    mu = rx_mean_dbm_at_distance(d_m, params)
    z = (mu - params.pmin_dbm) / (params.sigma_db * sqrt(2.0))
    p = 0.5 * (1.0 + erf(z))
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p


# -----------------------------
# p(link) — cached wrapper (same semantics)
# -----------------------------
@lru_cache(maxsize=65536)
def _prob_link_literature_cached(
    d_m: float,
    pt_dbm: float,
    gt_db: float,
    gr_db: float,
    freq_hz: float,
    n_exp: float,
    d0_m: float,
    pmin_dbm: float,
    sigma_db: float,
) -> float:
    params = RadioParams()
    # Populate RadioParams from the scalar values that form the cache key.
    params.pt_dbm = float(pt_dbm)
    params.gt_db = float(gt_db)
    params.gr_db = float(gr_db)
    params.freq_hz = float(freq_hz)
    params.n = float(n_exp)
    params.d0_m = float(d0_m)
    params.pmin_dbm = float(pmin_dbm)
    params.sigma_db = float(sigma_db)
    return _prob_link_literature_uncached(float(d_m), params)


def prob_link_literature(d_m: float, params: RadioParams) -> float:
    """Cached version: returns exactly the same value as the original calculation."""
    return _prob_link_literature_cached(
        float(d_m),
        float(getattr(params, 'pt_dbm', RadioParams.pt_dbm)),
        float(params.gt_db),
        float(params.gr_db),
        float(params.freq_hz),
        float(params.n),
        float(params.d0_m),
        float(params.pmin_dbm),
        float(params.sigma_db),
    )

# Reliability from node 0 to node N-1
# Reliability from node 0 to node N-1
# -----------------------------
def _dist_matrix_from_positions(pos_xy: np.ndarray, square_size_m: float, adj: Optional[np.ndarray] = None) -> np.ndarray:
    """pos_xy: (N,2) in grid coordinates (cells). Multiplied by square_size_m to obtain meters.
If adj is provided, distances are computed only where adj>0 (a small optimization)."""
    coords_m = pos_xy.astype(float) * float(square_size_m)
    n = coords_m.shape[0]
    if adj is None:
        # full matrix
        dif = coords_m[:, None, :] - coords_m[None, :, :]
        return np.sqrt((dif ** 2).sum(axis=2))
    else:
        dist = np.zeros((n, n), dtype=float)
        src, dst = np.where(adj > 0)
        dx = coords_m[dst, 0] - coords_m[src, 0]
        dy = coords_m[dst, 1] - coords_m[src, 1]
        dist[src, dst] = np.sqrt(dx * dx + dy * dy)
        return dist

def _topo_order_or_none(adj_bin: np.ndarray) -> Optional[Sequence[int]]:
    """Returns a topological order if the graph is a DAG; otherwise, returns None.

Optimization (without changing semantics): avoids building a graph with networkx.
Uses a simple Kahn algorithm (N is small in this scenario)."""
    n = int(adj_bin.shape[0])
    indeg = adj_bin.sum(axis=0).astype(int).tolist()
    Q = [i for i in range(n) if indeg[i] == 0]
    order = []
    while Q:
        u = Q.pop()
        order.append(u)
        row = adj_bin[u]
        for v in range(n):
            if row[v] > 0:
                indeg[v] -= 1
                if indeg[v] == 0:
                    Q.append(v)
    if len(order) == n:
        return order
    return None


def calcula_confiabilidade_iterativa(
    adjacency: np.ndarray,
    positions_xy: np.ndarray,
    square_size_m: float,
    params: Optional[RadioParams] = None,
    max_iter_cycles: int = 200,
    tol_cycles: float = 1e-9,
) -> float:
    """Transmission reliability between the first node (0) and the last node (N-1),
combining links as *independent* random variables.

- For a DAG: exact computation in topological order.
- For graphs with cycles: fixed-point iteration on F (failure probability).

adjacency: NxN matrix (entries >0 are treated as existing links)
positions_xy: Nx2 coordinates in cells
square_size_m: meters per cell
params: RadioParams (uses defaults if None)"""
    if params is None:
        params = RadioParams()

    adj_bin = (adjacency > 0).astype(np.uint8)
    n = adj_bin.shape[0]
    if n == 0:
        return 0.0

    # precompute distances only where a link exists
    dist = _dist_matrix_from_positions(positions_xy, square_size_m, adj_bin)

    # predecessors for each v
    preds = [np.where(adj_bin[:, v] > 0)[0] for v in range(n)]

    # local function for p(link)
    def p_uv(u, v):
        d = float(dist[u, v])
        return prob_link_literature(d, params)

    # ---- DAG case: DP in topological order (exact)
    topo = _topo_order_or_none(adj_bin)
    if topo is not None:
        F = np.zeros(n, dtype=float)  # failure probability up to v
        F[0] = 0.0
        # Optional: if there is no path from node 0 to node N-1, return 0
        # (F[n-1]=1 => rel=0)
        for v in topo:
            if v == 0:
                continue
            Fv = 1.0
            for u in preds[v]:
                # F_v_via_u = 1 - (1 - F[u]) * p(u->v)
                Fv *= (1.0 - (1.0 - F[u]) * p_uv(u, v))
            F[v] = Fv
        rel = 1.0 - F[n - 1]
        if rel < 0.0: rel = 0.0
        if rel > 1.0: rel = 1.0
        return rel

    # ---- Cyclic-graph case: fixed-point iteration on F
    # Initialize F=1 (pessimistic) except at the source
    F = np.ones(n, dtype=float)
    F[0] = 0.0

    def update_F(F_old: np.ndarray) -> np.ndarray:
        F_new = F_old.copy()
        for v in range(1, n):
            Fv = 1.0
            for u in preds[v]:
                Fv *= (1.0 - (1.0 - F_old[u]) * p_uv(u, v))
            F_new[v] = Fv
        return F_new

    for _ in range(max_iter_cycles):
        F_next = update_F(F)
        if np.max(np.abs(F_next - F)) <= tol_cycles:
            F = F_next
            break
        F = F_next

    rel = 1.0 - F[n - 1]
    if rel < 0.0: rel = 0.0
    if rel > 1.0: rel = 1.0
    return rel

# =============================================================================
# End of module
# =============================================================================
