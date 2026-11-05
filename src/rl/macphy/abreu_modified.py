from __future__ import annotations

"""Abreu-style analytical solver with a paper-aligned single-path core.

Design goals of this module:
- keep the *single-path* / unidirectional chain as close as practical to the
  two-level framework described by Begin/Baynat/Guérin-Lassous/Abreu;
- keep the *multi-path* logic as an external aggregation layer on top of the
  single-path solver;
- expose stable public functions already used by the environment.

The implementation below follows the paper structure in the places where the
article gives explicit formulas:
- high level: tandem of M/M/1/K queues;
- queue throughput: X_n = mu_n * (1 - pi_n(0));
- utilization: U_n = 1 - pi_n(0);
- overflow probability: P_n = pi_n(K_n);
- frame-loss probability: f_n = FER_n + COL_n - FER_n * COL_n;
- CTMC service-time recursion: S_n = t_1 + f_n (t_2 + ... + f_n t_m);
- per-line time: t_k = DIFS + (W_k / 2) * r_n + T_n;
- loop time: r_n = t_slot * (1 + beta_n / gamma_n);
- mean backoff duration B_n (Eq. 21 style) from the exact-attempt
  probabilities phi_{k,n} and mean number of frame transmissions Phi_n.

What remains heuristic / extrapolated:
- the paper leaves COL_n, beta_n and gamma_n to configuration-specific
  formulas; this module implements a chain-oriented version faithful to the
  case studies: visible contenders are nodes within two hops (carrier-sense
  range covering 1-hop and 2-hop neighbors), while hidden contenders are
  modeled separately and only affect collisions, not backoff freezing;
- multi-path coupling remains an external extension.
"""

from dataclasses import dataclass
from typing import List, Sequence, Any
import math
import numpy as np
import torch

try:
    from scipy.stats import ncx2
except Exception:  # pragma: no cover - optional runtime fallback
    ncx2 = None


@dataclass
class PathMetrics:
    path: List[int]
    offered_rate_mbps: float
    effective_rate_mbps: float
    reliability: float
    loss_rate: float
    delay_ms: float
    goodput_mbps: float
    max_feasible_rate_mbps: float
    interference_load: float
    rho_max: float
    debug: dict[str, Any] | None = None
    reliability_block: float = 0.0
    delivery_ratio: float = 0.0
    reliability_hybrid: float = 0.0


@dataclass
class RedundantMetrics:
    paths: List[PathMetrics]
    reliability: float
    loss_rate: float
    delay_ms: float
    goodput_mbps: float
    max_feasible_rate_mbps: float
    omega: np.ndarray
    reliability_block: float = 0.0
    delivery_ratio: float = 0.0
    reliability_hybrid: float = 0.0


@dataclass
class FlowRequest:
    paths: List[List[int]]
    offered_rate_mbps: float
    redundancy_mode: str = "single_best"
    max_active_paths: int | None = None
    ber_per_path: Sequence[Sequence[float]] | None = None
    phy_rates_per_path: Sequence[Sequence[float]] | None = None
    label: str = ""
    per_path_offered_rates_mbps: Sequence[float] | None = None


@dataclass
class ActivePathMetrics:
    path_index: int
    active: bool
    offered_rate_mbps: float
    effective_rate_mbps: float
    physical_load_scale: float
    path_metrics: PathMetrics


@dataclass
class FlowMetrics:
    label: str
    offered_rate_mbps: float
    redundancy_mode: str
    active_path_indices: List[int]
    path_metrics: List[ActivePathMetrics]
    reliability: float
    loss_rate: float
    delay_ms: float
    goodput_mbps: float
    max_feasible_rate_mbps: float
    rho_max: float
    reliability_block: float = 0.0
    delivery_ratio: float = 0.0
    reliability_hybrid: float = 0.0


@dataclass
class MultiFlowMetrics:
    flows: List[FlowMetrics]
    total_goodput_mbps: float
    min_flow_reliability: float
    mean_flow_reliability: float
    weighted_flow_reliability: float
    probability_all_flows_succeed: float
    probability_at_least_one_flow_succeeds: float
    worst_delay_ms: float
    max_rho: float
    omega: np.ndarray

    @property
    def flow_metrics(self) -> List[FlowMetrics]:
        return self.flows


@dataclass
class HopParams:
    frame_bits: int
    phy_rate_mbps: float
    ber: float
    slot_time_us: float
    difs_us: float
    sifs_us: float
    ack_us: float
    ack_timeout_us: float
    cw_min: int
    cw_max: int
    max_attempts: int
    sender_xy: tuple[float, float] = (0.0, 0.0)
    receiver_xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class QueueState:
    lam_pps: float
    mu_pps: float
    X_pps: float
    U: float
    P_block: float
    Q_avg: float
    Dq_ms: float


@dataclass
class MacState:
    fer: float
    p_coll: float
    frame_loss: float
    alpha: float
    beta: float
    gamma: float
    r_slot_us: float
    service_us: float
    phi_exact: np.ndarray
    phi_reach: np.ndarray
    phi_mean_frames: float
    b_mean_slots: float
    b_mean_us: float
    np_freezes: float
    collision_hidden: float
    collision_simultaneous: float
    busy_mean_us: float


@dataclass
class LinkModelParams:
    # Parâmetros ancorados em Goddemeier & Wietfeld (2015), "Investigation of
    # Air-to-Air Channel Characteristics and a UAV Specific Extension to the Rice
    # Model", IEEE Globecom Workshops — medições A2A IEEE 802.11 a 2,4 GHz,
    # altitudes <50 m (cenário do trabalho: drones a 30-50 m).
    pt_dbm: float = 20.0          # paper: 20 dBm de potência de transmissão
    gt_db: float = 5.0            # antena omnidirecional 5 dBi (Goddemeier 2015)
    gr_db: float = 5.0            # antena omnidirecional 5 dBi (Goddemeier 2015)
    freq_hz: float = 2.4e9
    # Expoente de path loss = 2,6: valor EMPÍRICO medido pelo paper (Seção IV,
    # coeficiente de propagação de 2,6 no ajuste de Friis). Corrige o 2,15
    # anterior, que era otimista (free-space puro). Nota: o paper descreve a
    # atenuação como "free-space-like" (lei de potência única, sem breakpoint),
    # mas o expoente ajustado é 2,6, não 2,0.
    path_loss_exp: float = 2.6
    d0_m: float = 1.0
    # Average-SNR model (Friis/log-distance) + Rician fading outage
    noise_floor_dbm: float = -92.82405078805685
    snr_threshold_db: float = 15.202126093501773
    snr_gap_db: float = 0.0
    # Rician K a 40 m = 10,05 (linear) medido pelo paper (Seção IV: K = 3,533 /
    # 10,120 / 10,048 para 10/25/40 m; estabiliza entre 25-40 m). 10·log10(10,05)
    # ≈ 10,0 dB. Substitui o 3,72 dB anterior, que correspondia a ~10 m de
    # altitude (errada). K maior = canal mais dominado por LOS / mais estável.
    rician_k_db: float = 10.02
    # Optional extra margin / implementation knobs
    shadow_margin_db: float = 0.20950616399203986
    succ_floor: float = 1e-4
    succ_cap: float = 1.0 - 1e-4
    # Keep a mild optional long-hop attenuation knob for future tuning;
    # disabled by default in the Rician model.
    long_hop_knee_m: float = 211.73258874371447
    long_hop_scale_m: float = 619.8206935811197
    long_hop_tail_power: float = 1.7
    long_hop_tail_strength: float = 0.1223698259620352

def _gpu_backend_enabled(use_torch_backend: bool = False, torch_device: str | None = None) -> bool:
    if not bool(use_torch_backend):
        return False
    if not torch.cuda.is_available():
        return False
    if torch_device is None:
        return True
    return str(torch_device).startswith("cuda")


def _resolve_torch_device(torch_device: str | None = None) -> torch.device:
    if torch_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(str(torch_device))
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_ber_per_link_from_geometry_torch(
    path: Sequence[int],
    pos_m: np.ndarray,
    frame_bits: int,
    link_model: LinkModelParams | None = None,
    torch_device: str | None = None,
) -> List[float]:
    p = list(map(int, path))
    if len(p) < 2:
        return []
    dev = _resolve_torch_device(torch_device)
    pos_t = torch.as_tensor(np.asarray(pos_m, dtype=np.float64), dtype=torch.float64, device=dev)
    idx_u = torch.as_tensor(p[:-1], dtype=torch.long, device=dev)
    idx_v = torch.as_tensor(p[1:], dtype=torch.long, device=dev)
    d = torch.linalg.norm(pos_t.index_select(0, idx_u)[:, :2] - pos_t.index_select(0, idx_v)[:, :2], dim=1)

    lm = link_model if link_model is not None else LinkModelParams()
    d0 = max(float(lm.d0_m), 1e-9)
    d = torch.clamp(d, min=d0)
    f_mhz = max(1e-6, float(lm.freq_hz) / 1e6)
    pl0 = 32.44 + 20.0 * math.log10(d0 / 1000.0) + 20.0 * math.log10(f_mhz)
    prx_dbm = (
        float(lm.pt_dbm + lm.gt_db + lm.gr_db - pl0)
        - 10.0 * float(lm.path_loss_exp) * torch.log10(d / d0)
    )
    avg_snr_db = prx_dbm - float(lm.noise_floor_dbm) - float(lm.shadow_margin_db)
    # Mantém aproximação conservadora já usada como fallback quando SciPy não está disponível.
    K = float(10.0 ** (float(lm.rician_k_db) / 10.0))
    gamma_bar = torch.pow(10.0, avg_snr_db / 10.0)
    gamma_th = float(10.0 ** ((float(lm.snr_threshold_db) + float(lm.snr_gap_db)) / 10.0))
    ratio = gamma_th / torch.clamp(gamma_bar, min=1e-18)
    mu = 1.0
    sigma = math.sqrt((1.0 + 2.0 * K) / ((1.0 + K) ** 2))
    z = (ratio - mu) / max(1e-9, sigma)
    p_out = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    p_succ = 1.0 - torch.clamp(p_out, 0.0, 1.0)
    lo = float(max(0.0, min(1.0, lm.succ_floor)))
    hi = float(max(lo, min(1.0, lm.succ_cap)))
    if float(getattr(lm, 'long_hop_tail_strength', 0.0)) > 0.0:
        knee = float(max(float(lm.d0_m), getattr(lm, 'long_hop_knee_m', 220.0)))
        scale = float(max(1e-9, getattr(lm, 'long_hop_scale_m', 520.0)))
        power = float(max(1.0, getattr(lm, 'long_hop_tail_power', 1.7)))
        strength = float(max(0.0, getattr(lm, 'long_hop_tail_strength', 0.0)))
        mask = d > knee
        if bool(mask.any()):
            ratio_tail = torch.clamp((d - knee) / scale, min=0.0)
            tail = torch.exp(-strength * torch.pow(ratio_tail, power))
            p_succ = torch.where(mask, lo + (p_succ - lo) * tail, p_succ)
    p_succ = torch.clamp(p_succ, lo, hi)
    ps = p_succ.detach().cpu().numpy().tolist()
    return [_link_ber_from_success_prob(float(s), frame_bits) for s in ps]


def build_interference_matrix_torch(
    paths: Sequence[Sequence[int]],
    pos_m: np.ndarray,
    carrier_sense_range_m: float,
    shared_edge_weight: float = 0.40,
    shared_node_weight: float = 0.20,
    cs_weight: float = 0.10,
    path_owner: Sequence[int] | None = None,
    same_flow_discount: float = 0.5660007226003583,
    torch_device: str | None = None,
) -> np.ndarray:
    k = len(paths)
    omega = np.zeros((k, k), dtype=np.float32)
    if k <= 1:
        return omega
    owners = None if path_owner is None else list(map(int, path_owner))
    dev = _resolve_torch_device(torch_device)
    pos_t = torch.as_tensor(np.asarray(pos_m, dtype=np.float32), dtype=torch.float32, device=dev)
    senders = []
    edge_sets = []
    node_sets = []
    for path in paths:
        p = list(map(int, path))
        edge_sets.append({(p[i], p[i + 1]) for i in range(max(0, len(p) - 1))})
        node_sets.append(set(p[1:-1]))
        senders.append(p[:-1])
    for a in range(k):
        for b in range(a + 1, k):
            ea = edge_sets[a]; eb = edge_sets[b]
            na = node_sets[a]; nb = node_sets[b]
            edge_overlap = (len(ea & eb) / max(1, min(len(ea), len(eb)))) if ea and eb else 0.0
            node_overlap = (len(na & nb) / max(1, min(len(na), len(nb)))) if na and nb else 0.0
            ta = senders[a]; tb = senders[b]
            cs_overlap = 0.0
            if ta and tb:
                ia = torch.as_tensor(ta, dtype=torch.long, device=dev)
                ib = torch.as_tensor(tb, dtype=torch.long, device=dev)
                pa = pos_t.index_select(0, ia)[:, :2]
                pb = pos_t.index_select(0, ib)[:, :2]
                d = torch.cdist(pa, pb, p=2)
                hits = (d <= float(carrier_sense_range_m)).sum().item()
                # transmissor idêntico também conta
                cs_overlap = float(hits) / float(max(1, pa.shape[0] * pb.shape[0]))
            w = shared_edge_weight * edge_overlap + shared_node_weight * node_overlap + cs_weight * cs_overlap
            if owners is not None and a < len(owners) and b < len(owners) and owners[a] == owners[b]:
                w *= float(max(0.0, min(1.0, same_flow_discount)))
            omega[a, b] = omega[b, a] = float(max(0.0, min(1.0, w)))
    return omega

    succ_cap: float = 1.0 - 1e-4
    # Keep a mild optional long-hop attenuation knob for future tuning;
    # disabled by default in the Rician model.
    long_hop_knee_m: float = 211.73258874371447
    long_hop_scale_m: float = 619.8206935811197
    long_hop_tail_power: float = 1.7
    long_hop_tail_strength: float = 0.1223698259620352


def _rician_success_prob_from_avg_snr(avg_snr_db: float, link_model: LinkModelParams | None = None) -> float:
    lm = link_model if link_model is not None else LinkModelParams()
    lo = float(max(0.0, min(1.0, lm.succ_floor)))
    hi = float(max(lo, min(1.0, lm.succ_cap)))
    gamma_bar = float(10.0 ** (float(avg_snr_db) / 10.0))
    gamma_th = float(10.0 ** ((float(lm.snr_threshold_db) + float(lm.snr_gap_db)) / 10.0))
    if not math.isfinite(gamma_bar) or gamma_bar <= 0.0:
        return lo

    K = float(10.0 ** (float(lm.rician_k_db) / 10.0))
    ratio = float(max(0.0, gamma_th / max(1e-18, gamma_bar)))

    if ncx2 is not None:
        z = float(2.0 * (1.0 + K) * ratio)
        p_out = float(ncx2.cdf(z, df=2, nc=2.0 * K))
    else:  # conservative fallback without SciPy
        # Gaussian approximation around the LOS-dominated mean gain.
        mu = 1.0
        sigma = math.sqrt((1.0 + 2.0 * K) / ((1.0 + K) ** 2))
        thr = ratio
        z = (thr - mu) / max(1e-9, sigma)
        p_out = float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    p_succ = float(1.0 - max(0.0, min(1.0, p_out)))
    return float(max(lo, min(hi, p_succ)))


def _free_space_path_loss_db(d_m: float, freq_hz: float) -> float:
    d = max(1e-3, float(d_m))
    f_mhz = max(1e-6, float(freq_hz) / 1e6)
    return float(32.44 + 20.0 * math.log10(d / 1000.0) + 20.0 * math.log10(f_mhz))


def _link_success_prob_from_distance(d_m: float, link_model: LinkModelParams | None = None) -> float:
    """Average received power via Friis/log-distance + Rician outage closure.

    The physical layer now uses: distance -> average received power -> average
    SNR -> Rician frame-success probability. This is closer to a LOS air-to-air
    link than the older logistic surrogate.
    """
    lm = link_model if link_model is not None else LinkModelParams()
    d = max(float(lm.d0_m), float(d_m))
    pl0 = _free_space_path_loss_db(float(lm.d0_m), float(lm.freq_hz))
    prx_dbm = float(
        lm.pt_dbm + lm.gt_db + lm.gr_db - pl0
        - 10.0 * float(lm.path_loss_exp) * math.log10(d / max(1e-9, float(lm.d0_m)))
    )
    avg_snr_db = float(prx_dbm - float(lm.noise_floor_dbm) - float(lm.shadow_margin_db))
    p = _rician_success_prob_from_avg_snr(avg_snr_db, link_model=lm)

    lo = float(max(0.0, min(1.0, lm.succ_floor)))
    hi = float(max(lo, min(1.0, lm.succ_cap)))
    knee = float(max(float(lm.d0_m), getattr(lm, 'long_hop_knee_m', 220.0)))
    if d > knee:
        scale = float(max(1e-9, getattr(lm, 'long_hop_scale_m', 520.0)))
        power = float(max(1.0, getattr(lm, 'long_hop_tail_power', 1.7)))
        strength = float(max(0.0, getattr(lm, 'long_hop_tail_strength', 0.0)))
        if strength > 0.0:
            ratio = float(max(0.0, d - knee) / scale)
            tail = float(math.exp(-strength * (ratio ** power)))
            p = float(lo + (p - lo) * tail)

    return float(max(lo, min(hi, p)))


def _infer_ber_per_link_from_geometry(
    path: Sequence[int],
    pos_m: np.ndarray,
    frame_bits: int,
    link_model: LinkModelParams | None = None,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> List[float]:
    if _gpu_backend_enabled(use_torch_backend=use_torch_backend, torch_device=torch_device):
        try:
            return _infer_ber_per_link_from_geometry_torch(path, pos_m, frame_bits, link_model=link_model, torch_device=torch_device)
        except Exception:
            pass
    p = list(map(int, path))
    if len(p) < 2:
        return []
    pos = np.asarray(pos_m, dtype=np.float64)
    out: List[float] = []
    for i in range(len(p) - 1):
        u = int(p[i]); v = int(p[i + 1])
        d = float(np.linalg.norm(pos[u, :2] - pos[v, :2]))
        succ = _link_success_prob_from_distance(d, link_model=link_model)
        out.append(_link_ber_from_success_prob(succ, frame_bits))
    return out


def enumerate_simple_paths(adj01: np.ndarray, src: int, dst: int, max_paths: int = 8, max_hops: int | None = None) -> List[List[int]]:
    n = int(adj01.shape[0])
    src = int(src)
    dst = int(dst)
    if src < 0 or dst < 0 or src >= n or dst >= n:
        return []
    out_neighbors = [np.flatnonzero(adj01[u] > 0).astype(int).tolist() for u in range(n)]
    for neigh in out_neighbors:
        neigh.sort()
    if max_hops is None:
        max_hops = max(1, n - 1)

    paths: List[List[int]] = []
    visited = [False] * n
    cur: List[int] = []

    def dfs(u: int):
        if len(paths) >= max_paths:
            return
        cur.append(u)
        if u == dst:
            paths.append(cur.copy())
            cur.pop()
            return
        if len(cur) - 1 >= int(max_hops):
            cur.pop()
            return
        visited[u] = True
        for v in out_neighbors[u]:
            if not visited[v]:
                dfs(int(v))
                if len(paths) >= max_paths:
                    break
        visited[u] = False
        cur.pop()

    dfs(src)
    paths.sort(key=lambda p: (len(p), p))
    return paths[:max_paths]


def _pairwise_path_overlap_score(
    path_a: Sequence[int],
    path_b: Sequence[int],
    pos_m: np.ndarray,
    carrier_sense_range_m: float,
    shared_edge_weight: float,
    shared_node_weight: float,
    cs_weight: float,
) -> float:
    pa = list(map(int, path_a))
    pb = list(map(int, path_b))
    if len(pa) < 2 or len(pb) < 2:
        return 0.0
    pos = np.asarray(pos_m, dtype=np.float32)
    ea = {(pa[i], pa[i + 1]) for i in range(len(pa) - 1)}
    eb = {(pb[i], pb[i + 1]) for i in range(len(pb) - 1)}
    na = set(pa[1:-1])
    nb = set(pb[1:-1])
    ta = set(pa[:-1])
    tb = set(pb[:-1])

    edge_overlap = (len(ea & eb) / max(1, min(len(ea), len(eb)))) if ea and eb else 0.0
    node_overlap = (len(na & nb) / max(1, min(len(na), len(nb)))) if na and nb else 0.0

    cs_hits = 0
    cs_tot = 0
    for u in ta:
        for v in tb:
            if u == v:
                cs_hits += 1
            else:
                d = float(np.linalg.norm(pos[u, :2] - pos[v, :2]))
                if d <= carrier_sense_range_m:
                    cs_hits += 1
            cs_tot += 1
    cs_overlap = (cs_hits / max(1, cs_tot)) if cs_tot > 0 else 0.0

    w = shared_edge_weight * edge_overlap + shared_node_weight * node_overlap + cs_weight * cs_overlap
    return float(max(0.0, min(1.0, w)))


def build_interference_matrix(
    paths: Sequence[Sequence[int]],
    pos_m: np.ndarray,
    carrier_sense_range_m: float,
    shared_edge_weight: float = 0.40,
    shared_node_weight: float = 0.20,
    cs_weight: float = 0.10,
    path_owner: Sequence[int] | None = None,
    same_flow_discount: float = 0.5660007226003583,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> np.ndarray:
    if _gpu_backend_enabled(use_torch_backend=use_torch_backend, torch_device=torch_device):
        try:
            return build_interference_matrix_torch(paths, pos_m, carrier_sense_range_m, shared_edge_weight=shared_edge_weight, shared_node_weight=shared_node_weight, cs_weight=cs_weight, path_owner=path_owner, same_flow_discount=same_flow_discount, torch_device=torch_device)
        except Exception:
            pass
    """Return the pairwise path-coupling matrix omega.

    omega[a, b] is a dimensionless coupling score in [0, 1].
    It summarizes how much path *b* should perturb path *a* because of:
      - shared directed edges;
      - shared intermediate nodes;
      - transmitter pairs inside carrier-sense range.

    The same-flow case is deliberately softened. Same-flow redundancy is now
    handled mainly by explicit copy multiplicity on the active DAG (dedup only
    at the final destination), so using a strong global omega for same-flow
    paths would double-count the penalty and make the model pessimistic.
    """
    k = len(paths)
    omega = np.zeros((k, k), dtype=np.float32)
    if k <= 1:
        return omega

    owners = None if path_owner is None else list(map(int, path_owner))
    for a in range(k):
        for b in range(a + 1, k):
            w = _pairwise_path_overlap_score(
                paths[a], paths[b], pos_m, carrier_sense_range_m,
                shared_edge_weight=shared_edge_weight,
                shared_node_weight=shared_node_weight,
                cs_weight=cs_weight,
            )
            if owners is not None and a < len(owners) and b < len(owners) and owners[a] == owners[b]:
                w *= float(max(0.0, min(1.0, same_flow_discount)))
            omega[a, b] = omega[b, a] = float(max(0.0, min(1.0, w)))
    return omega


def _path_edges(path: Sequence[int]) -> List[tuple[int, int]]:
    p = list(map(int, path))
    return [(p[i], p[i + 1]) for i in range(max(0, len(p) - 1))]


def _topological_order_from_active_paths(paths: Sequence[Sequence[int]]) -> List[int]:
    nodes: set[int] = set()
    out: dict[int, set[int]] = {}
    indeg: dict[int, int] = {}
    edges_seen: set[tuple[int, int]] = set()
    for path in paths:
        for u, v in _path_edges(path):
            u = int(u)
            v = int(v)
            nodes.add(u)
            nodes.add(v)
            out.setdefault(u, set()).add(v)
            out.setdefault(v, set())
            indeg.setdefault(u, 0)
            indeg.setdefault(v, 0)
            if (u, v) not in edges_seen:
                edges_seen.add((u, v))
                indeg[v] = indeg.get(v, 0) + 1
    if not nodes:
        return []
    for u in list(nodes):
        out.setdefault(int(u), set())
        indeg.setdefault(int(u), 0)
    queue = [u for u in sorted(nodes) if indeg.get(int(u), 0) == 0]
    order: List[int] = []
    while queue:
        u = queue.pop(0)
        order.append(int(u))
        for v in sorted(out.get(int(u), set())):
            indeg[int(v)] -= 1
            if indeg[int(v)] == 0:
                queue.append(int(v))
    if len(order) == len(nodes):
        return order
    seen = set()
    fallback: List[int] = []
    for path in paths:
        for u in path:
            uu = int(u)
            if uu not in seen:
                seen.add(uu)
                fallback.append(uu)
    return fallback


def _edge_copy_multiplicity_no_dedup(paths: Sequence[Sequence[int]]) -> dict[tuple[int, int], float]:
    """Per-edge copy multiplicity with dedup only at the final destination.

    Shared prefixes keep multiplicity 1 until the first split.
    After a split, each branch carries the full copy.
    After a merge, copies accumulate and keep propagating together.
    """
    paths = [list(map(int, p)) for p in paths if len(p) >= 2]
    if not paths:
        return {}
    order = _topological_order_from_active_paths(paths)
    if not order:
        return {}

    src = int(paths[0][0])
    out = {}
    for p in paths:
        for u, v in _path_edges(p):
            out.setdefault(int(u), set()).add(int(v))
            out.setdefault(int(v), set())

    copies_at_node = {int(u): 0.0 for u in order}
    copies_at_node[src] = 1.0
    edge_mult = {}
    for u in order:
        cu = float(copies_at_node.get(int(u), 0.0))
        if cu <= 0.0:
            continue
        for v in sorted(out.get(int(u), set())):
            edge_mult[(int(u), int(v))] = edge_mult.get((int(u), int(v)), 0.0) + cu
            copies_at_node[int(v)] = copies_at_node.get(int(v), 0.0) + cu
    return edge_mult


def _path_mean_extra_copies(path: Sequence[int], edge_mult: dict[tuple[int, int], float]) -> float:
    edges = _path_edges(path)
    if not edges:
        return 0.0
    vals = [max(0.0, float(edge_mult.get(e, 1.0)) - 1.0) for e in edges]
    return float(np.mean(vals, dtype=np.float64)) if vals else 0.0


def _path_mean_physical_scale(path: Sequence[int], edge_mult: dict[tuple[int, int], float]) -> float:
    edges = _path_edges(path)
    if not edges:
        return 1.0
    vals = [max(1.0, float(edge_mult.get(e, 1.0))) for e in edges]
    return float(np.mean(vals, dtype=np.float64)) if vals else 1.0


def _mm1k_stationary(rho: float, k: int) -> np.ndarray:
    rho = float(max(0.0, rho))
    k = int(max(1, k))
    if abs(rho - 1.0) < 1e-12:
        return np.full((k + 1,), 1.0 / float(k + 1), dtype=np.float64)
    den = 1.0 - rho ** (k + 1)
    if abs(den) < 1e-12:
        return np.zeros((k + 1,), dtype=np.float64)
    pi0 = (1.0 - rho) / den
    arr = np.asarray([pi0 * (rho ** n) for n in range(k + 1)], dtype=np.float64)
    s = float(arr.sum())
    return arr / s if s > 0.0 else np.zeros((k + 1,), dtype=np.float64)


def _link_ber_from_success_prob(success_prob: float, frame_bits: int) -> float:
    p = float(max(1e-12, min(1.0 - 1e-12, success_prob)))
    b = int(max(8, frame_bits))
    return float(max(1e-12, min(0.5, 1.0 - p ** (1.0 / b))))


def _cw_sequence(cw_min: int, cw_max: int, max_attempts: int) -> List[int]:
    cw_min = int(max(1, cw_min))
    cw_max = int(max(cw_min, cw_max))
    m = int(max(1, max_attempts))
    out: List[int] = []
    for i in range(1, m + 1):
        wi = min((2 ** (i - 1)) * (cw_min + 1) - 1, cw_max)
        out.append(int(max(1, wi)))
    return out


def _build_path_hop_params(
    path: Sequence[int],
    pos_m: np.ndarray,
    frame_size_bytes: int,
    phy_rates_per_path: Sequence[float] | None,
    phy_rate_mbps: float,
    ber_per_link: Sequence[float] | None,
    slot_time_us: float,
    difs_us: float,
    sifs_us: float,
    ack_us: float,
    ack_timeout_us: float,
    cw_min: int,
    cw_max: int,
    max_attempts: int,
) -> List[HopParams]:
    frame_bits = int(max(64, int(frame_size_bytes) * 8))
    hops = max(1, len(path) - 1)
    pos = np.asarray(pos_m, dtype=np.float64)
    out: List[HopParams] = []
    for i in range(hops):
        rate = float(phy_rates_per_path[i]) if phy_rates_per_path is not None and i < len(phy_rates_per_path) else float(phy_rate_mbps)
        ber = float(ber_per_link[i]) if ber_per_link is not None and i < len(ber_per_link) else 1e-6
        u = int(path[i])
        v = int(path[i + 1])
        sender_xy = (float(pos[u, 0]), float(pos[u, 1]))
        receiver_xy = (float(pos[v, 0]), float(pos[v, 1]))
        out.append(HopParams(
            frame_bits=frame_bits,
            phy_rate_mbps=max(1e-9, rate),
            ber=max(1e-12, min(0.5, ber)),
            slot_time_us=float(slot_time_us),
            difs_us=float(difs_us),
            sifs_us=float(sifs_us),
            ack_us=float(ack_us),
            ack_timeout_us=float(ack_timeout_us),
            cw_min=int(cw_min),
            cw_max=int(cw_max),
            max_attempts=int(max(1, max_attempts)),
            sender_xy=sender_xy,
            receiver_xy=receiver_xy,
        ))
    return out


def _solve_high_level_queue(lam_pps: float, service_us: float, k: int) -> QueueState:
    lam = float(max(0.0, lam_pps))
    mu = 1e6 / max(1.0, float(service_us))
    rho = lam / max(1e-12, mu)
    rho = min(rho, 1e6)
    pi = _mm1k_stationary(rho, k)
    p0 = float(pi[0]) if pi.size else 1.0
    pK = float(pi[-1]) if pi.size else 0.0
    U = max(0.0, min(1.0, 1.0 - p0))
    X = float(mu * U)
    Q = float(np.sum(np.arange(len(pi), dtype=np.float64) * pi))
    Dq_ms = 0.0 if X <= 1e-12 else float(1000.0 * Q / X)
    Dq_ms = min(Dq_ms, 1e6)
    return QueueState(lam_pps=lam, mu_pps=mu, X_pps=X, U=U, P_block=pK, Q_avg=Q, Dq_ms=Dq_ms)


def _attempt_statistics(frame_loss: float, max_attempts: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exact-attempt probabilities phi_k, reach probabilities, and Phi.

    phi_exact[k-1] = probability a datagram uses exactly k frame transmissions.
    phi_reach[k-1] = probability the process reaches attempt k.
    Phi = E[number of frame transmissions per datagram].
    """
    m = int(max(1, max_attempts))
    f = float(max(1e-12, min(0.999999, frame_loss)))
    phi_exact = np.zeros((m,), dtype=np.float64)
    phi_reach = np.zeros((m,), dtype=np.float64)
    for k in range(1, m + 1):
        phi_reach[k - 1] = f ** (k - 1)
        if k < m:
            phi_exact[k - 1] = (f ** (k - 1)) * (1.0 - f)
        else:
            phi_exact[k - 1] = f ** (k - 1)
    Phi = float(np.sum(np.arange(1, m + 1, dtype=np.float64) * phi_exact))
    return phi_exact, phi_reach, max(1.0, Phi)


def _paper_mean_backoff_slots(cw_seq: Sequence[int], phi_exact: np.ndarray, Phi: float) -> float:
    """Eq. (21)-style mean backoff duration in slots.

    B_n = [sum_k ((W_1 + ... + W_k)/2) * phi_k] / Phi
    """
    if len(cw_seq) == 0:
        return 1.0
    csum = 0.0
    num = 0.0
    for k, wi in enumerate(cw_seq):
        csum += float(wi)
        pk = float(phi_exact[k]) if k < len(phi_exact) else 0.0
        num += 0.5 * csum * pk
    return float(max(1.0, num / max(1e-12, float(Phi))))


def _paper_eta_correction(U: float, service_us: float, tx_us: float) -> float:
    """Corrective factor η_n from the non-saturated case.

    The PDF text layout makes the exact typography awkward, but the expression
    used in the paper corresponds to a ratio between the active-backoff time and
    the non-idle time. The stable dimensionless form is:

        η_n = (S_n - T) / (S_n * (1-U_n)/U_n + S_n - T)

    which satisfies η_n -> 1 in saturation and η_n -> 0 when U_n -> 0.
    """
    U = float(max(1e-9, min(1.0, U)))
    S = float(max(tx_us, service_us))
    T = float(max(1e-9, tx_us))
    denom = S * ((1.0 - U) / U) + max(0.0, S - T)
    if denom <= 1e-12:
        return 1.0
    return float(max(0.0, min(1.0, max(0.0, S - T) / denom)))


def _paper_visible_indices(idx: int, hop_params: Sequence[HopParams], carrier_sense_range_m: float, geom_margin: float = 1.10) -> List[int]:
    n = len(hop_params)
    xi = np.asarray(hop_params[idx].sender_xy, dtype=np.float64)
    thr = float(max(0.0, carrier_sense_range_m)) * float(max(1.0, geom_margin))
    out: List[int] = []
    for j in range(n):
        if j == idx:
            continue
        xj = np.asarray(hop_params[j].sender_xy, dtype=np.float64)
        d = float(np.linalg.norm(xi - xj))
        if d <= thr:
            out.append(j)
    return out


def _paper_hidden_indices(idx: int, hop_params: Sequence[HopParams], carrier_sense_range_m: float, geom_margin: float = 1.10, hidden_factor: float = 2.0) -> List[int]:
    xi = np.asarray(hop_params[idx].sender_xy, dtype=np.float64)
    lo = float(max(0.0, carrier_sense_range_m)) * float(max(1.0, geom_margin))
    hi = lo * float(max(1.0, hidden_factor))
    out: List[int] = []
    for j in range(len(hop_params)):
        if j == idx:
            continue
        xj = np.asarray(hop_params[j].sender_xy, dtype=np.float64)
        d = float(np.linalg.norm(xi - xj))
        if lo < d <= hi:
            out.append(j)
    return out


def _neighbor_phi_and_backoff(j: int, qstates: Sequence[QueueState], hop_params: Sequence[HopParams], prev_mstates: Sequence[MacState] | None) -> tuple[np.ndarray, float, float, float]:
    if prev_mstates is not None and 0 <= j < len(prev_mstates):
        ms = prev_mstates[j]
        return ms.phi_exact, ms.phi_mean_frames, ms.b_mean_slots, ms.frame_loss
    hp_j = hop_params[j]
    fer_j = 1.0 - (1.0 - hp_j.ber) ** hp_j.frame_bits
    frame_loss_j = max(1e-9, min(0.999999, fer_j))
    phi_exact_j, _phi_reach_j, Phi_j = _attempt_statistics(frame_loss_j, hp_j.max_attempts)
    b_slots_j = _paper_mean_backoff_slots(_cw_sequence(hp_j.cw_min, hp_j.cw_max, hp_j.max_attempts), phi_exact_j, Phi_j)
    return phi_exact_j, Phi_j, b_slots_j, frame_loss_j


def _estimate_collision_terms_single_path(
    idx: int,
    qstates: Sequence[QueueState],
    hop_params: Sequence[HopParams],
    prev_mstates: Sequence[MacState] | None,
    interference_load: float,
    carrier_sense_range_m: float,
    interference_to_freeze_scale: float = 1.0,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Estimate simultaneous-start collisions, hidden collisions, beta and gamma.

    This follows the paper case-study logic rather than the older ad hoc closure:
    - visible contenders are nodes inside carrier-sense range (up to 2 hops);
    - hidden contenders are only beyond carrier-sense range;
    - simultaneous-start term uses the paper-style U_j / B_j closure;
    - beta uses 1 / beta_n = B_n / np_n;
    - gamma uses the mean freezing duration T + DIFS.
    """
    n = len(qstates)
    hp_i = hop_params[idx]
    visible = _paper_visible_indices(idx, hop_params, carrier_sense_range_m)
    hidden = _paper_hidden_indices(idx, hop_params, carrier_sense_range_m)

    tx_us_i = hp_i.frame_bits / max(1e-9, hp_i.phy_rate_mbps) + hp_i.sifs_us + hp_i.ack_us

    # Self stats with FER-only bootstrap if needed.
    if prev_mstates is not None and 0 <= idx < len(prev_mstates):
        ms_i_prev = prev_mstates[idx]
        phi_exact_i = ms_i_prev.phi_exact
        Phi_i = ms_i_prev.phi_mean_frames
        b_slots_i = ms_i_prev.b_mean_slots
        frame_loss_i_prev = ms_i_prev.frame_loss
    else:
        fer_i_boot = 1.0 - (1.0 - hp_i.ber) ** hp_i.frame_bits
        frame_loss_i_prev = max(1e-9, min(0.999999, fer_i_boot))
        phi_exact_i, _phi_reach_i, Phi_i = _attempt_statistics(frame_loss_i_prev, hp_i.max_attempts)
        b_slots_i = _paper_mean_backoff_slots(_cw_sequence(hp_i.cw_min, hp_i.cw_max, hp_i.max_attempts), phi_exact_i, Phi_i)
    b_mean_us_i = max(1e-9, b_slots_i * hp_i.slot_time_us)

    # --- simultaneous starts among visible contenders ---
    st_terms = []
    busy_candidates = [tx_us_i]
    F_i = float(qstates[idx].X_pps * Phi_i) if 0 <= idx < len(qstates) else 0.0

    for j in visible:
        qj = qstates[j]
        hp_j = hop_params[j]
        _phi_exact_j, Phi_j, b_slots_j, _frame_loss_j = _neighbor_phi_and_backoff(j, qstates, hop_params, prev_mstates)
        tau_j = float(max(0.0, min(0.999999, qj.U / max(1.0, b_slots_j))))
        st_terms.append(tau_j)
        tx_us_j = hp_j.frame_bits / max(1e-9, hp_j.phy_rate_mbps) + hp_j.sifs_us + hp_j.ack_us
        busy_candidates.append(tx_us_j)
    p_simul = 1.0 - float(np.prod([1.0 - x for x in st_terms], dtype=np.float64)) if st_terms else 0.0

    # --- hidden-node collisions ---
    # Paper case: ACKs from a node beyond carrier-sense range collide with an edge node.
    hidden_activity = 0.0
    for j in hidden:
        qj = qstates[j]
        hp_j = hop_params[j]
        _phi_exact_j, Phi_j, _b_slots_j, _frame_loss_j = _neighbor_phi_and_backoff(j, qstates, hop_params, prev_mstates)
        F_j = float(qj.X_pps * Phi_j)
        # We model the collision duration with the ACK transmission time, consistent with
        # the 4-node hidden-node case discussed in the paper.
        hidden_activity += F_j * (hp_j.ack_us * 1e-6)
    p_hidden = float(max(0.0, min(0.999999, hidden_activity)))

    # The paper explicitly treats HID and ST as disjoint in the 4-node case.
    p_coll = float(max(0.0, min(0.999999, p_simul + p_hidden)))

    # --- beta_n and gamma_n ---
    # gamma_n: inverse of mean freezing duration = T + DIFS.
    busy_mean_us = float(np.mean(busy_candidates)) if busy_candidates else tx_us_i
    gamma = 1.0 / max(1e-9, busy_mean_us + hp_i.difs_us)

    # beta_n: 1 / beta_n = B_n / np_n.
    eta_i = _paper_eta_correction(qstates[idx].U, service_us=(prev_mstates[idx].service_us if prev_mstates is not None and idx < len(prev_mstates) else tx_us_i + hp_i.difs_us + b_mean_us_i), tx_us=tx_us_i)
    np_freezes = 0.0
    for j in visible:
        qj = qstates[j]
        _phi_exact_j, Phi_j, _b_slots_j, _frame_loss_j = _neighbor_phi_and_backoff(j, qstates, hop_params, prev_mstates)
        F_j = float(qj.X_pps * Phi_j)
        if F_i > 1e-12:
            np_freezes += (F_j / F_i) * eta_i
    np_freezes += float(max(0.0, float(interference_load)) * max(0.0, float(interference_to_freeze_scale)))
    beta = np_freezes / max(1e-9, b_mean_us_i)
    beta = float(max(1e-9, beta))

    return (
        float(p_simul),
        float(p_hidden),
        float(p_coll),
        float(beta),
        float(gamma),
        float(Phi_i),
        float(b_slots_i),
        float(b_mean_us_i),
        float(np_freezes),
    )


def _solve_low_level_mac_ctmc_like(
    hp: HopParams,
    qstates: Sequence[QueueState],
    hop_params: Sequence[HopParams],
    prev_mstates: Sequence[MacState] | None,
    idx: int,
    interference_load: float,
    carrier_sense_range_m: float,
    interference_to_freeze_scale: float = 1.0,
) -> MacState:
    fer = 1.0 - (1.0 - hp.ber) ** hp.frame_bits
    (
        p_simul,
        p_hidden,
        p_coll,
        beta,
        gamma,
        _Phi_i_prev,
        _b_slots_i_prev,
        _b_mean_us_i_prev,
        np_freezes,
    ) = _estimate_collision_terms_single_path(idx, qstates, hop_params, prev_mstates, interference_load, carrier_sense_range_m=carrier_sense_range_m, interference_to_freeze_scale=interference_to_freeze_scale)
    frame_loss = fer + p_coll - fer * p_coll
    frame_loss = max(1e-9, min(0.999999, frame_loss))

    phi_exact, phi_reach, Phi = _attempt_statistics(frame_loss, hp.max_attempts)
    cw_seq = _cw_sequence(hp.cw_min, hp.cw_max, hp.max_attempts)
    b_mean_slots = _paper_mean_backoff_slots(cw_seq, phi_exact, Phi)
    b_mean_us = b_mean_slots * hp.slot_time_us

    alpha = 1.0 / max(1e-9, hp.slot_time_us)
    tx_us = hp.frame_bits / max(1e-9, hp.phy_rate_mbps) + hp.sifs_us + hp.ack_us
    r_slot_us = (1.0 / alpha) * (1.0 + beta / max(1e-9, gamma))

    # Paper Eq. (13) + Eq. (14):
    #   S_n = t_1 + f (t_2 + ... + f t_m)
    #   t_k = DIFS + (W_k / 2) r_n + T_n
    tau = 0.0
    for wi in reversed(cw_seq):
        tk = hp.difs_us + 0.5 * float(wi) * r_slot_us + tx_us
        tau = tk + frame_loss * tau
    tau = float(max(tx_us, tau))

    return MacState(
        fer=float(fer),
        p_coll=float(p_coll),
        frame_loss=float(frame_loss),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        r_slot_us=float(r_slot_us),
        service_us=float(tau),
        phi_exact=phi_exact,
        phi_reach=phi_reach,
        phi_mean_frames=float(Phi),
        b_mean_slots=float(b_mean_slots),
        b_mean_us=float(b_mean_us),
        np_freezes=float(np_freezes),
        collision_hidden=float(p_hidden),
        collision_simultaneous=float(p_simul),
        busy_mean_us=float(tx_us + hp.difs_us),
    )


def _fixed_point_single_path(
    hop_params: Sequence[HopParams],
    offered_rate_mbps: float,
    interference_load: float,
    buffer_size_packets: int,
    carrier_sense_range_m: float,
    max_fp_iters: int = 40,
    tol: float = 1e-5,
    return_trace: bool = False,
    interference_to_freeze_scale: float = 1.0,
) -> tuple[List[QueueState], List[MacState]] | tuple[List[QueueState], List[MacState], list[dict[str, Any]]]:
    hops = len(hop_params)
    frame_bits = hop_params[0].frame_bits if hop_params else 12000
    src_pps = float(max(0.0, offered_rate_mbps) * 1e6 / max(1.0, frame_bits))

    qstates: List[QueueState] = []
    mstates: List[MacState] = []
    trace: list[dict[str, Any]] = []

    # bootstrap with FER-only service times.
    for hp in hop_params:
        fer = 1.0 - (1.0 - hp.ber) ** hp.frame_bits
        frame_loss = max(1e-9, min(0.999999, fer))
        phi_exact, phi_reach, Phi = _attempt_statistics(frame_loss, hp.max_attempts)
        cw_seq = _cw_sequence(hp.cw_min, hp.cw_max, hp.max_attempts)
        b_slots = _paper_mean_backoff_slots(cw_seq, phi_exact, Phi)
        tx_us = hp.frame_bits / max(1e-9, hp.phy_rate_mbps) + hp.sifs_us + hp.ack_us
        tau = 0.0
        for wi in reversed(cw_seq):
            tk = hp.difs_us + 0.5 * float(wi) * hp.slot_time_us + tx_us
            tau = tk + frame_loss * tau
        mstates.append(MacState(
            fer=float(fer),
            p_coll=0.0,
            frame_loss=float(frame_loss),
            alpha=float(1.0 / max(1e-9, hp.slot_time_us)),
            beta=1e-9,
            gamma=float(1.0 / max(1e-9, tx_us + hp.difs_us)),
            r_slot_us=float(hp.slot_time_us),
            service_us=float(max(tx_us, tau)),
            phi_exact=phi_exact,
            phi_reach=phi_reach,
            phi_mean_frames=float(Phi),
            b_mean_slots=float(b_slots),
            b_mean_us=float(b_slots * hp.slot_time_us),
            np_freezes=0.0,
            collision_hidden=0.0,
            collision_simultaneous=0.0,
            busy_mean_us=float(tx_us + hp.difs_us),
        ))

    prev_services = np.asarray([ms.service_us for ms in mstates], dtype=np.float64)
    for it in range(max(1, int(max_fp_iters))):
        qstates = []
        lam = src_pps
        for i, hp in enumerate(hop_params):
            q = _solve_high_level_queue(lam_pps=lam, service_us=mstates[i].service_us, k=buffer_size_packets)
            qstates.append(q)
            lam = q.X_pps

        new_mstates: List[MacState] = []
        for i, hp in enumerate(hop_params):
            new_mstates.append(
                _solve_low_level_mac_ctmc_like(
                    hp=hp,
                    qstates=qstates,
                    hop_params=hop_params,
                    prev_mstates=mstates,
                    idx=i,
                    interference_load=interference_load,
                    carrier_sense_range_m=carrier_sense_range_m,
                    interference_to_freeze_scale=interference_to_freeze_scale,
                )
            )

        mstates = new_mstates
        cur_services = np.asarray([ms.service_us for ms in mstates], dtype=np.float64)
        max_abs = float(np.max(np.abs(cur_services - prev_services))) if cur_services.size else 0.0
        max_rel = float(max_abs / max(1.0, float(np.max(prev_services)) if prev_services.size else 1.0))
        if return_trace:
            trace.append({
                "iter": int(it + 1),
                "services_us": cur_services.tolist(),
                "max_abs_service_delta_us": max_abs,
                "max_rel_service_delta": max_rel,
            })
        prev_services = cur_services
        if max_rel < float(tol):
            break

    if return_trace:
        return qstates, mstates, trace
    return qstates, mstates


def _solve_single_path(
    path: Sequence[int],
    pos_m: np.ndarray,
    offered_rate_mbps: float,
    interference_load: float,
    frame_size_bytes: int,
    buffer_size_packets: int,
    phy_rate_mbps: float,
    ber_per_link: Sequence[float] | None,
    slot_time_us: float,
    difs_us: float,
    sifs_us: float,
    ack_us: float,
    cw_min: int,
    max_retries: int,
    carrier_sense_range_m: float,
    reliability_min: float,
    delay_max_ms: float,
    n_search_steps: int = 18,
    phy_rates_per_path: Sequence[float] | None = None,
    ack_timeout_us: float | None = None,
    cw_max: int = 1023,
    max_attempts: int | None = None,
    interference_to_freeze_scale: float = 1.0,
    return_debug: bool = False,
    compute_max_feasible_rate: bool = True,
    link_model: LinkModelParams | None = None,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> PathMetrics:

    path = list(map(int, path))
    if len(path) < 2:
        return PathMetrics(
            path=path,
            offered_rate_mbps=float(offered_rate_mbps),
            effective_rate_mbps=float(offered_rate_mbps),
            reliability=0.0,
            loss_rate=1.0,
            delay_ms=float('inf'),
            goodput_mbps=0.0,
            max_feasible_rate_mbps=0.0,
            interference_load=float(interference_load),
            rho_max=0.0,
            debug={"error": "path_too_short"},
        )
    if max_attempts is None:
        max_attempts = max(1, int(max_retries) + 1)
    if ack_timeout_us is None:
        ack_timeout_us = float(ack_us)
    if ber_per_link is None:
        frame_bits = int(max(64, int(frame_size_bytes) * 8))
        ber_per_link = _infer_ber_per_link_from_geometry(path, pos_m, frame_bits, link_model=link_model, use_torch_backend=use_torch_backend, torch_device=torch_device)
    hop_params = _build_path_hop_params(
        path=path,
        pos_m=pos_m,
        frame_size_bytes=frame_size_bytes,
        phy_rates_per_path=phy_rates_per_path,
        phy_rate_mbps=phy_rate_mbps,
        ber_per_link=ber_per_link,
        slot_time_us=slot_time_us,
        difs_us=difs_us,
        sifs_us=sifs_us,
        ack_us=ack_us,
        ack_timeout_us=ack_timeout_us,
        cw_min=cw_min,
        cw_max=cw_max,
        max_attempts=max_attempts,
    )
    if return_debug:
        qstates, mstates, trace = _fixed_point_single_path(
            hop_params=hop_params,
            offered_rate_mbps=offered_rate_mbps,
            interference_load=interference_load,
            buffer_size_packets=buffer_size_packets,
            carrier_sense_range_m=carrier_sense_range_m,
            return_trace=True,
            interference_to_freeze_scale=interference_to_freeze_scale,
        )
    else:
        qstates, mstates = _fixed_point_single_path(
            hop_params=hop_params,
            offered_rate_mbps=offered_rate_mbps,
            interference_load=interference_load,
            buffer_size_packets=buffer_size_packets,
            carrier_sense_range_m=carrier_sense_range_m,
            return_trace=False,
            interference_to_freeze_scale=interference_to_freeze_scale,
        )
        trace = []

    src_pps = float(max(0.0, offered_rate_mbps) * 1e6 / max(1.0, hop_params[0].frame_bits))
    goodput_pps = float(qstates[-1].X_pps if qstates else 0.0)
    if abs(goodput_pps - src_pps) <= 1e-6 * max(1.0, src_pps):
        goodput_pps = float(src_pps)
    goodput_mbps = float(goodput_pps * hop_params[0].frame_bits / 1e6)
    if abs(goodput_mbps - float(offered_rate_mbps)) <= 1e-6 * max(1.0, float(offered_rate_mbps)):
        goodput_mbps = float(offered_rate_mbps)
    rel_block = float(np.prod([max(0.0, 1.0 - q.P_block) for q in qstates], dtype=np.float64)) if qstates else 0.0
    rel_ratio = 0.0 if src_pps <= 1e-12 else float(max(0.0, min(1.0, goodput_pps / src_pps)))
    reliability_block = float(max(0.0, min(1.0, rel_block)))
    delivery_ratio = float(max(0.0, min(1.0, rel_ratio)))
    if src_pps > 1e-12 and goodput_pps > 0.0:
        # Hybrid delivery-QoS score kept for backward compatibility.
        reliability_hybrid = float(max(0.0, min(1.0, 0.30 * reliability_block + 0.70 * delivery_ratio)))
    else:
        reliability_hybrid = float(reliability_block)
    reliability = float(reliability_hybrid)
    loss_rate = float(max(0.0, min(1.0, 1.0 - delivery_ratio)))
    delay_ms = float(sum(q.Dq_ms for q in qstates) + sum(ms.service_us for ms in mstates) / 1000.0)
    rho_max = 0.0 if not qstates else float(max(q.lam_pps / max(1e-9, q.mu_pps) for q in qstates))

    if compute_max_feasible_rate:
        def metrics_for_rate(rate_mbps: float) -> tuple[float, float]:
            qs, ms = _fixed_point_single_path(
                hop_params=hop_params,
                offered_rate_mbps=rate_mbps,
                interference_load=interference_load,
                buffer_size_packets=buffer_size_packets,
                carrier_sense_range_m=carrier_sense_range_m,
                max_fp_iters=30,
                tol=1e-5,
                return_trace=False,
                interference_to_freeze_scale=interference_to_freeze_scale,
            )
            src = float(max(0.0, rate_mbps) * 1e6 / max(1.0, hop_params[0].frame_bits))
            rel_prod = float(np.prod([max(0.0, 1.0 - q.P_block) for q in qs], dtype=np.float64)) if qs else 0.0
            gpps = float(qs[-1].X_pps if qs else 0.0)
            if abs(gpps - src) <= 1e-6 * max(1.0, src):
                gpps = float(src)
            rel_ratio = 0.0 if src <= 1e-12 else float(max(0.0, min(1.0, gpps / src)))
            # Capacity search keeps using the hybrid score so legacy feasibility
            # behavior remains comparable, while exposed metrics are separated.
            rel = float(max(0.0, min(1.0, 0.30 * rel_prod + 0.70 * rel_ratio)))
            dly = float(sum(q.Dq_ms for q in qs) + sum(msi.service_us for msi in ms) / 1000.0)
            return rel, dly

        lo = 0.0
        hi = max(1.0, float(offered_rate_mbps), max(hp.phy_rate_mbps for hp in hop_params))
        for _ in range(8):
            rel_h, del_h = metrics_for_rate(hi)
            if rel_h < reliability_min or del_h > delay_max_ms:
                break
            hi *= 1.5
            if hi > 4.0 * max(hp.phy_rate_mbps for hp in hop_params):
                break
        for _ in range(int(max(8, n_search_steps))):
            mid = 0.5 * (lo + hi)
            rel_m, del_m = metrics_for_rate(mid)
            if rel_m >= reliability_min and del_m <= delay_max_ms:
                lo = mid
            else:
                hi = mid
        max_feasible = float(lo)
    else:
        max_feasible = 0.0

    debug = None
    if return_debug:
        debug = {
            "trace": trace,
            "qstates": [q.__dict__ for q in qstates],
            "mstates": [
                {
                    "fer": ms.fer,
                    "p_coll": ms.p_coll,
                    "frame_loss": ms.frame_loss,
                    "alpha": ms.alpha,
                    "beta": ms.beta,
                    "gamma": ms.gamma,
                    "r_slot_us": ms.r_slot_us,
                    "service_us": ms.service_us,
                    "phi_exact": ms.phi_exact.tolist(),
                    "phi_reach": ms.phi_reach.tolist(),
                    "phi_mean_frames": ms.phi_mean_frames,
                    "b_mean_slots": ms.b_mean_slots,
                    "b_mean_us": ms.b_mean_us,
                    "np_freezes": ms.np_freezes,
                    "collision_hidden": ms.collision_hidden,
                    "collision_simultaneous": ms.collision_simultaneous,
                    "busy_mean_us": ms.busy_mean_us,
                }
                for ms in mstates
            ],
        }

    return PathMetrics(
        path=path,
        offered_rate_mbps=float(offered_rate_mbps),
        effective_rate_mbps=float(offered_rate_mbps) + float(interference_load) * float(offered_rate_mbps),
        reliability=float(reliability),
        loss_rate=float(loss_rate),
        delay_ms=float(delay_ms),
        goodput_mbps=float(goodput_mbps),
        max_feasible_rate_mbps=float(max_feasible),
        interference_load=float(interference_load),
        rho_max=float(rho_max),
        debug=debug,
        reliability_block=float(reliability_block),
        delivery_ratio=float(delivery_ratio),
        reliability_hybrid=float(reliability_hybrid),
    )


def debug_single_path_case(
    path: Sequence[int],
    pos_m: np.ndarray,
    offered_rate_mbps: float,
    frame_size_bytes: int = 1500,
    buffer_size_packets: int = 20,
    phy_rate_mbps: float = 54.0,
    ber_per_link: Sequence[float] | None = None,
    slot_time_us: float = 9.0,
    difs_us: float = 28.0,
    sifs_us: float = 10.0,
    ack_us: float = 24.0,
    ack_timeout_us: float = 34.0,
    cw_min: int = 15,
    cw_max: int = 1023,
    max_retries: int = 6,
    max_attempts: int | None = 7,
    interference_load: float = 0.0,
    phy_rates_per_path: Sequence[float] | None = None,
    compute_max_feasible_rate: bool = True,
) -> PathMetrics:
    return _solve_single_path(
        path=path,
        pos_m=pos_m,
        offered_rate_mbps=offered_rate_mbps,
        interference_load=interference_load,
        frame_size_bytes=frame_size_bytes,
        buffer_size_packets=buffer_size_packets,
        phy_rate_mbps=phy_rate_mbps,
        ber_per_link=ber_per_link,
        slot_time_us=slot_time_us,
        difs_us=difs_us,
        sifs_us=sifs_us,
        ack_us=ack_us,
        ack_timeout_us=ack_timeout_us,
        cw_min=cw_min,
        cw_max=cw_max,
        max_retries=max_retries,
        max_attempts=max_attempts,
        carrier_sense_range_m=0.0,
        reliability_min=0.95,
        delay_max_ms=50.0,
        phy_rates_per_path=phy_rates_per_path,
        return_debug=True,
        compute_max_feasible_rate=compute_max_feasible_rate,
        interference_to_freeze_scale=1.0,
    )


def _rank_path_candidates(
    paths: Sequence[Sequence[int]],
    pos_m: np.ndarray,
    offered_rate_mbps: float,
    frame_size_bytes: int,
    buffer_size_packets: int,
    phy_rate_mbps: float,
    ber_per_path: Sequence[Sequence[float]] | None,
    carrier_sense_range_m: float,
    slot_time_us: float,
    difs_us: float,
    sifs_us: float,
    ack_us: float,
    ack_timeout_us: float,
    cw_min: int,
    cw_max: int,
    max_retries: int,
    max_attempts: int | None,
    reliability_min: float,
    delay_max_ms: float,
    phy_rates_per_path: Sequence[Sequence[float]] | None,
    compute_max_feasible_rate: bool,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> List[int]:
    scored: List[tuple[float, float, float, int, int]] = []
    for idx, path in enumerate(paths):
        ber_links = ber_per_path[idx] if ber_per_path is not None and idx < len(ber_per_path) else None
        phy_links = phy_rates_per_path[idx] if phy_rates_per_path is not None and idx < len(phy_rates_per_path) else None
        pm = _solve_single_path(
            path=path,
            pos_m=pos_m,
            offered_rate_mbps=float(offered_rate_mbps),
            interference_load=0.0,
            frame_size_bytes=frame_size_bytes,
            buffer_size_packets=buffer_size_packets,
            phy_rate_mbps=phy_rate_mbps,
            ber_per_link=ber_links,
            slot_time_us=slot_time_us,
            difs_us=difs_us,
            sifs_us=sifs_us,
            ack_us=ack_us,
            ack_timeout_us=ack_timeout_us,
            cw_min=cw_min,
            cw_max=cw_max,
            max_retries=max_retries,
            max_attempts=max_attempts,
            carrier_sense_range_m=carrier_sense_range_m,
            reliability_min=reliability_min,
            delay_max_ms=delay_max_ms,
            phy_rates_per_path=phy_links,
            return_debug=False,
            compute_max_feasible_rate=compute_max_feasible_rate,
            interference_to_freeze_scale=1.0,
        )
        scored.append((float(pm.reliability), float(pm.goodput_mbps), -float(pm.delay_ms), -int(len(path)), -int(idx), idx))
    scored.sort(reverse=True)
    return [int(t[-1]) for t in scored]


def _normalize_flow_request(req: FlowRequest) -> FlowRequest:
    paths = [list(map(int, p)) for p in (req.paths or []) if len(p) >= 2]
    return FlowRequest(
        paths=paths,
        offered_rate_mbps=float(req.offered_rate_mbps),
        redundancy_mode=str(req.redundancy_mode or "single_best"),
        max_active_paths=req.max_active_paths,
        ber_per_path=req.ber_per_path,
        phy_rates_per_path=req.phy_rates_per_path,
        label=str(req.label or ""),
        per_path_offered_rates_mbps=req.per_path_offered_rates_mbps,
    )


def _select_active_path_indices(
    req: FlowRequest,
    pos_m: np.ndarray,
    frame_size_bytes: int,
    buffer_size_packets: int,
    phy_rate_mbps: float,
    carrier_sense_range_m: float,
    slot_time_us: float,
    difs_us: float,
    sifs_us: float,
    ack_us: float,
    ack_timeout_us: float,
    cw_min: int,
    cw_max: int,
    max_retries: int,
    max_attempts: int | None,
    reliability_min: float,
    delay_max_ms: float,
    compute_max_feasible_rate: bool,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> List[int]:
    req = _normalize_flow_request(req)
    if not req.paths:
        return []
    ranking = _rank_path_candidates(
        paths=req.paths,
        pos_m=pos_m,
        offered_rate_mbps=float(req.offered_rate_mbps),
        frame_size_bytes=frame_size_bytes,
        buffer_size_packets=buffer_size_packets,
        phy_rate_mbps=phy_rate_mbps,
        ber_per_path=req.ber_per_path,
        carrier_sense_range_m=carrier_sense_range_m,
        slot_time_us=slot_time_us,
        difs_us=difs_us,
        sifs_us=sifs_us,
        ack_us=ack_us,
        ack_timeout_us=ack_timeout_us,
        cw_min=cw_min,
        cw_max=cw_max,
        max_retries=max_retries,
        max_attempts=max_attempts,
        reliability_min=reliability_min,
        delay_max_ms=delay_max_ms,
        phy_rates_per_path=req.phy_rates_per_path,
        compute_max_feasible_rate=compute_max_feasible_rate,
        use_torch_backend=use_torch_backend,
        torch_device=torch_device,
    )
    mode = req.redundancy_mode.lower()
    if mode in {"single_best", "single", "best"}:
        return ranking[:1]
    if mode in {"replicate_top_k", "top_k", "replicate"}:
        k = int(req.max_active_paths) if req.max_active_paths is not None else len(ranking)
        return ranking[:max(1, min(len(ranking), k))]
    if mode in {"all", "all_paths"}:
        return ranking
    k = int(req.max_active_paths) if req.max_active_paths is not None else 1
    return ranking[:max(1, min(len(ranking), k))]


def _flow_path_offered_rate(req: FlowRequest, path_index: int) -> float:
    if req.per_path_offered_rates_mbps is not None and path_index < len(req.per_path_offered_rates_mbps):
        return float(req.per_path_offered_rates_mbps[path_index])
    return float(req.offered_rate_mbps)


def _flow_metric_from_paths(active_paths: List[ActivePathMetrics], attr_name: str, default_attr: str = "reliability") -> float:
    if not active_paths:
        return 0.0
    vals = np.asarray([
        max(0.0, min(1.0, float(getattr(ap.path_metrics, attr_name, getattr(ap.path_metrics, default_attr, 0.0)))))
        for ap in active_paths
    ], dtype=np.float64)
    if vals.size == 1:
        return float(vals[0])
    val_indep = float(1.0 - np.prod(1.0 - vals, dtype=np.float64))
    val_best = float(np.max(vals))
    extras = [max(0.0, float(ap.physical_load_scale) - 1.0) for ap in active_paths]
    mean_extra = float(np.mean(extras, dtype=np.float64)) if extras else 0.0
    dep_penalty = float(min(0.45, 0.35 * mean_extra))
    independence = float(max(0.55, min(1.0, 1.0 - dep_penalty)))
    return float(max(0.0, min(1.0, val_best + independence * (val_indep - val_best))))


def _flow_reliability_from_paths(active_paths: List[ActivePathMetrics]) -> float:
    if not active_paths:
        return 0.0
    rels = np.asarray([max(0.0, min(1.0, ap.path_metrics.reliability)) for ap in active_paths], dtype=np.float64)
    if rels.size == 1:
        return float(rels[0])
    rel_indep = float(1.0 - np.prod(1.0 - rels, dtype=np.float64))
    rel_best = float(np.max(rels))
    # Dependence is tied to *real physical duplicate load* after merges, not to
    # raw path overlap. Shared prefixes before the first split do not create two
    # copies in the network and therefore should not collapse the redundancy gain.
    extras = [max(0.0, float(ap.physical_load_scale) - 1.0) for ap in active_paths]
    mean_extra = float(np.mean(extras, dtype=np.float64)) if extras else 0.0
    # Dedup only at the final destination means a shared suffix does create
    # additional physical load, but it should not be treated as near-total
    # statistical dependence between copies. We therefore attenuate the
    # independence gain gently instead of collapsing it linearly with mean_extra.
    dep_penalty = float(min(0.45, 0.35 * mean_extra))
    independence = float(max(0.55, min(1.0, 1.0 - dep_penalty)))
    return float(max(0.0, min(1.0, rel_best + independence * (rel_indep - rel_best))))


def solve_modified_abreu_multiflow(
    flows: Sequence[FlowRequest],
    pos_m: np.ndarray,
    frame_size_bytes: int = 1500,
    buffer_size_packets: int = 20,
    phy_rate_mbps: float = 54.0,
    ber_per_path: Sequence[Sequence[float]] | None = None,
    carrier_sense_range_m: float = 199.4323529413177,
    slot_time_us: float = 9.0,
    difs_us: float = 28.0,
    sifs_us: float = 10.0,
    ack_us: float = 24.0,
    ack_timeout_us: float = 34.0,
    cw_min: int = 15,
    cw_max: int = 1023,
    max_retries: int = 6,
    max_attempts: int | None = 7,
    reliability_min: float = 0.95,
    delay_max_ms: float = 50.0,
    compute_max_feasible_rate: bool = True,
    same_flow_discount: float = 0.5660007226003583,
    self_overlap_load_scale: float = 0.11292807059201448,
    cross_flow_load_scale: float = 0.5811699219640487,
    interference_to_freeze_scale: float = 0.4245764622910097,
    link_model: LinkModelParams | None = None,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> MultiFlowMetrics:
    del ber_per_path

    reqs = [_normalize_flow_request(fr) for fr in flows]
    reqs = [fr for fr in reqs if fr.paths]
    if not reqs:
        return MultiFlowMetrics([], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float('inf'), 0.0, np.zeros((0, 0), dtype=np.float32))

    selected_local: List[List[int]] = []
    for req in reqs:
        selected_local.append(_select_active_path_indices(
            req=req,
            pos_m=pos_m,
            frame_size_bytes=frame_size_bytes,
            buffer_size_packets=buffer_size_packets,
            phy_rate_mbps=phy_rate_mbps,
            carrier_sense_range_m=carrier_sense_range_m,
            slot_time_us=slot_time_us,
            difs_us=difs_us,
            sifs_us=sifs_us,
            ack_us=ack_us,
            ack_timeout_us=ack_timeout_us,
            cw_min=cw_min,
            cw_max=cw_max,
            max_retries=max_retries,
            max_attempts=max_attempts,
            reliability_min=reliability_min,
            delay_max_ms=delay_max_ms,
            compute_max_feasible_rate=compute_max_feasible_rate,
            use_torch_backend=use_torch_backend,
            torch_device=torch_device,
        ))

    flat_paths: List[List[int]] = []
    flat_owner: List[int] = []
    flat_local_idx: List[int] = []
    flat_offered: List[float] = []
    flat_ber: List[Sequence[float] | None] = []
    flat_phy: List[Sequence[float] | None] = []
    for flow_idx, req in enumerate(reqs):
        for local_idx in selected_local[flow_idx]:
            flat_paths.append(req.paths[local_idx])
            flat_owner.append(int(flow_idx))
            flat_local_idx.append(int(local_idx))
            flat_offered.append(_flow_path_offered_rate(req, local_idx))
            flat_ber.append(req.ber_per_path[local_idx] if req.ber_per_path is not None and local_idx < len(req.ber_per_path) else None)
            flat_phy.append(req.phy_rates_per_path[local_idx] if req.phy_rates_per_path is not None and local_idx < len(req.phy_rates_per_path) else None)

    if not flat_paths:
        return MultiFlowMetrics([], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float('inf'), 0.0, np.zeros((0, 0), dtype=np.float32))

    omega = build_interference_matrix(flat_paths, pos_m, carrier_sense_range_m, path_owner=flat_owner, same_flow_discount=same_flow_discount, use_torch_backend=use_torch_backend, torch_device=torch_device)
    omega_raw = build_interference_matrix(flat_paths, pos_m, carrier_sense_range_m, path_owner=None, same_flow_discount=1.0, use_torch_backend=use_torch_backend, torch_device=torch_device)

    flow_edge_mult: List[dict[tuple[int, int], float]] = []
    for flow_idx, req in enumerate(reqs):
        active_paths = [req.paths[j] for j in selected_local[flow_idx]]
        flow_edge_mult.append(_edge_copy_multiplicity_no_dedup(active_paths))

    wrappers_by_flow: List[List[ActivePathMetrics]] = [[] for _ in reqs]
    active_global_by_flow: List[List[int]] = [[] for _ in reqs]

    for gidx, path in enumerate(flat_paths):
        flow_idx = int(flat_owner[gidx])
        local_idx = int(flat_local_idx[gidx])
        own_edge_mult = flow_edge_mult[flow_idx]
        self_extra = _path_mean_extra_copies(path, own_edge_mult)
        physical_scale = _path_mean_physical_scale(path, own_edge_mult)
        cross_extra = 0.0
        for j in range(len(flat_paths)):
            if j == gidx or flat_owner[j] == flow_idx:
                continue
            cross_extra += float(omega[gidx, j]) * (float(flat_offered[j]) / max(1e-9, phy_rate_mbps))
        inter_load = float(self_overlap_load_scale) * float(self_extra) + float(cross_flow_load_scale) * float(max(0.0, cross_extra))
        pm = _solve_single_path(
            path=path,
            pos_m=pos_m,
            offered_rate_mbps=float(flat_offered[gidx]),
            interference_load=inter_load,
            frame_size_bytes=frame_size_bytes,
            buffer_size_packets=buffer_size_packets,
            phy_rate_mbps=phy_rate_mbps,
            ber_per_link=flat_ber[gidx],
            slot_time_us=slot_time_us,
            difs_us=difs_us,
            sifs_us=sifs_us,
            ack_us=ack_us,
            ack_timeout_us=ack_timeout_us,
            cw_min=cw_min,
            cw_max=cw_max,
            max_retries=max_retries,
            max_attempts=max_attempts,
            carrier_sense_range_m=carrier_sense_range_m,
            reliability_min=reliability_min,
            delay_max_ms=delay_max_ms,
            phy_rates_per_path=flat_phy[gidx],
            return_debug=False,
            compute_max_feasible_rate=compute_max_feasible_rate,
            interference_to_freeze_scale=interference_to_freeze_scale,
            link_model=link_model,
        )
        wrapper = ActivePathMetrics(
            path_index=local_idx,
            active=True,
            offered_rate_mbps=float(flat_offered[gidx]),
            effective_rate_mbps=float(flat_offered[gidx]) * float(max(1.0, physical_scale)),
            physical_load_scale=float(max(1.0, physical_scale)),
            path_metrics=pm,
        )
        wrappers_by_flow[flow_idx].append(wrapper)
        active_global_by_flow[flow_idx].append(gidx)

    flow_metrics: List[FlowMetrics] = []
    for flow_idx, req in enumerate(reqs):
        active_wrappers = wrappers_by_flow[flow_idx]
        active_indices_local = [int(w.path_index) for w in active_wrappers]
        if active_wrappers:
            flow_rel = _flow_reliability_from_paths(active_wrappers)
            flow_rel_block = _flow_metric_from_paths(active_wrappers, "reliability_block")
            # Useful delivered traffic for redundant copies is still capped by the
            # source rate; delivery_ratio captures how much of the requested flow arrived.
            flow_delay = float(min(w.path_metrics.delay_ms for w in active_wrappers))
            if len(active_wrappers) == 1:
                # For a non-redundant single active path, the useful goodput is the
                # single-path throughput already computed by the hierarchical model.
                flow_goodput = float(max(0.0, min(float(req.offered_rate_mbps), active_wrappers[0].path_metrics.goodput_mbps)))
            else:
                # With end-only deduplication across redundant copies, useful goodput
                # is capped by the original source rate and modulated by the end-to-end
                # probability that at least one copy reaches the destination.
                flow_goodput = float(max(0.0, min(float(req.offered_rate_mbps), float(req.offered_rate_mbps) * flow_rel)))
            if abs(flow_goodput - float(req.offered_rate_mbps)) <= 1e-6 * max(1.0, float(req.offered_rate_mbps)):
                flow_goodput = float(req.offered_rate_mbps)
            flow_delivery_ratio = float(0.0 if float(req.offered_rate_mbps) <= 1e-12 else max(0.0, min(1.0, flow_goodput / float(req.offered_rate_mbps))))
            flow_hybrid = float(max(0.0, min(1.0, 0.30 * flow_rel_block + 0.70 * flow_delivery_ratio)))
            flow_mfr = float(max(w.path_metrics.max_feasible_rate_mbps for w in active_wrappers))
            flow_rho = float(max(w.path_metrics.rho_max for w in active_wrappers))
        else:
            flow_rel = 0.0
            flow_rel_block = 0.0
            flow_delivery_ratio = 0.0
            flow_hybrid = 0.0
            flow_delay = float('inf')
            flow_goodput = 0.0
            flow_mfr = 0.0
            flow_rho = 0.0
        flow_metrics.append(FlowMetrics(
            label=req.label,
            offered_rate_mbps=float(req.offered_rate_mbps),
            redundancy_mode=req.redundancy_mode,
            active_path_indices=active_indices_local,
            path_metrics=active_wrappers,
            reliability=float(max(0.0, min(1.0, flow_rel))),
            loss_rate=float(max(0.0, min(1.0, 1.0 - flow_rel))),
            delay_ms=float(flow_delay),
            goodput_mbps=float(flow_goodput),
            max_feasible_rate_mbps=float(flow_mfr),
            rho_max=float(flow_rho),
            reliability_block=float(flow_rel_block),
            delivery_ratio=float(flow_delivery_ratio),
            reliability_hybrid=float(flow_hybrid),
        ))

    flow_rels = np.asarray([fm.reliability for fm in flow_metrics], dtype=np.float64)
    flow_rel_blocks = np.asarray([getattr(fm, "reliability_block", fm.reliability) for fm in flow_metrics], dtype=np.float64)
    flow_delivery_ratios = np.asarray([getattr(fm, "delivery_ratio", 0.0) for fm in flow_metrics], dtype=np.float64)
    flow_hybrids = np.asarray([getattr(fm, "reliability_hybrid", fm.reliability) for fm in flow_metrics], dtype=np.float64)
    flow_goodputs = np.asarray([fm.goodput_mbps for fm in flow_metrics], dtype=np.float64)
    flow_offered = np.asarray([max(0.0, fm.offered_rate_mbps) for fm in flow_metrics], dtype=np.float64)
    total_goodput = float(np.sum(flow_goodputs, dtype=np.float64)) if flow_goodputs.size else 0.0
    min_rel = float(np.min(flow_rels)) if flow_rels.size else 0.0
    mean_rel = float(np.mean(flow_rels, dtype=np.float64)) if flow_rels.size else 0.0
    if flow_rels.size and float(np.sum(flow_offered)) > 0.0:
        weighted_rel = float(np.sum(flow_rels * flow_offered, dtype=np.float64) / np.sum(flow_offered, dtype=np.float64))
    else:
        weighted_rel = mean_rel
    p_all = float(np.prod(flow_rels, dtype=np.float64)) if flow_rels.size else 0.0
    p_any = float(1.0 - np.prod(1.0 - flow_rels, dtype=np.float64)) if flow_rels.size else 0.0
    worst_delay = float(np.max([fm.delay_ms for fm in flow_metrics], initial=0.0)) if flow_metrics else float('inf')
    max_rho = float(np.max([fm.rho_max for fm in flow_metrics], initial=0.0)) if flow_metrics else 0.0
    min_rel_block = float(np.min(flow_rel_blocks)) if flow_rel_blocks.size else 0.0
    min_delivery_ratio = float(np.min(flow_delivery_ratios)) if flow_delivery_ratios.size else 0.0
    min_hybrid = float(np.min(flow_hybrids)) if flow_hybrids.size else 0.0

    out = MultiFlowMetrics(
        flows=flow_metrics,
        total_goodput_mbps=float(total_goodput),
        min_flow_reliability=float(min_rel),
        mean_flow_reliability=float(mean_rel),
        weighted_flow_reliability=float(weighted_rel),
        probability_all_flows_succeed=float(p_all),
        probability_at_least_one_flow_succeeds=float(p_any),
        worst_delay_ms=float(worst_delay),
        max_rho=float(max_rho),
        omega=omega,
    )
    out.reliability_block = float(min_rel_block)
    out.delivery_ratio = float(min_delivery_ratio)
    out.reliability_hybrid = float(min_hybrid)
    return out


def solve_modified_abreu_redundant(
    paths: Sequence[Sequence[int]],
    pos_m: np.ndarray,
    offered_rates_mbps: Sequence[float] | float,
    frame_size_bytes: int = 1500,
    buffer_size_packets: int = 20,
    phy_rate_mbps: float = 54.0,
    ber_per_path: Sequence[Sequence[float]] | None = None,
    carrier_sense_range_m: float = 199.4323529413177,
    slot_time_us: float = 9.0,
    difs_us: float = 28.0,
    sifs_us: float = 10.0,
    ack_us: float = 24.0,
    ack_timeout_us: float = 34.0,
    cw_min: int = 15,
    cw_max: int = 1023,
    max_retries: int = 6,
    max_attempts: int | None = 7,
    reliability_min: float = 0.95,
    delay_max_ms: float = 50.0,
    max_iters: int = 8,
    tol: float = 1e-3,
    phy_rates_per_path: Sequence[Sequence[float]] | None = None,
    offered_rates_reverse_mbps: Sequence[float] | float | None = None,
    compute_max_feasible_rate: bool = True,
    same_flow_discount: float = 0.5660007226003583,
    self_overlap_load_scale: float = 0.11292807059201448,
    cross_flow_load_scale: float = 0.5811699219640487,
    interference_to_freeze_scale: float = 0.4245764622910097,
    link_model: LinkModelParams | None = None,
    use_torch_backend: bool = False,
    torch_device: str | None = None,
) -> RedundantMetrics:
    del max_iters, tol, offered_rates_reverse_mbps

    paths = [list(map(int, p)) for p in paths if len(p) >= 2]
    if not paths:
        return RedundantMetrics([], 0.0, 1.0, float("inf"), 0.0, 0.0, np.zeros((0, 0), dtype=np.float32))

    if isinstance(offered_rates_mbps, (int, float)):
        req = FlowRequest(
            paths=paths,
            offered_rate_mbps=float(offered_rates_mbps),
            redundancy_mode="replicate_top_k",
            max_active_paths=len(paths),
            ber_per_path=ber_per_path,
            phy_rates_per_path=phy_rates_per_path,
            label="redundant_flow",
        )
    else:
        rates = np.asarray(list(offered_rates_mbps), dtype=np.float64)
        if rates.size < len(paths):
            last = float(rates[-1]) if rates.size > 0 else 0.0
            rates = np.pad(rates, (0, len(paths) - rates.size), constant_values=last)
        rates = rates[:len(paths)]
        req = FlowRequest(
            paths=paths,
            offered_rate_mbps=float(np.max(rates) if rates.size else 0.0),
            redundancy_mode="replicate_top_k",
            max_active_paths=len(paths),
            ber_per_path=ber_per_path,
            phy_rates_per_path=phy_rates_per_path,
            label="redundant_flow",
            per_path_offered_rates_mbps=rates.tolist(),
        )

    mm = solve_modified_abreu_multiflow(
        flows=[req],
        pos_m=pos_m,
        frame_size_bytes=frame_size_bytes,
        buffer_size_packets=buffer_size_packets,
        phy_rate_mbps=phy_rate_mbps,
        carrier_sense_range_m=carrier_sense_range_m,
        slot_time_us=slot_time_us,
        difs_us=difs_us,
        sifs_us=sifs_us,
        ack_us=ack_us,
        ack_timeout_us=ack_timeout_us,
        cw_min=cw_min,
        cw_max=cw_max,
        max_retries=max_retries,
        max_attempts=max_attempts,
        reliability_min=reliability_min,
        delay_max_ms=delay_max_ms,
        compute_max_feasible_rate=compute_max_feasible_rate,
        same_flow_discount=same_flow_discount,
        use_torch_backend=use_torch_backend,
        torch_device=torch_device,
        self_overlap_load_scale=self_overlap_load_scale,
        cross_flow_load_scale=cross_flow_load_scale,
        interference_to_freeze_scale=interference_to_freeze_scale,
        link_model=link_model,
    )
    if not mm.flows:
        return RedundantMetrics([], 0.0, 1.0, float("inf"), 0.0, 0.0, mm.omega)
    fm = mm.flows[0]
    return RedundantMetrics(
        paths=[ap.path_metrics for ap in fm.path_metrics],
        reliability=float(fm.reliability),
        loss_rate=float(fm.loss_rate),
        delay_ms=float(fm.delay_ms),
        goodput_mbps=float(fm.goodput_mbps),
        max_feasible_rate_mbps=float(fm.max_feasible_rate_mbps),
        omega=mm.omega,
        reliability_block=float(getattr(fm, "reliability_block", fm.reliability)),
        delivery_ratio=float(getattr(fm, "delivery_ratio", 0.0)),
        reliability_hybrid=float(getattr(fm, "reliability_hybrid", fm.reliability)),
    )


__all__ = [
    'PathMetrics', 'RedundantMetrics', 'FlowRequest', 'ActivePathMetrics', 'FlowMetrics', 'MultiFlowMetrics',
    'HopParams', 'QueueState', 'MacState', 'LinkModelParams',
    'enumerate_simple_paths', 'build_interference_matrix', 'solve_modified_abreu_redundant', 'solve_modified_abreu_multiflow',
    'debug_single_path_case', '_build_path_hop_params', '_fixed_point_single_path', '_link_ber_from_success_prob', '_rician_success_prob_from_avg_snr', '_link_success_prob_from_distance', '_infer_ber_per_link_from_geometry',
]
