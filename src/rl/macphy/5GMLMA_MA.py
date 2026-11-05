#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native PyTorch HAPPO trainer for Drone5G on Python 3.12.

This file intentionally avoids the legacy MARLlib/Ray 1.8 stack.
It implements the core HAPPO ingredients directly in PyTorch:
- one actor per agent (no parameter sharing),
- centralized critic over shared state,
- sequential actor updates with multiplicative correction factors,
- action masking,
- checkpoint / continue / best-model mirroring,
- TensorBoard logging compatible with the previous workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pipe, Process, get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _NullSummaryWriter:
    def __init__(self, log_dir: str | None = None, *args, **kwargs):
        self.log_dir = log_dir

    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def _make_summary_writer(log_dir: str):
    """Cria SummaryWriter somente quando habilitado.

    Com multiprocessing spawn, imports no topo são executados em todos os workers.
    Portanto o TensorBoard fica desativado por padrão para evitar queda dos workers
    quando o pacote tensorboard/tensorflow está inconsistente no ambiente.
    """
    enable_tb = str(os.environ.get("ENABLE_TENSORBOARD", "1")).strip().lower()
    if enable_tb not in {"1", "true", "yes", "on"}:
        print(
            "[INFO] TensorBoard SummaryWriter desativado manualmente (ENABLE_TENSORBOARD=0). "
            "Métricas JSON/console continuam ativas.",
            flush=True,
        )
        return _NullSummaryWriter(log_dir=log_dir)

    try:
        from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
        return _TorchSummaryWriter(log_dir=log_dir)
    except Exception as exc:
        print(
            "[WARN] TensorBoard SummaryWriter indisponível; seguindo sem escrita de eventos. "
            f"Erro: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return _NullSummaryWriter(log_dir=log_dir)


try:
    from .drone_env_multi_abreu import Drone5GHAPPOPyEnv
except ImportError:
    from drone_env_multi_abreu import Drone5GHAPPOPyEnv

MAX_DELAY_MS_DEFAULT = 300.0
RELIABILITY_TARGET_DEFAULT = 0.995
THROUGHPUT_MAX_MBPS_DEFAULT = 9600.0


# -----------------------------
# Utility helpers
# -----------------------------

def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"valor booleano inválido: {v}")


def _parse_hidden_sizes(s: str) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    return vals if vals else [256, 256]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_TERMINAL_INFO_KEYS: Tuple[str, ...] = (
    "reliability_terminal",
    "reliability_block_terminal",
    "reliability_initial",
    "repair_delta_terminal",
    "delivery_ratio_terminal",
    "reliability_hybrid_terminal",
    "throughput_terminal",
    "latency_terminal_ms",
    "active_drones_terminal",
    "qos_target_met_terminal",
    "path_exists_terminal",
    "bootstrap_path_success_terminal",
    "abreu_enabled_terminal",
    "num_paths_terminal",
    "success_terminal",
    "zone_clear_terminal",
    "zone_abs_pen_terminal",
    "zone_abs_pen_per_step_terminal",
    "zone_viol_frac_terminal",
    "zone_drones_mean_terminal",
    "zone_links_mean_terminal",
    "zone_drones_terminal",
    "zone_links_terminal",
    "zone_abs_pen_terminal",
    "zone_abs_pen_per_step_terminal",
    "zone_viol_frac_terminal",
    "zone_drones_mean_terminal",
    "zone_links_mean_terminal",
    "bad_deactivation_terminal",
    "prune_keep_path_terminal",
)
_AUX_INFO_KEYS: Tuple[str, ...] = (
    "_terminal",
    "_worker_restarted",
    "_worker_restart_reason",
    "_worker_restart_failed",
    "_worker_restart_exception",
)


def _compact_info_payload(info: Any, include_terminal: bool = False) -> Dict[str, Any]:
    if not isinstance(info, dict) or not info:
        return {}
    allowed_keys = _AUX_INFO_KEYS
    if include_terminal:
        allowed_keys = _AUX_INFO_KEYS + _TERMINAL_INFO_KEYS
    compact: Dict[str, Any] = {}
    for key in allowed_keys:
        if key in info:
            compact[key] = info[key]
    return compact


class _MetricAccumulator:
    __slots__ = ("sums", "counts", "values")

    def __init__(self, keys: Sequence[str]):
        self.sums = {str(k): 0.0 for k in keys}
        self.counts = {str(k): 0 for k in keys}
        # Valores individuais por chave (episódio a episódio): permitem std,
        # percentis e histogramas — a média sozinha é cega para a distribuição
        # BIMODAL deste problema (episódios ~1.0 vs ~0.35). Memória trivial
        # (~centenas de floats por update; o acumulador é recriado por janela).
        self.values = {str(k): [] for k in keys}

    def add(self, key: str, value: Any) -> None:
        v = _safe_float(value)
        # Auto-registro: aceita chaves NÃO declaradas na construção. Antes, uma
        # chave nova adicionada só à lista do ROLLOUT (ex.: zone_drones_terminal)
        # e não à do EVAL causava KeyError e DERRUBAVA o treino no 1º eval.
        if key not in self.sums:
            self.sums[key] = 0.0
            self.counts[key] = 0
            self.values[key] = []
        self.sums[key] += v
        self.counts[key] += 1
        self.values[key].append(v)

    def add_if_present(self, info: Dict[str, Any], key: str) -> None:
        if key in info:
            self.add(key, info[key])

    def mean(self, key: str) -> float:
        c = self.counts.get(key, 0)
        return float(self.sums.get(key, 0.0) / c) if c > 0 else 0.0

    def count(self, key: str) -> int:
        return int(self.counts.get(key, 0))

    def std(self, key: str) -> float:
        vals = self.values.get(key) or []
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def pctl(self, key: str, q: float) -> float:
        vals = self.values.get(key) or []
        return float(np.percentile(vals, q)) if vals else 0.0

    def values_of(self, key: str) -> list:
        return list(self.values.get(key) or [])


class RunningMeanStd:
    def __init__(self, shape: Sequence[int], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / max(tot_count, 1e-12)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / max(tot_count, 1e-12)
        new_var = m2 / max(tot_count, 1e-12)
        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)
        self.count = float(tot_count)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(y, -clip, clip)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


# -----------------------------
# Minimal vectorized env runner
# -----------------------------

def _worker(remote, parent_remote, env_config, seed):
    parent_remote.close()
    env = Drone5GHAPPOPyEnv(env_config)
    _auto_seed = {"next": None, "stride": 1}  # fluxo determinístico p/ auto-resets (eval)
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            try:
                cmd, data = remote.recv()
                if cmd == "step":
                    obs, rew, term, trunc, info = env.step(data)
                    done = bool(term or trunc)
                    if done:
                        compact_info = _compact_info_payload(dict(info or {}), include_terminal=True)
                        compact_info["_terminal"] = True
                        # Se um fluxo determinístico de seeds foi configurado no
                        # reset (avaliação), TODOS os episódios do worker usam
                        # seeds da sequência — não só o primeiro. Sem isso, dos
                        # 100 episódios do eval apenas os N primeiros (1 por
                        # env) eram semeados e o restante seguia o RNG livre,
                        # tornando as avaliações não-pareadas e ruidosas.
                        if _auto_seed["next"] is not None:
                            next_obs, _ = env.reset(seed=int(_auto_seed["next"]))
                            _auto_seed["next"] = int(_auto_seed["next"]) + int(_auto_seed["stride"])
                        else:
                            next_obs, _ = env.reset(seed=None)
                        remote.send((next_obs, float(rew), bool(term), bool(trunc), compact_info))
                    else:
                        remote.send((obs, float(rew), bool(term), bool(trunc), _compact_info_payload(info, include_terminal=False)))
                elif cmd == "reset":
                    if isinstance(data, dict):
                        obs, _ = env.reset(seed=data.get("seed"))
                        b = data.get("auto_seed_base")
                        _auto_seed["next"] = int(b) if b is not None else None
                        _auto_seed["stride"] = int(data.get("auto_seed_stride", 1))
                    else:
                        obs, _ = env.reset(seed=data)
                        _auto_seed["next"] = None
                        _auto_seed["stride"] = 1
                    remote.send((obs, {}))
                elif cmd == "close":
                    remote.close()
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                else:
                    raise RuntimeError(f"Comando desconhecido: {cmd}")
            except EOFError:
                break
            except BrokenPipeError:
                break
            except Exception as e:
                tb = traceback.format_exc()
                try:
                    remote.send(("__worker_error__", repr(e), tb))
                except Exception:
                    pass
                raise
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocVecEnvLite:
    def __init__(self, env_config: Dict[str, Any], num_envs: int, base_seed: int):
        assert num_envs >= 1
        self.num_envs = int(num_envs)
        self.ctx = get_context("spawn")
        self.env_config = dict(env_config)
        self.base_seed = int(base_seed)
        self.initial_seeds = [int(self.base_seed + idx * 1000) for idx in range(self.num_envs)]
        self.restart_counts = [0 for _ in range(self.num_envs)]
        self.remotes: List[Any] = [None] * self.num_envs
        self.work_remotes: List[Any] = [None] * self.num_envs
        self.processes: List[Process | None] = [None] * self.num_envs
        for idx in range(self.num_envs):
            self._spawn_one(idx)
        self._send_cmd(0, "get_spaces", None)
        self.observation_space, self.action_space = self._recv_cmd(0, allow_restart=False)

    def _seed_for_restart(self, idx: int) -> int:
        return int(self.initial_seeds[idx] + self.restart_counts[idx] * 10_000_000)

    def _spawn_one(self, idx: int) -> None:
        parent_remote, work_remote = Pipe()
        p = self.ctx.Process(target=_worker, args=(work_remote, parent_remote, self.env_config, self._seed_for_restart(idx)))
        p.daemon = True
        p.start()
        work_remote.close()
        old_remote = self.remotes[idx]
        if old_remote is not None:
            try:
                old_remote.close()
            except Exception:
                pass
        self.remotes[idx] = parent_remote
        self.work_remotes[idx] = work_remote
        self.processes[idx] = p

    def _terminate_one(self, idx: int) -> None:
        proc = self.processes[idx]
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=1.0)
            except Exception:
                pass
        self.processes[idx] = None
        remote = self.remotes[idx]
        if remote is not None:
            try:
                remote.close()
            except Exception:
                pass
        self.remotes[idx] = None
        self.work_remotes[idx] = None

    def _is_alive(self, idx: int) -> bool:
        proc = self.processes[idx]
        return (proc is not None) and proc.is_alive()

    def _restart_one(self, idx: int, reset_seed: int | None = None, reason: str = "") -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.restart_counts[idx] += 1
        self._terminate_one(idx)
        self._spawn_one(idx)
        info: Dict[str, Any] = {"_worker_restarted": True, "_worker_restart_reason": str(reason or "unknown")}
        try:
            self._send_cmd(idx, "reset", reset_seed)
            obs, reset_info = self._recv_cmd(idx, allow_restart=False)
            if isinstance(reset_info, dict):
                info.update(reset_info)
            return obs, info
        except Exception as e:
            info["_worker_restart_failed"] = True
            info["_worker_restart_exception"] = repr(e)
            raise RuntimeError(f"Falha ao reiniciar worker {idx}: {e}") from e

    def _send_cmd(self, idx: int, cmd: str, data: Any) -> None:
        if not self._is_alive(idx):
            raise RuntimeError(f"worker {idx} morto antes de enviar comando {cmd}")
        remote = self.remotes[idx]
        try:
            remote.send((cmd, data))
        except Exception as e:
            raise RuntimeError(f"falha ao enviar comando {cmd} para worker {idx}: {e}") from e

    def _recv_cmd(self, idx: int, allow_restart: bool = True) -> Any:
        remote = self.remotes[idx]
        try:
            out = remote.recv()
        except EOFError as e:
            raise RuntimeError(f"worker {idx} fechou o pipe (EOF)") from e
        except Exception as e:
            raise RuntimeError(f"falha ao receber de worker {idx}: {e}") from e
        if isinstance(out, tuple) and len(out) >= 1 and out[0] == "__worker_error__":
            _, err_repr, tb = out
            raise RuntimeError(f"worker {idx} erro interno: {err_repr}\n{tb}")
        return out

    def reset(self, seeds: Sequence[int] | None = None):
        if seeds is None:
            seeds = [None] * self.num_envs
        results: List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = []
        for idx, seed in enumerate(seeds):
            try:
                self._send_cmd(idx, "reset", seed)
                results.append(self._recv_cmd(idx))
            except Exception as e:
                obs, info = self._restart_one(idx, reset_seed=seed, reason=f"reset:{e}")
                results.append((obs, info))
        obs_list, info_list = zip(*results)
        return self._stack_obs(obs_list), list(info_list)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        results: List[Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]] = [None] * self.num_envs
        pending: List[int] = []
        for idx, action in enumerate(actions):
            try:
                self._send_cmd(idx, "step", np.asarray(action, dtype=np.int64))
                pending.append(idx)
            except Exception as e:
                obs, info = self._restart_one(idx, reset_seed=None, reason=f"step_send:{e}")
                info = dict(info or {})
                info["_terminal"] = True
                results[idx] = (obs, 0.0, False, True, info)
        for idx in pending:
            try:
                out = self._recv_cmd(idx)
                results[idx] = out
            except Exception as e:
                obs, info = self._restart_one(idx, reset_seed=None, reason=f"step_recv:{e}")
                info = dict(info or {})
                info["_terminal"] = True
                results[idx] = (obs, 0.0, False, True, info)
        obs, rew, term, trunc, infos = zip(*results)
        return self._stack_obs(obs), np.asarray(rew, dtype=np.float32), np.asarray(term, dtype=np.bool_), np.asarray(trunc, dtype=np.bool_), list(infos)

    def close(self):
        for idx, r in enumerate(self.remotes):
            if r is not None:
                try:
                    r.send(("close", None))
                except Exception:
                    pass
        for idx in range(self.num_envs):
            proc = self.processes[idx]
            if proc is not None:
                try:
                    proc.join(timeout=2.0)
                except Exception:
                    pass
                if proc.is_alive():
                    try:
                        proc.terminate()
                        proc.join(timeout=1.0)
                    except Exception:
                        pass
        for idx, r in enumerate(self.remotes):
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass

    @staticmethod
    def _stack_obs(obs_list: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = obs_list[0].keys()
        return {k: np.stack([np.asarray(o[k]) for o in obs_list], axis=0) for k in keys}


# -----------------------------
# Neural networks
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int], out_dim: int, activate_last: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, int(h)))
            layers.append(nn.ReLU())
            last = int(h)
        layers.append(nn.Linear(last, out_dim))
        if activate_last:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.model = MLP(obs_dim, hidden_sizes, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    def dist(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> Categorical:
        logits = self.forward(obs)
        if action_mask is not None:
            invalid = action_mask <= 0.5
            fill_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(invalid, fill_value)
        return Categorical(logits=logits)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.model = MLP(state_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state).squeeze(-1)


@dataclass
class RolloutBatch:
    obs: np.ndarray
    state: np.ndarray
    action_mask: np.ndarray
    alive_mask: np.ndarray
    turn_mask: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    # truncations[t]=1.0 marca término por TIMEOUT real. Término por SUCESSO
    # (KPI atingido) tem dones=1 mas truncations=0: nesse caso o GAE faz
    # BOOTSTRAP do valor futuro em vez de zerá-lo, removendo a arbitragem de
    # "evitar o sucesso para alongar o episódio".
    truncations: np.ndarray
    values: np.ndarray
    next_state: np.ndarray


# -----------------------------
# Trainer
# -----------------------------

class NativeHAPPOTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        _set_seed(args.seed)
        self.device = self._resolve_device(args.device)

        env_cfg = self._build_env_config(args)
        self.env_cfg = env_cfg
        # PRÉ-CALIBRAÇÃO no processo pai: as varreduras analíticas (linha mínima
        # e baseline de drones p/ QoS) custam ~25s e são idênticas em todos os
        # workers. Computa UMA vez aqui e injeta no cfg; os 128+ workers pulam a
        # varredura no 1º reset. Sem isso o lançamento vira uma tempestade de
        # calibrações simultâneas que parece um travamento de dezenas de minutos.
        try:
            import time as _time
            _t0 = _time.time()
            print("[CALIB] pré-computando calibrações analíticas no processo pai...", flush=True)
            _probe = Drone5GHAPPOPyEnv(dict(env_cfg))
            _probe.reset(seed=int(args.seed))
            _base = getattr(_probe, "base_env", None) or getattr(_probe, "env", None)
            _line = int(_base._analytic_min_line_hops())
            _dmin = int(_base._analytic_min_qos_drones())
            env_cfg = {**env_cfg,
                       "min_line_hops_precomputed": _line,
                       "min_qos_drones_precomputed": _dmin}
            try:
                _probe.close()
            except Exception:
                pass
            del _probe
            self.env_cfg = env_cfg  # mantém o cfg injetado para reinicios de workers
            print(f"[CALIB] linha={_line} hops | baseline_qos={_dmin} drones | {_time.time()-_t0:.1f}s (workers pulam a varredura)", flush=True)
        except Exception as _e:
            print(f"[CALIB] pré-computação falhou ({_e}); workers calibrarão localmente (lançamento mais lento)", flush=True)
        self.envs = SubprocVecEnvLite(env_cfg, num_envs=args.num_workers, base_seed=args.seed)
        self.eval_envs = SubprocVecEnvLite(env_cfg, num_envs=max(1, args.eval_envs), base_seed=args.seed + 500_000)

        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        obs_spaces = getattr(self.obs_space, "spaces", self.obs_space)
        self.max_agents = int(obs_spaces["obs"].shape[0])
        self.obs_dim = int(obs_spaces["obs"].shape[1])
        self.state_dim = int(obs_spaces["state"].shape[0])
        self.act_dim = int(obs_spaces["action_mask"].shape[1])
        hidden = _parse_hidden_sizes(args.policy_arch)
        self._configure_torch_runtime()
        self.actors = nn.ModuleList([ActorNet(self.obs_dim, self.act_dim, hidden) for _ in range(self.max_agents)]).to(self.device)
        self.critic = CriticNet(self.state_dim, hidden).to(self.device)
        self.actor_opts = [torch.optim.Adam(actor.parameters(), lr=args.lr_start) for actor in self.actors]
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr_start)
        self.amp_enabled = self._resolve_amp_enabled()
        self.amp_dtype = self._resolve_amp_dtype()
        self.actor_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.device.type == "cuda")
        self.critic_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.device.type == "cuda")
        self.actor_runners = [self._maybe_compile_module(actor, tag=f"actor_{i}") for i, actor in enumerate(self.actors)]
        self.critic_runner = self._maybe_compile_module(self.critic, tag="critic")

        self.obs_rms = RunningMeanStd((self.obs_dim,)) if args.obs_norm else None
        self.state_rms = RunningMeanStd((self.state_dim,)) if args.obs_norm else None

        self.global_step = 0
        self.update_idx = 0
        self.episodes_finished = 0
        self.best_metric = -float("inf") if args.best_mode == "max" else float("inf")
        # Estado para o critério lexicográfico (best_metric == "lex_qos_reward"):
        # QoS é o critério PRIMÁRIO (não pode piorar além de uma tolerância) e a
        # recompensa é o DESEMPATE (entre modelos de QoS equivalente, salva o de
        # maior recompensa). Protege contra reward hacking (recompensa subindo
        # com QoS caindo NÃO salva) e garante progresso contínuo (quando o QoS
        # empata, a recompensa ainda move o best).
        self.best_qos = -float("inf")
        self.best_reward = -float("inf")
        # Nível lexicográfico PRIMÁRIO (acima do qos): fração de episódios com
        # zona totalmente evacuada (drones E links fora). Como violar zona é
        # inaceitável por definição, um modelo que respeita as zonas deve vencer
        # um que não respeita, MESMO com qos menor. Sem isto, o best ficava
        # preso num modelo antigo de qos alto que ainda deixava links cruzando.
        self.best_zone_clear = -float("inf")
        self.best_lex_tol = float(getattr(args, "best_lex_tol", 0.03))  # tolerância: QoS não pode cair mais que isso
        self._lex_pass_streak = 0
        self._lex_window_qos = float("inf")
        self._lex_window_zc = float("inf")
        self._lex_window_reward = float("inf")
        self.best_step = 0
        self.best_phase_id = self._phase_identity(args)

        self.run_dir, self.ckpt_dir, self.artifacts_dir, self.tb_dir, self.best_dir = self._init_dirs(args)
        self.writer = _make_summary_writer(log_dir=str(self.tb_dir))
        (self.artifacts_dir / "config.json").write_text(json.dumps(self._serializable_args(args), indent=2, ensure_ascii=False))
        self._write_algorithm_validation()

        self._save_frequency_updates = self._resolve_checkpoint_freq_updates()
        self._eval_frequency_updates = max(1, int(args.eval_interval_updates))
        self._last_rollout_metrics: Dict[str, float] = {}
        self._nonfinite_counts: Dict[str, int] = {}
        self._rollout_buffer_shape: Tuple[int, int, int, int, int, int] | None = None
        self._rollout_buffers: Dict[str, np.ndarray] = {}

        self._maybe_restore()

    # ---------- init helpers ----------
    def _resolve_device(self, device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _configure_torch_runtime(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(self.args.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(self.args.allow_tf32)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _resolve_amp_enabled(self) -> bool:
        mode = str(self.args.amp).strip().lower()
        if mode == "auto":
            return self.device.type == "cuda"
        return mode in {"1", "true", "yes", "on"}

    def _resolve_amp_dtype(self) -> torch.dtype:
        mode = str(self.args.amp_dtype).strip().lower()
        if mode == "auto":
            return torch.float16 if self.device.type == "cuda" else torch.bfloat16
        if mode in {"fp16", "float16", "half"}:
            return torch.float16
        if mode in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float32

    def _autocast_ctx(self):
        if not self.amp_enabled:
            return nullcontext()
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        if self.device.type == "cpu" and self.amp_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    def _maybe_compile_module(self, module: nn.Module, tag: str) -> nn.Module:
        mode = str(self.args.compile_models).strip().lower()
        should_compile = (self.device.type == "cuda") if mode == "auto" else (mode in {"1", "true", "yes", "on"})
        if not should_compile:
            return module
        if not hasattr(torch, "compile"):
            return module
        try:
            compiled = torch.compile(module, mode=str(self.args.compile_mode))
            return compiled
        except Exception as exc:
            print(f"[WARN] torch.compile falhou para {tag}: {exc}", flush=True)
            return module

    def _to_device_tensor(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        arr = np.ascontiguousarray(array)
        tensor = torch.from_numpy(arr)
        if self.device.type == "cuda":
            try:
                tensor = tensor.pin_memory()
            except Exception:
                pass
            return tensor.to(device=self.device, dtype=dtype, non_blocking=True)
        return tensor.to(device=self.device, dtype=dtype)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        invalid = action_mask <= 0.5
        fill_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(invalid, fill_value)

    def _actor_dist(self, aid: int, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> Categorical:
        logits = self.actor_runners[aid](obs)
        logits = self._sanitize_torch(logits, f"actor_logits_a{aid}", nan=0.0, posinf=1e4, neginf=-1e4)
        logits = self._masked_logits(logits, action_mask)
        logits = self._sanitize_torch(logits, f"actor_logits_masked_a{aid}", nan=0.0, posinf=1e4, neginf=-1e4)
        return Categorical(logits=logits)

    def _compiled_step_begin(self) -> None:
        if self.device.type != "cuda":
            return
        compiler_mod = getattr(torch, "compiler", None)
        mark_fn = getattr(compiler_mod, "cudagraph_mark_step_begin", None) if compiler_mod is not None else None
        if mark_fn is None:
            return
        try:
            mark_fn()
        except Exception:
            pass

    def _critic_forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_runner(state)

    def _record_nonfinite(self, name: str, count: int) -> None:
        if int(count) <= 0:
            return
        if not hasattr(self, "_nonfinite_counts"):
            self._nonfinite_counts = {}
        self._nonfinite_counts[name] = int(self._nonfinite_counts.get(name, 0)) + int(count)

    def _sanitize_np(self, arr: np.ndarray, name: str, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> np.ndarray:
        arr = np.asarray(arr)
        bad = ~np.isfinite(arr)
        if np.any(bad):
            self._record_nonfinite(name, int(np.count_nonzero(bad)))
            arr = np.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)
        return arr

    def _sanitize_torch(self, tensor: torch.Tensor, name: str, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        bad = (~torch.isfinite(tensor)).sum().item()
        self._record_nonfinite(name, int(bad))
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)

    def _build_env_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "map_name": "Drone5G-NativeHAPPO",
            "episode_horizon": int(args.episode_horizon),
            "mask_actions": int(args.mask_actions),
            "max_drones": int(args.max_drones),
            "num_drones": int(args.max_drones),
            "max_out_connections": int(args.max_out_connections),
            "zonas_count": int(args.zonas_count),
            "zona_w": int(args.zona_w),
            "zona_h": int(args.zona_h),
            "zona_single_fixed": int(args.zona_single_fixed),
            "p_sem_zona": float(args.p_sem_zona),
            "obs_zone_map": int(args.obs_zone_map),
            "zone_map_scale": int(args.zone_map_scale),
            "obs_edge_cross": int(args.obs_edge_cross),
            "obs_link_quality": int(getattr(args, "obs_link_quality", 1)),
            "converge_on_state": int(getattr(args, "converge_on_state", 1)),
            "latency_headroom_penalty": float(getattr(args, "latency_headroom_penalty", 0.5)),
            "rel_ramp_ext_w": float(getattr(args, "rel_ramp_ext_w", 0.0)),
            "zone_violation_penalty": float(getattr(args, "zone_violation_penalty", 2.0)),
            "zone_link_weight_mult": float(getattr(args, "zone_link_weight_mult", 3.5)),
            "zone_success_suppression": float(getattr(args, "zone_success_suppression", 1.0)),
            "zone_violation_abs_penalty": float(getattr(args, "zone_violation_abs_penalty", 0.0)),
            "latency_from_longest_route": int(getattr(args, "latency_from_longest_route", 0)),
            "calibration_shadow_k": int(getattr(args, "calibration_shadow_k", 21)),
            "calibration_min_drones_quantile": float(getattr(args, "calibration_min_drones_quantile", 0.85)),
            "calibration_min_drones_cap": int(getattr(args, "calibration_min_drones_cap", 14)),
            "p_zone_violation_seed": float(getattr(args, "p_zone_violation_seed", 0.0)),
            "abreu_enabled": int(args.abreu_enabled),
            "abreu_max_paths": int(args.abreu_max_paths),
            "traffic_rate_mbps": float(args.traffic_rate_mbps),
            "frame_size_bytes": int(args.frame_size_bytes),
            "buffer_size_packets": int(args.buffer_size_packets),
            "phy_rate_mbps": float(args.phy_rate_mbps),
            "slot_time_us": float(args.slot_time_us),
            "difs_us": float(args.difs_us),
            "sifs_us": float(args.sifs_us),
            "ack_us": float(args.ack_us),
            "ack_timeout_us": float(args.ack_timeout_us),
            "cw_min": int(args.cw_min),
            "cw_max": int(args.cw_max),
            "max_retries": int(args.max_retries),
            "carrier_sense_range_m": float(args.carrier_sense_range_m),
            "abreu_reliability_min": float(args.abreu_reliability_min),
            "abreu_delay_max_ms": float(args.abreu_delay_max_ms),
            "success_max_latency_ms": float(args.max_delay_ms),
            "reliability_target": float(args.reliability_target),
            "delivery_ratio_target": float(args.delivery_ratio_target),
            "prune_delivery_min": float(args.prune_delivery_min),
            "prune_rel_min": float(args.prune_rel_min),
            "drone_active_cost_before_qos": float(args.drone_active_cost_before_qos),
            "bad_deactivation_penalty": float(args.bad_deactivation_penalty),
            "max_delay_ms": float(args.max_delay_ms),
            "throughput_max_mbps": float(args.throughput_max_mbps),
            "abreu_solver_gpu": bool(args.abreu_solver_gpu),
            "seed_initial_path": bool(args.seed_initial_path),
            "seed_randomness": float(args.seed_randomness),
            "link_neighbor_radius_m": float(args.link_neighbor_radius_m),
            "seed_num_paths": int(args.seed_num_paths),
            "seed_target_hop_m": float(args.seed_target_hop_m),
            "seed_viability_slack": float(args.seed_viability_slack),
            "seed_line_min": bool(args.seed_line_min),
            "drone_potential_cost": float(args.drone_potential_cost),
            "activate_spawn_near_line": bool(args.activate_spawn_near_line),
            # Recompensa híbrida: hybrid_gamma DEVE ser o gamma do GAE para a
            # garantia do potential-based shaping valer.
            "use_hybrid_reward": bool(args.use_hybrid_reward),
            "hybrid_gamma": float(args.gamma),
            "qos_maintain_bonus": float(args.qos_maintain_bonus),
            "converge_patience": int(args.converge_patience),
            "converge_eps": float(args.converge_eps),
        }

    @staticmethod
    def _phase_identity(args: argparse.Namespace) -> str:
        """Identidade da fase do currículo, derivada dos args que o start.sh já
        envia. Usada para (a) rezerar o baseline lexicográfico ao trocar de fase
        e (b) separar o best_dir por fase."""
        idx = int(getattr(args, "curriculum_phase_idx", -1))
        name = str(getattr(args, "curriculum_phase_name", "") or "").strip()
        if idx < 0 and not name:
            return ""
        return f"{idx}_{name}" if name else str(idx)

    def _init_dirs(self, args: argparse.Namespace):
        root = Path(args.results_root).expanduser().resolve()
        prefix = args.run_name if not args.instance_tag else f"{args.run_name}_{args.instance_tag}"
        preferred = root / prefix
        is_resume = bool(args.resume) or str(args.use_model).lower() in {"continue", "resume"}
        if is_resume:
            # RESUME: reaproveita o MESMO diretório do treino anterior para que os
            # logs do TensorBoard sejam ANEXADOS (sem criar novo conjunto de
            # gráficos). Procura: (1) o nome limpo 'prefix'; senão (2) o diretório
            # mais recente que casa com 'prefix*' (caso o treino original tenha
            # sido criado com timestamp).
            if preferred.exists():
                run_dir = preferred
            else:
                candidatos = sorted(
                    [p for p in root.glob(f"{prefix}*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime, reverse=True)
                run_dir = candidatos[0] if candidatos else preferred
                if candidatos:
                    print(f"[INFO] RESUME: anexando logs ao run anterior: {run_dir}")
        elif preferred.exists() and any(preferred.iterdir()):
            run_dir = root / f"{prefix}_{_timestamp()}"
        else:
            run_dir = preferred
        ckpt_dir = run_dir / "checkpoints"
        artifacts_dir = run_dir / "artifacts"
        tb_dir = run_dir / "tb_compat"
        best_dir = Path(args.best_model_dir).expanduser().resolve() if args.best_model_dir else (Path.cwd() / "ppo_best_model").resolve()
        # BEST POR FASE: sem isto há um único best global, e o best de uma fase
        # é sobrescrito pelo da seguinte — perdendo o melhor modelo de cada
        # etapa do currículo (que a tese precisa para comparação). Cada fase
        # tem sua própria dificuldade, então cada uma merece seu próprio best.
        if int(getattr(args, "best_per_phase", 1)):
            pid = self._phase_identity(args).strip()
            if pid:
                safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", pid).strip("_")
                if safe:
                    best_dir = best_dir.parent / f"{best_dir.name}_{safe}"
        for d in (run_dir, ckpt_dir, artifacts_dir, tb_dir, best_dir):
            d.mkdir(parents=True, exist_ok=True)
        return run_dir, ckpt_dir, artifacts_dir, tb_dir, best_dir

    def _write_algorithm_validation(self):
        payload = {
            "runtime": "native_pytorch_happo",
            "algorithm": "happo",
            "python": sys.version,
            "torch_version": torch.__version__,
            "device": str(self.device),
            "amp_enabled": bool(self.amp_enabled),
            "amp_dtype": str(self.amp_dtype),
            "compile_models": str(self.args.compile_models),
            "compile_mode": str(self.args.compile_mode),
            "allow_tf32": bool(self.args.allow_tf32),
            "notes": [
                "One actor per agent (no parameter sharing).",
                "Centralized critic on shared state.",
                "Sequential actor update with multiplicative factor correction.",
                "Turn-based environment integrated through active turn masks.",
            ],
        }
        (self.artifacts_dir / "algorithm_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def _resolve_checkpoint_freq_updates(self) -> int:
        if int(self.args.checkpoint_freq_updates) > 0:
            return int(self.args.checkpoint_freq_updates)
        steps_per_update = int(self.args.num_workers) * int(self.args.rollout_steps)
        target = max(1, int(self.args.checkpoint_target_timesteps))
        return max(1, int(round(target / max(1, steps_per_update))))

    def _serializable_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    def _restore_candidates(self) -> List[Path]:
        cands = []
        if self.args.restore_model_path:
            p = Path(self.args.restore_model_path).expanduser().resolve()
            if p.exists():
                cands.append(p)
        latest = self.ckpt_dir / "checkpoint_latest.pt"
        if latest.exists():
            cands.append(latest)
        cands.extend(sorted(self.ckpt_dir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        return cands

    def _maybe_restore(self):
        should_resume = bool(self.args.resume) or str(self.args.use_model).lower() in {"continue", "resume"}
        chosen = None
        last_error = ""
        if should_resume or self.args.restore_model_path:
            cands = self._restore_candidates()
            seed_path = ""
            if self.args.restore_model_path and not should_resume:
                seed_path = str(Path(self.args.restore_model_path).expanduser().resolve())
            for cand in cands:
                try:
                    # semeadura (fine-tuning de outro run) x resume (mesmo run)
                    self._load_checkpoint(cand, weights_only_seed=(str(cand) == seed_path))
                    chosen = cand
                    break
                except Exception as exc:
                    last_error = repr(exc)
                    print(f"[WARN] Falha ao restaurar checkpoint {cand}: {exc}", flush=True)
        payload = {
            "resume_requested": bool(should_resume),
            "restore_model_path": str(self.args.restore_model_path),
            "selected_checkpoint": str(chosen) if chosen is not None else "",
            "restore_last_error": str(last_error),
        }
        (self.artifacts_dir / "restore_resolution.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ---------- normalization ----------
    def _update_normalizers(self, obs: np.ndarray, state: np.ndarray, alive_mask: np.ndarray) -> None:
        if self.obs_rms is None or self.state_rms is None:
            return
        alive = alive_mask.reshape(-1) > 0.5
        obs_flat = obs.reshape(-1, self.obs_dim)
        if alive.any():
            self.obs_rms.update(obs_flat[alive])
        self.state_rms.update(state.reshape(-1, self.state_dim))

    def _norm_obs_tensor(self, obs_np: np.ndarray) -> torch.Tensor:
        if self.obs_rms is not None:
            obs_np = self.obs_rms.normalize(obs_np)
        return self._to_device_tensor(obs_np, dtype=torch.float32)

    def _norm_state_tensor(self, state_np: np.ndarray) -> torch.Tensor:
        if self.state_rms is not None:
            state_np = self.state_rms.normalize(state_np)
        return self._to_device_tensor(state_np, dtype=torch.float32)

    # ---------- rollout / train ----------
    @torch.no_grad()
    def _select_actions(self, obs_batch: Dict[str, np.ndarray], deterministic: bool = False):
        obs = obs_batch["obs"]
        state = obs_batch["state"]
        action_mask = obs_batch["action_mask"]
        turn_mask = obs_batch["turn_mask"]
        alive_mask = obs_batch["alive_mask"]
        # As estatísticas de normalização só são atualizadas durante a COLETA de
        # treino. No eval (deterministic=True) a atualização causava: (a) vazamento
        # das observações de avaliação para o normalizador do treino; (b) avaliação
        # não-pura (a "política efetiva" = rede + RMS mudava DURANTE o próprio
        # eval); (c) o checkpoint espelhado como best carregava um RMS deslocado
        # pelo eval que gerou a métrica.
        if not deterministic:
            self._update_normalizers(obs, state, alive_mask)
        n_envs = obs.shape[0]
        obs_t = self._norm_obs_tensor(obs)
        mask_t = self._to_device_tensor(action_mask, dtype=torch.float32)
        state_t = self._norm_state_tensor(state)
        turn_active = np.asarray(turn_mask[..., 0] > 0.5, dtype=np.bool_)
        active_aids = np.flatnonzero(np.any(turn_active, axis=0)).astype(np.int64, copy=False)

        self._compiled_step_begin()
        with self._autocast_ctx():
            values_t = self._sanitize_torch(self._critic_forward(state_t).clone(), "critic_values_select", nan=0.0, posinf=0.0, neginf=0.0)
            action_t = torch.zeros((n_envs, self.max_agents), dtype=torch.long, device=self.device)
            logprob_t = torch.zeros((n_envs, self.max_agents), dtype=torch.float32, device=self.device)
            for aid in active_aids.tolist():
                env_idx_np = np.flatnonzero(turn_active[:, aid]).astype(np.int64, copy=False)
                if env_idx_np.size == 0:
                    continue
                env_idx_t = torch.as_tensor(env_idx_np, dtype=torch.long, device=self.device)
                obs_sel = obs_t.index_select(0, env_idx_t)[:, aid, :]
                mask_sel = mask_t.index_select(0, env_idx_t)[:, aid, :]
                dist = self._actor_dist(aid, obs_sel, mask_sel)
                if deterministic:
                    act_sel = torch.argmax(dist.logits, dim=-1)
                else:
                    act_sel = dist.sample()
                lp_sel = self._sanitize_torch(dist.log_prob(act_sel).float(), f"actor_logprob_a{aid}", nan=0.0, posinf=0.0, neginf=0.0)
                aid_idx_t = torch.full_like(env_idx_t, int(aid))
                action_t.index_put_((env_idx_t, aid_idx_t), act_sel)
                logprob_t.index_put_((env_idx_t, aid_idx_t), lp_sel)
        actions = action_t.detach().cpu().numpy().astype(np.int64, copy=False)
        logprobs = self._sanitize_np(logprob_t.detach().cpu().numpy().astype(np.float32, copy=False), "actor_logprobs_np", nan=0.0, posinf=0.0, neginf=0.0)
        values = self._sanitize_np(values_t.detach().float().cpu().numpy().astype(np.float32, copy=False), "critic_values_np", nan=0.0, posinf=0.0, neginf=0.0)
        return actions, logprobs, values

    def collect_rollout(self, obs_batch: Dict[str, np.ndarray]) -> Tuple[RolloutBatch, Dict[str, float], Dict[str, np.ndarray]]:
        T = int(self.args.rollout_steps)
        N = int(self.args.num_workers)
        A = int(self.max_agents)
        O = int(self.obs_dim)
        S = int(self.state_dim)
        M = int(self.act_dim)

        rollout_buf = self._get_rollout_buffers(T, N, A, O, S, M)
        obs_buf = rollout_buf["obs"]
        state_buf = rollout_buf["state"]
        amask_buf = rollout_buf["action_mask"]
        alive_buf = rollout_buf["alive_mask"]
        turn_buf = rollout_buf["turn_mask"]
        actions_buf = rollout_buf["actions"]
        logprob_buf = rollout_buf["logprobs"]
        rewards_buf = rollout_buf["rewards"]
        dones_buf = rollout_buf["dones"]
        truncs_buf = rollout_buf["truncations"]
        values_buf = rollout_buf["values"]

        metric_keys = (
            "reliability_terminal",
            "reliability_block_terminal",
            "reliability_initial",
            "repair_delta_terminal",
            "delivery_ratio_terminal",
            "reliability_hybrid_terminal",
            "throughput_terminal",
            "latency_terminal_ms",
            "active_drones_terminal",
            "qos_target_met_terminal",
            "path_exists_terminal",
            "bootstrap_path_success_terminal",
            "abreu_enabled_terminal",
            "num_paths_terminal",
            "success_terminal",
            "bad_deactivation_terminal",
            "prune_keep_path_terminal",
            "episode_reward",
        )
        terminal_metrics = _MetricAccumulator(metric_keys)
        episode_returns = np.zeros((N,), dtype=np.float32)

        cur = obs_batch
        for t in range(T):
            obs_buf[t] = cur["obs"]
            state_buf[t] = cur["state"]
            amask_buf[t] = cur["action_mask"]
            alive_buf[t] = cur["alive_mask"]
            turn_buf[t] = cur["turn_mask"]
            acts, lps, vals = self._select_actions(cur, deterministic=False)
            next_obs, rewards, terms, truncs, infos = self.envs.step(acts)
            rewards = self._sanitize_np(np.asarray(rewards, dtype=np.float32), "rollout_rewards", nan=0.0, posinf=0.0, neginf=0.0)
            vals = self._sanitize_np(np.asarray(vals, dtype=np.float32), "rollout_values", nan=0.0, posinf=0.0, neginf=0.0)
            done = np.logical_or(terms, truncs)
            actions_buf[t] = acts
            logprob_buf[t] = lps
            rewards_buf[t] = rewards
            values_buf[t] = vals
            dones_buf[t] = done.astype(np.float32)
            truncs_buf[t] = np.asarray(truncs, dtype=np.float32)
            episode_returns += rewards
            self.global_step += int(N)
            for env_i, info in enumerate(infos):
                if bool(done[env_i]):
                    self.episodes_finished += 1
                    terminal_metrics.add("episode_reward", float(episode_returns[env_i]))
                    episode_returns[env_i] = 0.0
                    path_ok_terminal = bool(_safe_int(info.get("path_exists_terminal", 0), 0))
                    abreu_enabled_terminal = bool(_safe_int(info.get("abreu_enabled_terminal", 0), 0))
                    bootstrap_path_success = bool(_safe_int(info.get("bootstrap_path_success_terminal", 0), 0))
                    # Métricas físicas de QoS só são agregadas quando há caminho e o solver Abreu está ativo.
                    # No bootstrap (Abreu desligado), acompanhamos o sucesso de conectividade em métrica separada.
                    if path_ok_terminal and abreu_enabled_terminal:
                        terminal_metrics.add_if_present(info, "reliability_terminal")
                        terminal_metrics.add_if_present(info, "reliability_block_terminal")
                        terminal_metrics.add_if_present(info, "reliability_initial")
                        terminal_metrics.add_if_present(info, "repair_delta_terminal")
                        terminal_metrics.add_if_present(info, "delivery_ratio_terminal")
                        terminal_metrics.add_if_present(info, "reliability_hybrid_terminal")
                        terminal_metrics.add_if_present(info, "throughput_terminal")
                        terminal_metrics.add_if_present(info, "latency_terminal_ms")
                    else:
                        terminal_metrics.add("reliability_terminal", 0.0)
                        terminal_metrics.add("reliability_block_terminal", 0.0)
                        terminal_metrics.add("delivery_ratio_terminal", 0.0)
                        terminal_metrics.add("reliability_hybrid_terminal", 0.0)
                    terminal_metrics.add_if_present(info, "active_drones_terminal")
                    terminal_metrics.add_if_present(info, "qos_target_met_terminal")
                    terminal_metrics.add_if_present(info, "path_exists_terminal")
                    terminal_metrics.add("bootstrap_path_success_terminal", 1.0 if bootstrap_path_success else 0.0)
                    terminal_metrics.add("abreu_enabled_terminal", 1.0 if abreu_enabled_terminal else 0.0)
                    terminal_metrics.add_if_present(info, "num_paths_terminal")
                    terminal_metrics.add_if_present(info, "success_terminal")
                    # ZONAS: quantos drones/links ficaram DENTRO da zona no final
                    terminal_metrics.add_if_present(info, "zone_drones_terminal")
                    terminal_metrics.add_if_present(info, "zone_links_terminal")
                    terminal_metrics.add_if_present(info, "zone_clear_terminal")
                    terminal_metrics.add_if_present(info, "zone_abs_pen_terminal")
                    terminal_metrics.add_if_present(info, "zone_abs_pen_per_step_terminal")
                    terminal_metrics.add_if_present(info, "zone_viol_frac_terminal")
                    terminal_metrics.add_if_present(info, "zone_drones_mean_terminal")
                    terminal_metrics.add_if_present(info, "zone_links_mean_terminal")
                    terminal_metrics.add_if_present(info, "bad_deactivation_terminal")
                    terminal_metrics.add_if_present(info, "prune_keep_path_terminal")
            cur = next_obs

        episodes_finished = terminal_metrics.count("episode_reward")
        reward_from_terminal = episodes_finished > 0
        reliability_count = terminal_metrics.count("reliability_terminal")
        reliability_block_count = terminal_metrics.count("reliability_block_terminal")
        reliability_initial_count = terminal_metrics.count("reliability_initial")
        delivery_ratio_count = terminal_metrics.count("delivery_ratio_terminal")
        reliability_hybrid_count = terminal_metrics.count("reliability_hybrid_terminal")
        latency_count = terminal_metrics.count("latency_terminal_ms")
        throughput_count = terminal_metrics.count("throughput_terminal")
        active_drones_count = terminal_metrics.count("active_drones_terminal")
        qos_target_met_count = terminal_metrics.count("qos_target_met_terminal")
        path_exists_count = terminal_metrics.count("path_exists_terminal")
        bootstrap_path_success_count = terminal_metrics.count("bootstrap_path_success_terminal")
        abreu_enabled_count = terminal_metrics.count("abreu_enabled_terminal")
        num_paths_count = terminal_metrics.count("num_paths_terminal")
        success_count = terminal_metrics.count("success_terminal")
        bad_deactivation_count = terminal_metrics.count("bad_deactivation_terminal")
        prune_keep_path_count = terminal_metrics.count("prune_keep_path_terminal")
        rollout_metrics = {
            "rollout_reward_mean": terminal_metrics.mean("episode_reward") if reward_from_terminal else float(np.mean(rewards_buf)),
            "rollout_reward_source": "terminal_episode_mean" if reward_from_terminal else "per_step_reward_mean",
            "train_reliability_mean": terminal_metrics.mean("reliability_terminal"),
            "train_reliability_block_mean": terminal_metrics.mean("reliability_block_terminal"),
            "zone_drones_mean": terminal_metrics.mean("zone_drones_terminal"),
            "zone_links_mean": terminal_metrics.mean("zone_links_terminal"),
            "zone_clear_mean": terminal_metrics.mean("zone_clear_terminal"),
            "zone_abs_pen_mean": terminal_metrics.mean("zone_abs_pen_terminal"),
            "zone_abs_pen_per_step_mean": terminal_metrics.mean("zone_abs_pen_per_step_terminal"),
            "zone_viol_frac_mean": terminal_metrics.mean("zone_viol_frac_terminal"),
            "zone_drones_mean_mean": terminal_metrics.mean("zone_drones_mean_terminal"),
            "zone_links_mean_mean": terminal_metrics.mean("zone_links_mean_terminal"),
            "zone_terminal_count": float(terminal_metrics.count("zone_drones_terminal")),
            "train_reliability_block_std": terminal_metrics.std("reliability_block_terminal"),
            "train_reliability_block_p10": terminal_metrics.pctl("reliability_block_terminal", 10),
            "train_reliability_block_p90": terminal_metrics.pctl("reliability_block_terminal", 90),
            "train_latency_p90": terminal_metrics.pctl("latency_terminal_ms", 90),
            "train_delivery_ratio_mean": terminal_metrics.mean("delivery_ratio_terminal"),
            "train_reliability_hybrid_mean": terminal_metrics.mean("reliability_hybrid_terminal"),
            "train_latency_ms_mean": terminal_metrics.mean("latency_terminal_ms"),
            "train_throughput_mean": terminal_metrics.mean("throughput_terminal"),
            "active_drones_mean": terminal_metrics.mean("active_drones_terminal"),
            "qos_target_met_mean": terminal_metrics.mean("qos_target_met_terminal"),
            "path_exists_mean": terminal_metrics.mean("path_exists_terminal"),
            "bootstrap_path_success_mean": terminal_metrics.mean("bootstrap_path_success_terminal"),
            "abreu_enabled_mean": terminal_metrics.mean("abreu_enabled_terminal"),
            "num_paths_mean": terminal_metrics.mean("num_paths_terminal"),
            "success_mean": terminal_metrics.mean("success_terminal"),
            "bad_deactivation_mean": terminal_metrics.mean("bad_deactivation_terminal"),
            "prune_keep_path_mean": terminal_metrics.mean("prune_keep_path_terminal"),
            "episodes_finished": float(episodes_finished),
            "reliability_terminal_count": float(reliability_count),
            "reliability_block_terminal_count": float(reliability_block_count),
            "reliability_initial_mean": terminal_metrics.mean("reliability_initial"),
            "reliability_initial_count": float(reliability_initial_count),
            "repair_delta_mean": terminal_metrics.mean("repair_delta_terminal"),
            "delivery_ratio_terminal_count": float(delivery_ratio_count),
            "reliability_hybrid_terminal_count": float(reliability_hybrid_count),
            "latency_terminal_count": float(latency_count),
            "throughput_terminal_count": float(throughput_count),
            "active_drones_terminal_count": float(active_drones_count),
            "qos_target_met_terminal_count": float(qos_target_met_count),
            "path_exists_terminal_count": float(path_exists_count),
            "bootstrap_path_success_terminal_count": float(bootstrap_path_success_count),
            "abreu_enabled_terminal_count": float(abreu_enabled_count),
            "num_paths_terminal_count": float(num_paths_count),
            "success_terminal_count": float(success_count),
            "bad_deactivation_terminal_count": float(bad_deactivation_count),
            "prune_keep_path_terminal_count": float(prune_keep_path_count),
            "nonfinite_total": float(sum(int(v) for v in self._nonfinite_counts.values())),
        }
        self._last_rollout_metrics = rollout_metrics
        batch = RolloutBatch(
            obs=obs_buf,
            state=state_buf,
            action_mask=amask_buf,
            alive_mask=alive_buf,
            turn_mask=turn_buf,
            actions=actions_buf,
            logprobs=logprob_buf,
            rewards=rewards_buf,
            dones=dones_buf,
            truncations=truncs_buf,
            values=values_buf,
            next_state=cur["state"],
        )
        return batch, rollout_metrics, cur

    def _get_rollout_buffers(self, T: int, N: int, A: int, O: int, S: int, M: int) -> Dict[str, np.ndarray]:
        shape = (int(T), int(N), int(A), int(O), int(S), int(M))
        if self._rollout_buffer_shape != shape or not self._rollout_buffers:
            self._rollout_buffers = {
                "obs": np.zeros((T, N, A, O), dtype=np.float32),
                "state": np.zeros((T, N, S), dtype=np.float32),
                "action_mask": np.zeros((T, N, A, M), dtype=np.float32),
                "alive_mask": np.zeros((T, N, A, 1), dtype=np.float32),
                "turn_mask": np.zeros((T, N, A, 1), dtype=np.float32),
                "actions": np.zeros((T, N, A), dtype=np.int64),
                "logprobs": np.zeros((T, N, A), dtype=np.float32),
                "rewards": np.zeros((T, N), dtype=np.float32),
                "dones": np.zeros((T, N), dtype=np.float32),
                "truncations": np.zeros((T, N), dtype=np.float32),
                "values": np.zeros((T, N), dtype=np.float32),
            }
            self._rollout_buffer_shape = shape
        return self._rollout_buffers

    @torch.no_grad()
    def _compute_returns_advantages(self, batch: RolloutBatch) -> Tuple[np.ndarray, np.ndarray]:
        self._compiled_step_begin()
        with self._autocast_ctx():
            next_value = self._sanitize_np(
                self._critic_forward(self._norm_state_tensor(batch.next_state)).clone().detach().float().cpu().numpy().astype(np.float32),
                "critic_next_value",
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        rewards_np = self._sanitize_np(batch.rewards, "gae_rewards", nan=0.0, posinf=0.0, neginf=0.0)
        values_np = self._sanitize_np(batch.values, "gae_values", nan=0.0, posinf=0.0, neginf=0.0)
        dones_np = self._sanitize_np(batch.dones, "gae_dones", nan=1.0, posinf=1.0, neginf=1.0)
        truncs_np = self._sanitize_np(
            getattr(batch, "truncations", np.zeros_like(dones_np)),
            "gae_truncations", nan=0.0, posinf=0.0, neginf=0.0)
        # Término por SUCESSO = done E NÃO timeout. Nesse caso fazemos BOOTSTRAP
        # do valor futuro (não zeramos), pois o episódio foi cortado
        # artificialmente ao atingir o KPI — o futuro descontado existiria.
        success_term_np = np.clip(dones_np - truncs_np, 0.0, 1.0).astype(np.float32)
        T, N = rewards_np.shape
        adv = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros((N,), dtype=np.float32)
        for t in reversed(range(T)):
            # value_nonterminal controla o BOOTSTRAP do V(s'): zera só em timeout
            # real; no sucesso vale 1 (bootstrap). chain_nonterminal corta a
            # recursão GAE em QUALQUER término (não vaza entre episódios).
            value_nonterminal = np.clip(1.0 - (dones_np[t] - success_term_np[t]), 0.0, 1.0)
            chain_nonterminal = 1.0 - dones_np[t]
            if t == T - 1:
                next_values = next_value
            else:
                next_values = values_np[t + 1]
            delta = rewards_np[t] + self.args.gamma * next_values * value_nonterminal - values_np[t]
            delta = self._sanitize_np(delta, "gae_delta", nan=0.0, posinf=0.0, neginf=0.0)
            lastgaelam = delta + self.args.gamma * self.args.gae_lambda * chain_nonterminal * lastgaelam
            lastgaelam = self._sanitize_np(lastgaelam, "gae_lastgaelam", nan=0.0, posinf=0.0, neginf=0.0)
            adv[t] = lastgaelam
        returns = self._sanitize_np(adv + values_np, "gae_returns", nan=0.0, posinf=0.0, neginf=0.0)
        return returns.astype(np.float32, copy=False), adv.astype(np.float32, copy=False)

    def train_on_batch(self, batch: RolloutBatch) -> Dict[str, float]:
        returns, advantages = self._compute_returns_advantages(batch)
        T, N, A, _ = batch.obs.shape
        B = T * N
        mb_size = max(1, int(self.args.minibatch_size))

        state_np = batch.state.reshape(B, self.state_dim).astype(np.float32, copy=False)
        if self.state_rms is not None:
            state_np = self.state_rms.normalize(state_np).astype(np.float32, copy=False)
        returns_np = self._sanitize_np(returns.reshape(B).astype(np.float32, copy=False), "train_returns", nan=0.0, posinf=0.0, neginf=0.0)
        adv_np = self._sanitize_np(advantages.reshape(B).astype(np.float32, copy=False), "train_advantages", nan=0.0, posinf=0.0, neginf=0.0)
        adv_mean = float(np.mean(adv_np)) if adv_np.size else 0.0
        adv_std = float(np.std(adv_np)) if adv_np.size else 0.0
        if not np.isfinite(adv_mean):
            self._record_nonfinite("train_adv_mean", 1)
            adv_mean = 0.0
        if not np.isfinite(adv_std) or adv_std < 1e-8:
            if not np.isfinite(adv_std):
                self._record_nonfinite("train_adv_std", 1)
            adv_std = 1.0
        adv_np = (adv_np - adv_mean) / (adv_std + 1e-8)
        adv_np = self._sanitize_np(adv_np, "train_advantages_norm", nan=0.0, posinf=0.0, neginf=0.0)

        # critic update (GPU transfer per minibatch only)
        critic_losses = []
        for _ in range(int(self.args.n_epochs)):
            perm = np.random.permutation(B)
            for start in range(0, B, mb_size):
                idx_np = perm[start:start + mb_size]
                state_mb = self._to_device_tensor(state_np[idx_np], dtype=torch.float32)
                returns_mb = self._to_device_tensor(returns_np[idx_np], dtype=torch.float32)
                self.critic_opt.zero_grad(set_to_none=True)
                with self._autocast_ctx():
                    pred = self._critic_forward(state_mb)
                    critic_loss = F.mse_loss(pred.float(), returns_mb)
                self.critic_scaler.scale(critic_loss).backward()
                if self.critic_scaler.is_enabled():
                    self.critic_scaler.unscale_(self.critic_opt)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip)
                self.critic_scaler.step(self.critic_opt)
                self.critic_scaler.update()
                critic_losses.append(float(critic_loss.detach().float().cpu().item()))

        # actor updates (sequential with factor correction, keeping full batch on CPU)
        factor_np = np.ones((B,), dtype=np.float32)
        actor_loss_sum = 0.0
        entropy_sum = 0.0
        actor_updates = 0

        agent_order = list(range(A))
        if not self.args.fixed_order:
            random.shuffle(agent_order)

        obs_all_np = batch.obs.reshape(B, A, self.obs_dim).astype(np.float32, copy=False)
        if self.obs_rms is not None:
            obs_all_np = self.obs_rms.normalize(obs_all_np).astype(np.float32, copy=False)
        am_all_np = batch.action_mask.reshape(B, A, self.act_dim).astype(np.float32, copy=False)
        act_all_np = batch.actions.reshape(B, A).astype(np.int64, copy=False)
        old_logp_all_np = batch.logprobs.reshape(B, A).astype(np.float32, copy=False)
        turn_all_np = batch.turn_mask.reshape(B, A).astype(np.float32, copy=False)
        alive_all_np = batch.alive_mask.reshape(B, A).astype(np.float32, copy=False)

        for aid in agent_order:
            valid_idx_np = np.nonzero((turn_all_np[:, aid] > 0.5) & (alive_all_np[:, aid] > 0.5))[0]
            if valid_idx_np.size == 0:
                continue
            actor = self.actors[aid]
            opt = self.actor_opts[aid]
            old_logp_np = old_logp_all_np[:, aid]
            for _ in range(int(self.args.n_epochs)):
                perm_np = np.random.permutation(valid_idx_np)
                for start in range(0, perm_np.size, mb_size):
                    idx_np = perm_np[start:start + mb_size]
                    obs_mb = self._to_device_tensor(obs_all_np[idx_np, aid, :], dtype=torch.float32)
                    am_mb = self._to_device_tensor(am_all_np[idx_np, aid, :], dtype=torch.float32)
                    act_mb = self._to_device_tensor(act_all_np[idx_np, aid], dtype=torch.long)
                    old_logp_mb = self._to_device_tensor(old_logp_np[idx_np], dtype=torch.float32)
                    adv_mb = self._to_device_tensor(adv_np[idx_np], dtype=torch.float32)
                    factor_mb = self._to_device_tensor(factor_np[idx_np], dtype=torch.float32)
                    opt.zero_grad(set_to_none=True)
                    with self._autocast_ctx():
                        dist = self._actor_dist(aid, obs_mb, am_mb)
                        new_logp = dist.log_prob(act_mb)
                        ratio = torch.exp(new_logp - old_logp_mb)
                        surr1 = factor_mb * ratio * adv_mb
                        surr2 = factor_mb * torch.clamp(ratio, 1.0 - self.args.clip_start, 1.0 + self.args.clip_start) * adv_mb
                        entropy = dist.entropy().mean()
                        actor_loss = -torch.min(surr1, surr2).mean() - self.args.ent_coef * entropy
                    self.actor_scaler.scale(actor_loss).backward()
                    if self.actor_scaler.is_enabled():
                        self.actor_scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(actor.parameters(), self.args.grad_clip)
                    self.actor_scaler.step(opt)
                    self.actor_scaler.update()
                    actor_loss_sum += float(actor_loss.detach().float().cpu().item())
                    entropy_sum += float(entropy.detach().float().cpu().item())
                    actor_updates += 1
            # factor correction using minibatches to avoid full-batch GPU transfer
            with torch.no_grad():
                for start in range(0, valid_idx_np.size, mb_size):
                    idx_np = valid_idx_np[start:start + mb_size]
                    obs_mb = self._to_device_tensor(obs_all_np[idx_np, aid, :], dtype=torch.float32)
                    am_mb = self._to_device_tensor(am_all_np[idx_np, aid, :], dtype=torch.float32)
                    act_mb = self._to_device_tensor(act_all_np[idx_np, aid], dtype=torch.long)
                    old_logp_mb = old_logp_np[idx_np]
                    with self._autocast_ctx():
                        dist_mb = self._actor_dist(aid, obs_mb, am_mb)
                        new_logp_mb = dist_mb.log_prob(act_mb).float().detach().cpu().numpy()
                    factor_np[idx_np] *= np.exp(new_logp_mb - old_logp_mb)

        return {
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "actor_loss": float(actor_loss_sum / max(1, actor_updates)),
            "entropy": float(entropy_sum / max(1, actor_updates)),
            "adv_mean": float(np.mean(adv_np)) if adv_np.size else 0.0,
        }

    # ---------- evaluation / logging ----------
    @torch.no_grad()
    def evaluate(self, episodes: int) -> Dict[str, float]:
        # Fluxos de seed DISJUNTOS e IDÊNTICOS entre avaliações: o env i usa
        # base + 700k + i no 1º episódio e base + 800k + i*1000 + k nos
        # seguintes (k = episódio local). Todos os ~100 episódios ficam
        # semeados -> avaliações pareadas (mesmos cenários/sombras a cada
        # eval), reduzindo drasticamente o ruído da comparação de best.
        obs, _ = self.eval_envs.reset(seeds=[
            {"seed": self.args.seed + 700_000 + i,
             "auto_seed_base": self.args.seed + 800_000 + i * 1000,
             "auto_seed_stride": 1}
            for i in range(self.args.eval_envs)
        ])
        metric_keys = (
            "reliability_terminal",
            "reliability_block_terminal",
            "reliability_initial",
            "repair_delta_terminal",
            "delivery_ratio_terminal",
            "reliability_hybrid_terminal",
            "throughput_terminal",
            "latency_terminal_ms",
            "active_drones_terminal",
            "episode_reward",
            "qos_target_met_terminal",
            "path_exists_terminal",
            "bootstrap_path_success_terminal",
            "abreu_enabled_terminal",
            "num_paths_terminal",
            "success_terminal",
            "bad_deactivation_terminal",
            "prune_keep_path_terminal",
            # ZONAS PROIBIDAS (violações no estado terminal)
            "zone_drones_terminal",
            "zone_links_terminal",
            "zone_clear_terminal",
        )
        results = _MetricAccumulator(metric_keys)
        env_returns = np.zeros((self.args.eval_envs,), dtype=np.float32)
        finished = 0
        while finished < int(episodes):
            acts, _, _ = self._select_actions(obs, deterministic=True)
            obs, rewards, terms, truncs, infos = self.eval_envs.step(acts)
            done = np.logical_or(terms, truncs)
            env_returns += rewards
            for i, info in enumerate(infos):
                if bool(done[i]):
                    finished += 1
                    results.add("episode_reward", float(env_returns[i]))
                    env_returns[i] = 0.0
                    path_ok_terminal = bool(_safe_int(info.get("path_exists_terminal", 0), 0))
                    abreu_enabled_terminal = bool(_safe_int(info.get("abreu_enabled_terminal", 0), 0))
                    bootstrap_path_success = bool(_safe_int(info.get("bootstrap_path_success_terminal", 0), 0))
                    if path_ok_terminal and abreu_enabled_terminal:
                        results.add_if_present(info, "reliability_terminal")
                        results.add_if_present(info, "reliability_block_terminal")
                        results.add_if_present(info, "reliability_initial")
                        results.add_if_present(info, "repair_delta_terminal")
                        results.add_if_present(info, "delivery_ratio_terminal")
                        results.add_if_present(info, "reliability_hybrid_terminal")
                        results.add_if_present(info, "throughput_terminal")
                        results.add_if_present(info, "latency_terminal_ms")
                    else:
                        results.add("reliability_terminal", 0.0)
                        results.add("reliability_block_terminal", 0.0)
                        results.add("delivery_ratio_terminal", 0.0)
                        results.add("reliability_hybrid_terminal", 0.0)
                    results.add_if_present(info, "active_drones_terminal")
                    results.add_if_present(info, "qos_target_met_terminal")
                    results.add_if_present(info, "path_exists_terminal")
                    results.add("bootstrap_path_success_terminal", 1.0 if bootstrap_path_success else 0.0)
                    results.add("abreu_enabled_terminal", 1.0 if abreu_enabled_terminal else 0.0)
                    results.add_if_present(info, "num_paths_terminal")
                    results.add_if_present(info, "success_terminal")
                    results.add_if_present(info, "zone_drones_terminal")
                    results.add_if_present(info, "zone_links_terminal")
                    results.add_if_present(info, "zone_clear_terminal")
                    results.add_if_present(info, "zone_abs_pen_terminal")
                    results.add_if_present(info, "zone_abs_pen_per_step_terminal")
                    results.add_if_present(info, "zone_viol_frac_terminal")
                    results.add_if_present(info, "zone_drones_mean_terminal")
                    results.add_if_present(info, "zone_links_mean_terminal")
                    results.add_if_present(info, "bad_deactivation_terminal")
                    results.add_if_present(info, "prune_keep_path_terminal")
                    if finished >= int(episodes):
                        break
        return {
            "eval_reward_mean": results.mean("episode_reward"),
            "eval_reliability_mean": results.mean("reliability_terminal"),
            "eval_reliability_block_mean": results.mean("reliability_block_terminal"),
            "eval_zone_drones_mean": results.mean("zone_drones_terminal"),
            "eval_zone_links_mean": results.mean("zone_links_terminal"),
            "eval_zone_clear_mean": results.mean("zone_clear_terminal"),
            "eval_zone_abs_pen_mean": results.mean("zone_abs_pen_terminal"),
            "eval_zone_abs_pen_per_step_mean": results.mean("zone_abs_pen_per_step_terminal"),
            "eval_zone_viol_frac_mean": results.mean("zone_viol_frac_terminal"),
            "eval_zone_drones_mean_mean": results.mean("zone_drones_mean_terminal"),
            "eval_zone_links_mean_mean": results.mean("zone_links_mean_terminal"),
            "eval_zone_count": float(results.count("zone_drones_terminal")),
            "eval_reliability_block_std": results.std("reliability_block_terminal"),
            "eval_reliability_block_p10": results.pctl("reliability_block_terminal", 10),
            "eval_reliability_block_p90": results.pctl("reliability_block_terminal", 90),
            "eval_latency_p90": results.pctl("latency_terminal_ms", 90),
            # valores brutos (1 por episódio) p/ histograma no TensorBoard;
            # consumidos e removidos em _write_scalars (não vão para o JSON).
            "_eval_rel_block_values": results.values_of("reliability_block_terminal"),
            "eval_reliability_initial_mean": results.mean("reliability_initial"),
            "eval_repair_delta_mean": results.mean("repair_delta_terminal"),
            "eval_delivery_ratio_mean": results.mean("delivery_ratio_terminal"),
            "eval_reliability_hybrid_mean": results.mean("reliability_hybrid_terminal"),
            "eval_latency_ms_mean": results.mean("latency_terminal_ms"),
            "eval_throughput_mean": results.mean("throughput_terminal"),
            "eval_active_drones_mean": results.mean("active_drones_terminal"),
            "eval_qos_target_met_mean": results.mean("qos_target_met_terminal"),
            "eval_path_exists_mean": results.mean("path_exists_terminal"),
            "eval_bootstrap_path_success_mean": results.mean("bootstrap_path_success_terminal"),
            "eval_abreu_enabled_mean": results.mean("abreu_enabled_terminal"),
            "eval_num_paths_mean": results.mean("num_paths_terminal"),
            "eval_success_mean": results.mean("success_terminal"),
            "eval_bad_deactivation_mean": results.mean("bad_deactivation_terminal"),
            "eval_prune_keep_path_mean": results.mean("prune_keep_path_terminal"),
            "eval_episodes_finished": float(results.count("episode_reward")),
            "eval_reliability_count": float(results.count("reliability_terminal")),
            "eval_reliability_block_count": float(results.count("reliability_block_terminal")),
            "eval_delivery_ratio_count": float(results.count("delivery_ratio_terminal")),
            "eval_reliability_hybrid_count": float(results.count("reliability_hybrid_terminal")),
            "eval_latency_count": float(results.count("latency_terminal_ms")),
            "eval_throughput_count": float(results.count("throughput_terminal")),
            "eval_path_exists_count": float(results.count("path_exists_terminal")),
            "eval_bootstrap_path_success_count": float(results.count("bootstrap_path_success_terminal")),
            "eval_abreu_enabled_count": float(results.count("abreu_enabled_terminal")),
            "eval_num_paths_count": float(results.count("num_paths_terminal")),
            "eval_success_count": float(results.count("success_terminal")),
            "eval_bad_deactivation_count": float(results.count("bad_deactivation_terminal")),
            "eval_prune_keep_path_count": float(results.count("prune_keep_path_terminal")),
        }

    def _write_scalars(self, rollout: Dict[str, float], train_info: Dict[str, float], eval_info: Dict[str, float] | None):
        step = int(self.global_step)

        def _add_scalar_if_finite(tag: str, value: float, cond: bool = True):
            if not cond:
                return
            try:
                val = float(value)
            except Exception:
                return
            if np.isfinite(val):
                self.writer.add_scalar(tag, val, step)

        reliability_count = float(rollout.get("reliability_terminal_count", 0.0))
        reliability_block_count = float(rollout.get("reliability_block_terminal_count", 0.0))
        delivery_ratio_count = float(rollout.get("delivery_ratio_terminal_count", 0.0))
        reliability_hybrid_count = float(rollout.get("reliability_hybrid_terminal_count", 0.0))
        latency_count = float(rollout.get("latency_terminal_count", 0.0))
        throughput_count = float(rollout.get("throughput_terminal_count", 0.0))
        active_count = float(rollout.get("active_drones_terminal_count", 0.0))
        path_exists_count = float(rollout.get("path_exists_terminal_count", 0.0))
        bootstrap_path_success_count = float(rollout.get("bootstrap_path_success_terminal_count", 0.0))
        abreu_enabled_count = float(rollout.get("abreu_enabled_terminal_count", 0.0))
        num_paths_count = float(rollout.get("num_paths_terminal_count", 0.0))

        self.writer.add_scalar("rollout/ep_rew_mean", rollout.get("rollout_reward_mean", 0.0), step)
        self.writer.add_scalar("main/rollout_reward_mean", rollout.get("rollout_reward_mean", 0.0), step)
        # REMOVIDO: main/train_reliability_mean era um ALIAS exato de
        # main/reliability_terminal_mean (mesma fonte: média dos TERMINAIS dos
        # episódios do update) — duas curvas idênticas com nomes que sugeriam
        # semânticas diferentes. Mantém-se apenas o nome correto (terminal).
        _add_scalar_if_finite("main/train_reliability_block_mean", rollout.get("train_reliability_block_mean", 0.0), reliability_block_count > 0)
        _add_scalar_if_finite("main/train_delivery_ratio_mean", rollout.get("train_delivery_ratio_mean", 0.0), delivery_ratio_count > 0)
        _add_scalar_if_finite("main/train_reliability_hybrid_mean", rollout.get("train_reliability_hybrid_mean", 0.0), reliability_hybrid_count > 0)
        _add_scalar_if_finite("main/active_drones_mean", rollout.get("active_drones_mean", 0.0), active_count > 0)
        _add_scalar_if_finite("main/train_latency_ms_mean", rollout.get("train_latency_ms_mean", 0.0), latency_count > 0)
        _add_scalar_if_finite("main/train_throughput_mean", rollout.get("train_throughput_mean", 0.0), throughput_count > 0)
        _add_scalar_if_finite("main/reliability_terminal_mean", rollout.get("train_reliability_mean", 0.0), reliability_count > 0)
        _add_scalar_if_finite("main/reliability_initial_mean", rollout.get("reliability_initial_mean", 0.0), float(rollout.get("reliability_initial_count", 0.0)) > 0)
        _add_scalar_if_finite("main/repair_delta_mean", rollout.get("repair_delta_mean", 0.0), float(rollout.get("reliability_initial_count", 0.0)) > 0)
        _add_scalar_if_finite("main/latency_terminal_ms_mean", rollout.get("train_latency_ms_mean", 0.0), latency_count > 0)
        _add_scalar_if_finite("main/throughput_terminal_mean", rollout.get("train_throughput_mean", 0.0), throughput_count > 0)
        _add_scalar_if_finite("main/qos_target_met_mean", rollout.get("qos_target_met_mean", 0.0), float(rollout.get("qos_target_met_terminal_count", 0.0)) > 0)
        _add_scalar_if_finite("main/path_exists_terminal", rollout.get("path_exists_mean", 0.0), path_exists_count > 0)
        _add_scalar_if_finite("main/path_exists_terminal_mean", rollout.get("path_exists_mean", 0.0), path_exists_count > 0)
        _add_scalar_if_finite("main/bootstrap_path_success_mean", rollout.get("bootstrap_path_success_mean", 0.0), bootstrap_path_success_count > 0)
        _add_scalar_if_finite("main/abreu_enabled_terminal_mean", rollout.get("abreu_enabled_mean", 0.0), abreu_enabled_count > 0)
        _add_scalar_if_finite("main/num_paths_terminal_mean", rollout.get("num_paths_mean", 0.0), num_paths_count > 0)
        _add_scalar_if_finite("main/numero_de_caminhos", rollout.get("num_paths_mean", 0.0), num_paths_count > 0)
        _add_scalar_if_finite("bootstrap/path_exists_mean", rollout.get("path_exists_mean", 0.0), path_exists_count > 0)
        _add_scalar_if_finite("bootstrap/path_success_mean", rollout.get("bootstrap_path_success_mean", 0.0), bootstrap_path_success_count > 0)
        _add_scalar_if_finite("main/success_mean", rollout.get("success_mean", 0.0), float(rollout.get("success_terminal_count", 0.0)) > 0)
        _add_scalar_if_finite("main/bad_deactivation_mean", rollout.get("bad_deactivation_mean", 0.0), float(rollout.get("bad_deactivation_terminal_count", 0.0)) > 0)
        _add_scalar_if_finite("main/prune_keep_path_mean", rollout.get("prune_keep_path_mean", 0.0), float(rollout.get("prune_keep_path_terminal_count", 0.0)) > 0)

        # Aliases claros na aba main para acompanhar o treino.
        _add_scalar_if_finite("main/recompensa", rollout.get("rollout_reward_mean", 0.0), True)
        _add_scalar_if_finite("main/confiabilidade", rollout.get("train_reliability_block_mean", 0.0), reliability_block_count > 0)
        # DISPERSÃO entre episódios (a média é cega para a bimodalidade):
        _add_scalar_if_finite("main/confiabilidade_std", rollout.get("train_reliability_block_std", 0.0), reliability_block_count > 1)
        _add_scalar_if_finite("main/confiabilidade_p10", rollout.get("train_reliability_block_p10", 0.0), reliability_block_count > 0)
        _add_scalar_if_finite("main/confiabilidade_p90", rollout.get("train_reliability_block_p90", 0.0), reliability_block_count > 0)
        _add_scalar_if_finite("main/latency_terminal_p90", rollout.get("train_latency_p90", 0.0), reliability_block_count > 0)
        _add_scalar_if_finite("main/taxa_entrega", rollout.get("train_delivery_ratio_mean", 0.0), delivery_ratio_count > 0)
        _add_scalar_if_finite("main/confiabilidade_hibrida", rollout.get("train_reliability_hybrid_mean", 0.0), reliability_hybrid_count > 0)
        _add_scalar_if_finite("main/vazao_maxima", rollout.get("train_throughput_mean", 0.0), throughput_count > 0)
        _add_scalar_if_finite("main/latencia_maxima", rollout.get("train_latency_ms_mean", 0.0), latency_count > 0)
        _add_scalar_if_finite("main/numero_de_drones", rollout.get("active_drones_mean", 0.0), active_count > 0)
        _add_scalar_if_finite("main/sucesso", rollout.get("success_mean", 0.0), float(rollout.get("success_terminal_count", 0.0)) > 0)
        # ---- ZONAS PROIBIDAS (violações no estado terminal) ----
        _zc = float(rollout.get("zone_terminal_count", 0.0)) > 0
        _add_scalar_if_finite("main/zona_drones_dentro", rollout.get("zone_drones_mean", 0.0), _zc)
        _add_scalar_if_finite("main/zona_links_cruzando", rollout.get("zone_links_mean", 0.0), _zc)
        _add_scalar_if_finite("main/zona_evacuada", rollout.get("zone_clear_mean", 0.0), _zc)
        _add_scalar_if_finite("main/qos_atingido", rollout.get("qos_target_met_mean", 0.0), float(rollout.get("qos_target_met_terminal_count", 0.0)) > 0)
        _add_scalar_if_finite("main/caminho_terminal", rollout.get("path_exists_mean", 0.0), path_exists_count > 0)
        _add_scalar_if_finite("main/sucesso_bootstrap_caminho", rollout.get("bootstrap_path_success_mean", 0.0), bootstrap_path_success_count > 0)

        self.writer.add_scalar("main/episodes_finished_per_update", rollout.get("episodes_finished", 0.0), step)
        self.writer.add_scalar("main/episodios_finalizados", rollout.get("episodes_finished", 0.0), step)
        self.writer.add_scalar("main/reliability_terminal_count", rollout.get("reliability_terminal_count", 0.0), step)
        self.writer.add_scalar("main/reliability_block_terminal_count", rollout.get("reliability_block_terminal_count", 0.0), step)
        self.writer.add_scalar("main/delivery_ratio_terminal_count", rollout.get("delivery_ratio_terminal_count", 0.0), step)
        self.writer.add_scalar("main/reliability_hybrid_terminal_count", rollout.get("reliability_hybrid_terminal_count", 0.0), step)
        self.writer.add_scalar("main/latency_terminal_count", rollout.get("latency_terminal_count", 0.0), step)
        self.writer.add_scalar("main/throughput_terminal_count", rollout.get("throughput_terminal_count", 0.0), step)
        self.writer.add_scalar("main/active_drones_terminal_count", rollout.get("active_drones_terminal_count", 0.0), step)
        self.writer.add_scalar("main/qos_target_met_terminal_count", rollout.get("qos_target_met_terminal_count", 0.0), step)
        self.writer.add_scalar("main/path_exists_terminal_count", rollout.get("path_exists_terminal_count", 0.0), step)
        self.writer.add_scalar("main/bootstrap_path_success_terminal_count", rollout.get("bootstrap_path_success_terminal_count", 0.0), step)
        self.writer.add_scalar("main/abreu_enabled_terminal_count", rollout.get("abreu_enabled_terminal_count", 0.0), step)
        self.writer.add_scalar("main/num_paths_terminal_count", rollout.get("num_paths_terminal_count", 0.0), step)
        self.writer.add_scalar("bootstrap/path_exists_terminal_count", rollout.get("path_exists_terminal_count", 0.0), step)
        self.writer.add_scalar("bootstrap/path_success_terminal_count", rollout.get("bootstrap_path_success_terminal_count", 0.0), step)
        self.writer.add_scalar("main/success_terminal_count", rollout.get("success_terminal_count", 0.0), step)
        self.writer.add_scalar("main/bad_deactivation_terminal_count", rollout.get("bad_deactivation_terminal_count", 0.0), step)
        self.writer.add_scalar("main/prune_keep_path_terminal_count", rollout.get("prune_keep_path_terminal_count", 0.0), step)
        self.writer.add_scalar("main/nonfinite_total", rollout.get("nonfinite_total", 0.0), step)
        self.writer.add_scalar("main/nao_finitos", rollout.get("nonfinite_total", 0.0), step)
        for nf_name, nf_count in sorted(getattr(self, "_nonfinite_counts", {}).items()):
            safe_name = str(nf_name).replace("/", "_")
            self.writer.add_scalar(f"debug_nonfinite/{safe_name}", int(nf_count), step)
        self.writer.add_scalar("train/actor_loss", train_info.get("actor_loss", 0.0), step)
        self.writer.add_scalar("train/critic_loss", train_info.get("critic_loss", 0.0), step)
        self.writer.add_scalar("main/perda_ator", train_info.get("actor_loss", 0.0), step)
        self.writer.add_scalar("main/perda_critico", train_info.get("critic_loss", 0.0), step)
        self.writer.add_scalar("train/entropy", train_info.get("entropy", 0.0), step)
        if eval_info is not None:
            _add_scalar_if_finite("eval_terminal/reliability_terminal_mean", eval_info.get("eval_reliability_mean", 0.0), float(eval_info.get("eval_reliability_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/reliability_block_terminal_mean", eval_info.get("eval_reliability_block_mean", 0.0), float(eval_info.get("eval_reliability_block_count", 0.0)) > 0)
            _ezc = float(eval_info.get("eval_zone_count", 0.0)) > 0
            _add_scalar_if_finite("eval_terminal/zona_drones_dentro", eval_info.get("eval_zone_drones_mean", 0.0), _ezc)
            _add_scalar_if_finite("eval_terminal/zona_links_cruzando", eval_info.get("eval_zone_links_mean", 0.0), _ezc)
            _add_scalar_if_finite("eval_terminal/zona_evacuada", eval_info.get("eval_zone_clear_mean", 0.0), _ezc)
            _has_rel = float(eval_info.get("eval_reliability_block_count", 0.0)) > 0
            _add_scalar_if_finite("eval_terminal/reliability_block_std", eval_info.get("eval_reliability_block_std", 0.0), _has_rel)
            _add_scalar_if_finite("eval_terminal/reliability_block_p10", eval_info.get("eval_reliability_block_p10", 0.0), _has_rel)
            _add_scalar_if_finite("eval_terminal/reliability_block_p90", eval_info.get("eval_reliability_block_p90", 0.0), _has_rel)
            _add_scalar_if_finite("eval_terminal/latency_p90", eval_info.get("eval_latency_p90", 0.0), _has_rel)
            # HISTOGRAMA da distribuição terminal (aba Distributions/Histograms):
            # é a evidência visual da bimodalidade e da sua dissolução — os
            # scalars de média/percentil não mostram os dois modos diretamente.
            _vals = eval_info.pop("_eval_rel_block_values", None)
            if _vals:
                try:
                    self.writer.add_histogram("eval_terminal/reliability_block_hist", np.asarray(_vals, dtype=np.float32), step)
                except Exception:
                    pass
            _add_scalar_if_finite("eval_terminal/delivery_ratio_terminal_mean", eval_info.get("eval_delivery_ratio_mean", 0.0), float(eval_info.get("eval_delivery_ratio_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/reliability_hybrid_terminal_mean", eval_info.get("eval_reliability_hybrid_mean", 0.0), float(eval_info.get("eval_reliability_hybrid_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/latency_terminal_ms_mean", eval_info.get("eval_latency_ms_mean", 0.0), float(eval_info.get("eval_latency_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/throughput_terminal_mean", eval_info.get("eval_throughput_mean", 0.0), float(eval_info.get("eval_throughput_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/active_drones_terminal_mean", eval_info.get("eval_active_drones_mean", 0.0), float(eval_info.get("eval_episodes_finished", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/qos_target_met_terminal_mean", eval_info.get("eval_qos_target_met_mean", 0.0), float(eval_info.get("eval_episodes_finished", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/path_exists_terminal_mean", eval_info.get("eval_path_exists_mean", 0.0), float(eval_info.get("eval_episodes_finished", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/bootstrap_path_success_mean", eval_info.get("eval_bootstrap_path_success_mean", 0.0), float(eval_info.get("eval_bootstrap_path_success_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/abreu_enabled_terminal_mean", eval_info.get("eval_abreu_enabled_mean", 0.0), float(eval_info.get("eval_abreu_enabled_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/num_paths_terminal_mean", eval_info.get("eval_num_paths_mean", 0.0), float(eval_info.get("eval_num_paths_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/success_terminal_mean", eval_info.get("eval_success_mean", 0.0), float(eval_info.get("eval_episodes_finished", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/bad_deactivation_mean", eval_info.get("eval_bad_deactivation_mean", 0.0), float(eval_info.get("eval_bad_deactivation_count", 0.0)) > 0)
            _add_scalar_if_finite("eval_terminal/prune_keep_path_mean", eval_info.get("eval_prune_keep_path_mean", 0.0), float(eval_info.get("eval_prune_keep_path_count", 0.0)) > 0)
            self.writer.add_scalar("eval_terminal/episodes_finished", eval_info.get("eval_episodes_finished", 0.0), step)
            self.writer.add_scalar("eval_terminal/reliability_count", eval_info.get("eval_reliability_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/latency_count", eval_info.get("eval_latency_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/throughput_count", eval_info.get("eval_throughput_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/path_exists_count", eval_info.get("eval_path_exists_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/bootstrap_path_success_count", eval_info.get("eval_bootstrap_path_success_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/abreu_enabled_count", eval_info.get("eval_abreu_enabled_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/num_paths_count", eval_info.get("eval_num_paths_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/success_count", eval_info.get("eval_success_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/bad_deactivation_count", eval_info.get("eval_bad_deactivation_count", 0.0), step)
            self.writer.add_scalar("eval_terminal/prune_keep_path_count", eval_info.get("eval_prune_keep_path_count", 0.0), step)
        self.writer.flush()

    def _write_latest_metrics(self, rollout_metrics: Dict[str, float], train_info: Dict[str, float] | None = None, eval_info: Dict[str, float] | None = None, metric_source: Dict[str, float] | None = None, metric_source_name: str = "rollout") -> None:
        """Salva um JSON leve para o curriculum_manager decidir avanço de fase.

        O arquivo é escrito em dois locais:
        - run_dir/artifacts/latest_metrics.json, específico da execução;
        - latest_metrics_path ou results_root/latest_metrics.json, usado pelo gerente de currículo.
        """
        train_info = train_info or {}
        eval_info = eval_info or {}
        metric_source = metric_source or (eval_info if eval_info else rollout_metrics)

        def f(key: str, default: float = 0.0) -> float:
            try:
                return float(metric_source.get(key, rollout_metrics.get(key, eval_info.get(key, default))))
            except Exception:
                return float(default)

        # Preferir avaliação quando ela existir; caso contrário, usar rollout.
        payload = {
            "global_step": int(self.global_step),
            "update_idx": int(self.update_idx),
            "episodes_finished": int(self.episodes_finished),
            "metric_source": str(metric_source_name),
            "curriculum_phase_name": str(getattr(self.args, "curriculum_phase_name", "")),
            "curriculum_phase_idx": int(getattr(self.args, "curriculum_phase_idx", -1)),
            "curriculum_block_idx": int(getattr(self.args, "curriculum_block_idx", -1)),
            "reliability_mean": f("eval_reliability_mean", f("train_reliability_mean", 0.0)),
            "reliability_block_mean": f("eval_reliability_block_mean", f("train_reliability_block_mean", 0.0)),
            "delivery_ratio_mean": f("eval_delivery_ratio_mean", f("train_delivery_ratio_mean", 0.0)),
            "reliability_hybrid_mean": f("eval_reliability_mean", f("train_reliability_mean", 0.0)),
            "throughput_mean": f("eval_throughput_mean", f("train_throughput_mean", 0.0)),
            "latency_ms_mean": f("eval_latency_ms_mean", f("train_latency_ms_mean", 0.0)),
            "active_drones_mean": f("eval_active_drones_mean", f("active_drones_mean", 0.0)),
            "qos_target_met_mean": f("eval_qos_target_met_mean", f("qos_target_met_mean", 0.0)),
            "path_exists_mean": f("eval_path_exists_mean", f("path_exists_mean", 0.0)),
            "num_paths_mean": f("eval_num_paths_mean", f("num_paths_mean", 0.0)),
            "bootstrap_path_success_mean": f("eval_bootstrap_path_success_mean", f("bootstrap_path_success_mean", 0.0)),
            "success_mean": f("eval_success_mean", f("success_mean", 0.0)),
            "bad_deactivation_mean": f("eval_bad_deactivation_mean", f("bad_deactivation_mean", 0.0)),
            # ZONAS: lidas pelo gerente de currículo (start.sh) para decidir avanço
            # das fases de zona. Sem estas chaves o critério leria 0 e travaria.
            "zone_clear_mean": f("eval_zone_clear_mean", f("zone_clear_mean", 0.0)),
            "zone_drones_mean": f("eval_zone_drones_mean", f("zone_drones_mean", 0.0)),
            "zone_links_mean": f("eval_zone_links_mean", f("zone_links_mean", 0.0)),
            "prune_keep_path_mean": f("eval_prune_keep_path_mean", f("prune_keep_path_mean", 0.0)),
            "rollout_reward_mean": float(rollout_metrics.get("rollout_reward_mean", 0.0)),
            "critic_loss": float(train_info.get("critic_loss", 0.0)),
            "actor_loss": float(train_info.get("actor_loss", 0.0)),
            "entropy": float(train_info.get("entropy", 0.0)),
            "nonfinite_total": float(rollout_metrics.get("nonfinite_total", 0.0)),
            "targets": {
                "reliability_target": float(getattr(self.args, "reliability_target", 0.0)),
                "delivery_ratio_target": float(getattr(self.args, "delivery_ratio_target", 0.0)),
                "prune_delivery_min": float(getattr(self.args, "prune_delivery_min", 0.0)),
                "prune_rel_min": float(getattr(self.args, "prune_rel_min", 0.0)),
                "carrier_sense_range_m": float(getattr(self.args, "carrier_sense_range_m", 0.0)),
                "traffic_rate_mbps": float(getattr(self.args, "traffic_rate_mbps", 0.0)),
                "max_delay_ms": float(getattr(self.args, "max_delay_ms", 0.0)),
            },
            "rollout": dict(rollout_metrics),
            "eval": dict(eval_info),
        }
        paths = [self.artifacts_dir / "latest_metrics.json"]
        # O arquivo GLOBAL (lido pelo gerente de currículo em start.sh para
        # decidir avanço de fase/estagnação) só é atualizado quando a fonte é o
        # EVAL (determinístico, 100 episódios). Antes ele era sobrescrito a cada
        # update — em 9 de 10 iterações com métricas de ROLLOUT (política
        # estocástica, poucos episódios) — e o start.sh não confere
        # `metric_source`, então a decisão do bloco podia se basear num snapshot
        # ruidoso dependendo do momento da leitura. O arquivo local (artifacts)
        # continua sendo escrito em toda iteração para depuração.
        if metric_source_name == "eval":
            configured = str(getattr(self.args, "latest_metrics_path", "") or "").strip()
            if configured:
                paths.append(Path(configured).expanduser().resolve())
            else:
                paths.append(Path(self.args.results_root).expanduser().resolve() / "latest_metrics.json")
        for path in paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception as exc:
                print(f"[WARN] Falha ao salvar latest_metrics em {path}: {exc}", flush=True)

    # ---------- checkpoint / best ----------
    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "update_idx": int(self.update_idx),
            "episodes_finished": int(self.episodes_finished),
            "best_metric": float(self.best_metric),
            "best_qos": float(self.best_qos),
            "best_zone_clear": float(self.best_zone_clear),
            "best_reward": float(self.best_reward),
            "best_step": int(self.best_step),
            "best_phase_id": str(getattr(self, "best_phase_id", "")),
            "args": self._serializable_args(self.args),
            "env_cfg": self.env_cfg,
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "actor_opts": [o.state_dict() for o in self.actor_opts],
            "critic_opt": self.critic_opt.state_dict(),
            "actor_scaler": self.actor_scaler.state_dict() if self.actor_scaler.is_enabled() else None,
            "critic_scaler": self.critic_scaler.state_dict() if self.critic_scaler.is_enabled() else None,
            "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
            "state_rms": None if self.state_rms is None else self.state_rms.state_dict(),
            "last_rollout_metrics": self._last_rollout_metrics,
        }

    def _save_checkpoint(self, path: Path) -> None:
        payload = self._checkpoint_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

        # Mantém somente o checkpoint recém-criado nesta pasta. O alias
        # checkpoint_latest.pt e o espelho ppo_best_model continuam existindo.
        deleted_old = []
        for old in sorted(self.ckpt_dir.glob("ckpt_step_*.pt")):
            try:
                if old.resolve() != path.resolve():
                    old.unlink()
                    deleted_old.append(str(old))
            except Exception:
                pass

        latest = self.ckpt_dir / "checkpoint_latest.pt"
        shutil.copy2(path, latest)
        meta = {
            "saved_path": str(path),
            "latest_path": str(latest),
            "deleted_old_checkpoints": deleted_old,
            "global_step": int(self.global_step),
            "update_idx": int(self.update_idx),
        }
        (self.artifacts_dir / "checkpoint_resolution.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def _load_checkpoint(self, path: Path, weights_only_seed: bool = False) -> None:
        """weights_only_seed=True: carrega APENAS pesos (ator/crítico) e os
        normalizadores; NÃO herda otimizadores, global_step nem best_*.

        Uso: FINE-TUNING a partir do best de OUTRO run (--restore-model-path).
        Herdar o estado do Adam de outra distribuição de dados provoca o mesmo
        descompasso que já derrubou um treino nesta base; e herdar global_step
        faria o run novo começar contando de 32M."""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        for actor, state in zip(self.actors, ckpt["actors"]):
            actor.load_state_dict(state)
        self.critic.load_state_dict(ckpt["critic"])
        for opt, state in zip(self.actor_opts, ckpt.get("actor_opts", [])):
            opt.load_state_dict(state)
        if "critic_opt" in ckpt:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if self.actor_scaler.is_enabled() and ckpt.get("actor_scaler") is not None:
            self.actor_scaler.load_state_dict(ckpt["actor_scaler"])
        if self.critic_scaler.is_enabled() and ckpt.get("critic_scaler") is not None:
            self.critic_scaler.load_state_dict(ckpt["critic_scaler"])
        if self.obs_rms is not None and ckpt.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if self.state_rms is not None and ckpt.get("state_rms") is not None:
            self.state_rms.load_state_dict(ckpt["state_rms"])
        if weights_only_seed:
            print(f"[SEED] Pesos carregados de {path} (otimizador/step/best NÃO herdados).", flush=True)
            return
        self.global_step = int(ckpt.get("global_step", 0))
        self.update_idx = int(ckpt.get("update_idx", 0))
        self.episodes_finished = int(ckpt.get("episodes_finished", 0))
        self.best_metric = float(ckpt.get("best_metric", self.best_metric))
        self.best_qos = float(ckpt.get("best_qos", self.best_qos))
        self.best_zone_clear = float(ckpt.get("best_zone_clear", self.best_zone_clear))
        self.best_reward = float(ckpt.get("best_reward", self.best_reward))
        self.best_step = int(ckpt.get("best_step", self.best_step))
        # Identidade da fase em que o best vigente foi medido. Se a fase MUDOU,
        # o baseline lexicográfico é rezerado (ver _is_better_lex): zone_clear e
        # qos medidos sob misturas diferentes (p_sem_zona 0.7 -> 0.3 -> 0.1,
        # zona 8x8 -> 12x12 -> 15x15) não são comparáveis. Sem isto, um salto de
        # zone_clear causado pela TROCA DE FASE — não por mérito da política —
        # satisfaz o nível 1 e sobrescreve um best de qos muito melhor.
        self.best_phase_id = str(ckpt.get("best_phase_id", ""))
        cur_phase = self._phase_identity(self.args)
        if cur_phase and self.best_phase_id and cur_phase != self.best_phase_id:
            print(f"[BEST] fase mudou ('{self.best_phase_id}' -> '{cur_phase}'): "
                  f"baseline lexicográfico rezerado (qos={self.best_qos:.3f} "
                  f"zone_clear={self.best_zone_clear:.3f} descartados como referência).", flush=True)
            self.best_qos = -float("inf")
            self.best_reward = -float("inf")
            self.best_zone_clear = -float("inf")
            self.best_metric = -float("inf") if self.args.best_mode == "max" else float("inf")
            self._lex_pass_streak = 0
        if cur_phase:
            self.best_phase_id = cur_phase

    def _is_better(self, metric: float) -> bool:
        if self.args.best_mode == "max":
            return metric > (self.best_metric + self.args.best_min_delta)
        return metric < (self.best_metric - self.args.best_min_delta)

    def _is_better_lex(self, qos: float, reward: float, zone_clear: float = None) -> bool:
        """Critério lexicográfico de 3 níveis: zone_clear > qos > recompensa.

        Prioridade (opção (a) do currículo de zonas):
          1º ZONE_CLEAR: respeitar as zonas é inegociável. Um modelo que evacua
             mais (zone_clear maior) vence, MESMO com qos menor.
          2º QOS: entre modelos de zone_clear equivalente, o de maior qos.
          3º RECOMPENSA: desempate final.

        Sem zonas no cenário (zone_clear=None ou best nunca teve zona), cai no
        critério antigo (qos > recompensa) — retrocompatível.
        NUNCA salva se um nível superior PIOROU além da tolerância.
        """
        tol = float(self.best_lex_tol)
        ztol = float(getattr(self.args, "best_zone_tol", 0.03))

        def _decide(result: bool, level: str, why: str) -> bool:
            # Registra QUAL nível decidiu, para o log do [BEST]. Sem isto, uma
            # decisão surpreendente (ex.: best com qos MENOR que o anterior) era
            # indistinguível de bug: os três níveis produzem a mesma linha.
            self._lex_last_level = level
            self._lex_last_why = why
            return result

        # Nível 1: zone_clear (só quando aplicável — há zonas medidas)
        zc_active = zone_clear is not None and (
            self.best_zone_clear > -1e17 or zone_clear > 0.0
        )
        if zc_active:
            bzc = self.best_zone_clear if self.best_zone_clear > -1e17 else 0.0
            if zone_clear > bzc + ztol:
                # evacua MAIS -> salva (mesmo com qos menor)
                return _decide(True, "L1:zone_clear", f"zc {zone_clear:.3f} > {bzc:.3f}+{ztol:.3f}")
            if zone_clear < bzc - ztol:
                # evacua MENOS -> nunca salva (viola prioridade)
                return _decide(False, "L1:zone_clear", f"zc {zone_clear:.3f} < {bzc:.3f}-{ztol:.3f}")
            # zone_clear equivalente -> desce para o critério de qos

        # Nível 2: qos
        if qos > self.best_qos + tol:
            return _decide(True, "L2:qos", f"qos {qos:.3f} > {self.best_qos:.3f}+{tol:.3f}")
        if qos < self.best_qos - tol:
            # qos piorou (com zone_clear equivalente) -> não salva
            return _decide(False, "L2:qos", f"qos {qos:.3f} < {self.best_qos:.3f}-{tol:.3f}")
        # Nível 3: qos equivalente -> desempate por recompensa
        _d = float(self.args.best_min_delta)
        return _decide(reward > self.best_reward + _d, "L3:reward",
                       f"rew {reward:.1f} vs {self.best_reward:.1f}+{_d:.1f}")

    def _mirror_best(self, source_ckpt: Path, metric_value: float) -> None:
        self.best_dir.mkdir(parents=True, exist_ok=True)
        dst = self.best_dir / "best_model.pt"
        shutil.copy2(source_ckpt, dst)
        (self.best_dir / "seed.txt").write_text(f"{self.args.seed}\n")
        norm_manifest = {
            "obs_norm_enabled": bool(self.args.obs_norm),
            "obs_rms_present": self.obs_rms is not None,
            "state_rms_present": self.state_rms is not None,
        }
        (self.best_dir / "normalization_manifest.json").write_text(json.dumps(norm_manifest, indent=2, ensure_ascii=False))
        best_info = {
            "checkpoint_path": str(dst),
            "source_checkpoint": str(source_ckpt),
            "metric_name": self.args.best_metric,
            "metric_value": float(metric_value),
            "global_step": int(self.global_step),
            "update_idx": int(self.update_idx),
            "seed": int(self.args.seed),
        }
        for name in ["_BEST_INFO.json", "test_bundle.json"]:
            (self.best_dir / name).write_text(json.dumps(best_info, indent=2, ensure_ascii=False))
        torch.save(best_info, self.best_dir / "test_bundle.pkl")

    # ---------- main loop ----------
    def _resolve_best_metric(self, metric_source: Dict[str, float], source_name: str) -> Tuple[str, Optional[float]]:
        # Retorna (nome_da_metrica, valor) APENAS se a métrica configurada
        # estiver presente nesta fonte. Caso contrário retorna (requested, None),
        # sinalizando que o best NÃO deve ser avaliado nesta iteração — evita o
        # bug de comparar métricas de escalas diferentes (ex.: recompensa ~150 vs
        # confiabilidade ~0,9) quando a métrica pedida só existe em uma das
        # fontes (rollout vs eval).
        requested = str(self.args.best_metric)
        if requested in metric_source:
            return requested, float(metric_source.get(requested, 0.0))
        return requested, None

    @staticmethod
    def _fmt_metric_value(value: float, count: float, suffix: str = "") -> str:
        if int(count) <= 0:
            return f"n/a{suffix}"
        if suffix == "ms":
            return f"{float(value):.3f}{suffix}"
        if suffix == "Mbps":
            return f"{float(value):.3f}{suffix}"
        return f"{float(value):.6f}{suffix}"

    def run(self):
        obs, _ = self.envs.reset(seeds=[self.args.seed + i for i in range(self.args.num_workers)])
        start_global_step = int(self.global_step)
        if str(getattr(self.args, "timesteps_mode", "total")).lower() == "incremental":
            target_global_step = start_global_step + int(self.args.timesteps)
        else:
            target_global_step = int(self.args.timesteps)
        run_plan = {
            "timesteps_mode": str(getattr(self.args, "timesteps_mode", "total")),
            "start_global_step": int(start_global_step),
            "target_global_step": int(target_global_step),
            "requested_timesteps": int(self.args.timesteps),
            "curriculum_phase_name": str(getattr(self.args, "curriculum_phase_name", "")),
            "curriculum_phase_idx": int(getattr(self.args, "curriculum_phase_idx", -1)),
            "curriculum_block_idx": int(getattr(self.args, "curriculum_block_idx", -1)),
        }
        try:
            (self.artifacts_dir / "run_plan.json").write_text(json.dumps(run_plan, indent=2, ensure_ascii=False))
        except Exception:
            pass
        while self.global_step < int(target_global_step):
            self.update_idx += 1
            batch, rollout_metrics, obs = self.collect_rollout(obs)
            train_info = self.train_on_batch(batch)
            eval_info = None
            if self.update_idx % self._eval_frequency_updates == 0:
                eval_info = self.evaluate(self.args.eval_episodes)
            self._write_scalars(rollout_metrics, train_info, eval_info)
            should_save = (self.update_idx % self._save_frequency_updates == 0) or (self.global_step >= int(target_global_step))
            saved_path = None
            if should_save:
                saved_path = self.ckpt_dir / f"ckpt_step_{int(self.global_step):012d}.pt"
                self._save_checkpoint(saved_path)
            metric_source = eval_info if eval_info is not None else rollout_metrics
            metric_source_name = "eval" if eval_info is not None else "rollout"
            self._write_latest_metrics(rollout_metrics, train_info, eval_info, metric_source, metric_source_name)
            if str(self.args.best_metric) == "lex_qos_reward":
                # Critério lexicográfico: precisa do QoS (do eval) e da recompensa
                # (do rollout) juntos -> só avalia nas iterações de avaliação.
                if eval_info is not None:
                    qos_now = float(eval_info.get("eval_qos_target_met_mean", 0.0))
                    # Desempate pela recompensa do PRÓPRIO eval (mesma política,
                    # determinística, pós-update) — antes usava a recompensa do
                    # rollout (política PRÉ-update, estocástica): par desalinhado
                    # e ruidoso para o critério lexicográfico.
                    reward_now = float(eval_info.get("eval_reward_mean", rollout_metrics.get("rollout_reward_mean", 0.0)))
                    # zone_clear do eval: nível lexicográfico PRIMÁRIO (respeitar
                    # zonas > qos). None quando o cenário não tem zonas medidas.
                    zc_now = eval_info.get("eval_zone_clear_mean", None)
                    zc_now = float(zc_now) if zc_now is not None else None
                    _zcs = "n/a" if zc_now is None else f"{zc_now:.3f}"
                    _bzcs = "n/a" if self.best_zone_clear <= -1e17 else f"{self.best_zone_clear:.3f}"
                    _bqs = "n/a" if self.best_qos <= -1e17 else f"{self.best_qos:.3f}"
                    if self._is_better_lex(qos_now, reward_now, zc_now):
                        # CONFIRMAÇÃO: o best só é espelhado depois de
                        # best_confirm_evals avaliações CONSECUTIVAS passando no
                        # critério. Como qos_mean é binário sobre ~100 episódios
                        # (erro-padrão ~0.05 perto de p=0.5), uma única avaliação
                        # "sortuda" podia virar best e, com a referência no
                        # máximo histórico, bloquear políticas genuinamente
                        # melhores. A janela guarda o MÍNIMO das avaliações
                        # confirmantes (estimativa conservadora do QoS real).
                        self._lex_pass_streak += 1
                        self._lex_window_qos = min(self._lex_window_qos, qos_now)
                        self._lex_window_reward = min(self._lex_window_reward, reward_now)
                        if zc_now is not None:
                            self._lex_window_zc = min(getattr(self, "_lex_window_zc", float("inf")), zc_now)
                        need = max(1, int(getattr(self.args, "best_confirm_evals", 2)))
                        if self._lex_pass_streak >= need:
                            # ANTI-DRIFT: a referência de QoS NUNCA decai
                            # (max() contra o mínimo conservador da janela);
                            # a tolerância vale contra o máximo histórico e o
                            # drift total fica limitado a best_lex_tol.
                            self.best_qos = max(self.best_qos, float(self._lex_window_qos))
                            self.best_reward = float(self._lex_window_reward)
                            _wzc = getattr(self, "_lex_window_zc", float("inf"))
                            if _wzc < float("inf"):
                                self.best_zone_clear = max(self.best_zone_clear, float(_wzc))
                            self.best_metric = qos_now  # registra o QoS como métrica principal
                            self.best_step = int(self.global_step)
                            if saved_path is None:
                                saved_path = self.ckpt_dir / f"ckpt_step_{int(self.global_step):012d}.pt"
                                self._save_checkpoint(saved_path)
                            self._mirror_best(saved_path, qos_now)
                            print(f"[BEST] step={self.global_step} source=lex "
                                  f"qos={qos_now:.6f} (best_ant={_bqs}) "
                                  f"zc={_zcs} (best_ant={_bzcs}) "
                                  f"reward={reward_now:.4f} "
                                  f"nivel={getattr(self, '_lex_last_level', '?')} "
                                  f"[{getattr(self, '_lex_last_why', '')}] "
                                  f"confirm={self._lex_pass_streak}", flush=True)
                            self._lex_pass_streak = 0
                            self._lex_window_qos = float("inf")
                            self._lex_window_reward = float("inf")
                            self._lex_window_zc = float("inf")
                        else:
                            print(f"[BEST?] step={self.global_step} source=lex "
                                  f"qos={qos_now:.6f} zc={_zcs} reward={reward_now:.4f} "
                                  f"nivel={getattr(self, '_lex_last_level', '?')} "
                                  f"[{getattr(self, '_lex_last_why', '')}] "
                                  f"aguardando confirmacao {self._lex_pass_streak}/{need}", flush=True)
                    else:
                        # Antes esta rejeicao era SILENCIOSA: se o best parasse de
                        # evoluir, nao havia como saber qual nivel estava barrando.
                        # Só loga quando uma sequência de confirmação foi
                        # INTERROMPIDA (o caso informativo: quase virou best e
                        # algo barrou). Rejeições de rotina ficam silenciosas
                        # para não inundar o log a cada avaliação.
                        if self._lex_pass_streak > 0:
                            print(f"[BEST-] step={self.global_step} source=lex "
                                  f"qos={qos_now:.6f} (best={_bqs}) "
                                  f"zc={_zcs} (best={_bzcs}) "
                                  f"reward={reward_now:.4f} "
                                  f"barrado_em={getattr(self, '_lex_last_level', '?')} "
                                  f"[{getattr(self, '_lex_last_why', '')}]", flush=True)
                        self._lex_pass_streak = 0
                        self._lex_window_qos = float("inf")
                        self._lex_window_reward = float("inf")
                        self._lex_window_zc = float("inf")
            else:
                metric_name, metric_value = self._resolve_best_metric(metric_source, metric_source_name)
                if metric_value is not None and self._is_better(metric_value):
                    self.best_metric = metric_value
                    self.best_step = int(self.global_step)
                    if saved_path is None:
                        saved_path = self.ckpt_dir / f"ckpt_step_{int(self.global_step):012d}.pt"
                        self._save_checkpoint(saved_path)
                    self._mirror_best(saved_path, metric_value)
                    print(f"[BEST] step={self.global_step} source={metric_source_name} metric={metric_name} value={metric_value:.6f}", flush=True)
            if self.update_idx % 5 == 0:
                rel_count = float(rollout_metrics.get('reliability_terminal_count', 0.0))
                lat_count = float(rollout_metrics.get('latency_terminal_count', 0.0))
                tp_count = float(rollout_metrics.get('throughput_terminal_count', 0.0))
                nf_detail = ",".join(
                    f"{k}:{v}"
                    for k, v in sorted(getattr(self, "_nonfinite_counts", {}).items(), key=lambda kv: -int(kv[1]))[:8]
                )
                if not nf_detail:
                    nf_detail = "none"
                print(
                    f"[TRAIN] step={self.global_step} upd={self.update_idx} "
                    f"reward={rollout_metrics.get('rollout_reward_mean', 0.0):.3f} "
                    f"reward_src={rollout_metrics.get('rollout_reward_source', 'unknown')} "
                    f"episodes_finished={int(rollout_metrics.get('episodes_finished', 0.0))} "
                    f"rel={self._fmt_metric_value(rollout_metrics.get('train_reliability_mean', 0.0), rel_count)} "
                    f"lat={self._fmt_metric_value(rollout_metrics.get('train_latency_ms_mean', 0.0), lat_count, 'ms')} "
                    f"tp={self._fmt_metric_value(rollout_metrics.get('train_throughput_mean', 0.0), tp_count, 'Mbps')} "
                    f"qos={rollout_metrics.get('qos_target_met_mean', 0.0):.3f} "
                    f"active={rollout_metrics.get('active_drones_mean', 0.0):.3f} "
                    f"bad_deact={rollout_metrics.get('bad_deactivation_mean', 0.0):.3f} "
                    # diagnóstico do custo absoluto de zona: zpen = custo médio POR PASSO
                    # efetivamente subtraído do reward; zviol = fração de passos com violação.
                    f"zpen={rollout_metrics.get('zone_abs_pen_per_step_mean', 0.0):.3f} "
                    f"zviol={rollout_metrics.get('zone_viol_frac_mean', 0.0):.2f} "
                    f"nonfinite={int(rollout_metrics.get('nonfinite_total', 0.0))} "
                    f"nonfinite_detail={nf_detail}",
                    flush=True,
                )
        final_ckpt = self.ckpt_dir / f"ckpt_step_{int(self.global_step):012d}.pt"
        self._save_checkpoint(final_ckpt)
        self.writer.close()
        self.close()

    def close(self):
        try:
            self.envs.close()
        except Exception:
            pass
        try:
            self.eval_envs.close()
        except Exception:
            pass


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native PyTorch HAPPO trainer for Drone5G (Python 3.12 compatible)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=64)
    p.add_argument("--eval-envs", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=80_000_000)
    p.add_argument("--rollout-steps", type=int, default=199)
    p.add_argument("--eval-interval-updates", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--checkpoint-freq-updates", type=int, default=0,
                   help="If <=0, resolve automatically from checkpoint-target-timesteps.")
    p.add_argument("--checkpoint-target-timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="drone5g_happo_real")
    p.add_argument("--instance-tag", type=str, default="")
    p.add_argument("--results-root", type=str, default=str(Path.home() / "ray_results"))
    p.add_argument("--best-model-dir", type=str, default="./ppo_best_model")
    p.add_argument("--best-metric", type=str, default="eval_qos_target_met_mean")
    p.add_argument("--best-mode", type=str, default="max", choices=["max", "min"])
    p.add_argument("--best-min-delta", type=float, default=1e-9)
    # Tolerância do lex na escala do RUÍDO do eval: qos_mean sobre 100 episódios
    # binários tem erro-padrão ~0.05 perto de p=0.5; tol=0.01 deixava avaliações
    # "sortudas" virarem best e bloquearem políticas melhores. 0.03 ~ 1 EP típico.
    p.add_argument("--best-lex-tol", type=float, default=0.03)
    p.add_argument("--best-zone-tol", type=float, default=0.03,
                   help="tolerância do nível zone_clear no critério de best")
    # Nº de avaliações CONSECUTIVAS que precisam passar no critério antes de
    # espelhar o best (anti "dia de sorte"). 1 = comportamento antigo.
    p.add_argument("--best-confirm-evals", type=int, default=2)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--use-model", type=str, default="")
    p.add_argument("--restore-model-path", type=str, default="")

    # HAPPO hyperparameters
    p.add_argument("--policy-arch", type=str, default="256,256")
    p.add_argument("--lr-start", type=float, default=3e-4)
    p.add_argument("--critic-lr-start", type=float, default=2e-4)
    p.add_argument("--clip-start", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=4096)
    p.add_argument("--obs-norm", type=_str2bool, default=True)
    p.add_argument("--fixed-order", type=_str2bool, default=False)
    p.add_argument("--amp", type=str, default="auto", choices=["auto", "true", "false", "on", "off", "yes", "no", "1", "0"])
    p.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--compile-models", type=str, default="auto", choices=["auto", "true", "false", "on", "off", "yes", "no", "1", "0"])
    p.add_argument("--compile-mode", type=str, default="reduce-overhead")
    p.add_argument("--allow-tf32", type=_str2bool, default=True)

    # env args
    p.add_argument("--episode-horizon", type=int, default=1000)
    p.add_argument("--mask-actions", type=int, default=1)
    p.add_argument("--max-drones", type=int, default=30)
    p.add_argument("--max-out-connections", type=int, default=4)
    p.add_argument("--zonas-count", type=int, default=0)
    p.add_argument("--zona-w", type=int, default=5)
    p.add_argument("--zona-h", type=int, default=5)
    p.add_argument("--zona-single-fixed", type=int, default=1)
    p.add_argument("--p-sem-zona", type=float, default=1.0)
    p.add_argument("--obs-zone-map", type=int, default=1)
    p.add_argument("--zone-map-scale", type=int, default=4)
    p.add_argument("--obs-edge-cross", type=int, default=1)
    # Canal NxN de qualidade por enlace na obs (prob. do par c/ sombra). Muda a
    # dimensao de entrada da politica: exige treino do zero (sem --use-model).
    p.add_argument("--obs-link-quality", type=int, default=1)
    p.add_argument("--latency-from-longest-route", type=int, default=0,
                   help="1 = LEGADO: latência = atraso do caminho com mais saltos (herdado dos "
                        "modelos de espaço livre). 0 = usa o atraso agregado do modelo de Abreu "
                        "(worst_delay_ms), consistente com reliability e goodput")
    p.add_argument("--zone-violation-abs-penalty", type=float, default=0.0,
                   help="custo ABSOLUTO por passo de violar zona (fora do potencial). "
                        "O termo do potencial telescopa e a supressão só escala bônus positivos, "
                        "então nenhum dos dois torna violar pior que contornar. 0 = desligado")
    p.add_argument("--best-per-phase", type=int, default=1,
                   help="1 = mantém um best_dir por fase (best_model_<fase>), preservando o melhor "
                        "modelo de cada fase; 0 = comportamento antigo (um best global sobrescrito)")
    p.add_argument("--calibration-shadow-k", type=int, default=21,
                   help="nº de sorteios de sombreamento na calibração analítica do baseline de drones")
    p.add_argument("--calibration-min-drones-cap", type=int, default=14,
                   help="teto do baseline de drones (capacidade do meio compartilhado). Medido com o "
                        "solver p/ carrier_sense=200m: 14 drones fecham KPI em ~50%% dos casos, 16 e 18 "
                        "em 0%%. 0 desabilita o teto")
    p.add_argument("--calibration-min-drones-quantile", type=float, default=0.85,
                   help="percentil dos mínimos por sorteio que vira o baseline de drones. "
                        "0.5 reproduz o comportamento antigo (mínimo típico); 0.85 = mínimo robusto ao fade")
    p.add_argument("--converge-on-state", type=int, default=1)
    p.add_argument("--latency-headroom-penalty", type=float, default=0.5)
    p.add_argument("--rel-ramp-ext-w", type=float, default=0.0)
    # ---- ZONAS: evacuação ----
    p.add_argument("--zone-success-suppression", type=float, default=1.0,
                   help="fator multiplicativo dos bonus de sucesso/QoS quando ha violacao de zona (1=off, 0.1=forte)")
    p.add_argument("--zone-link-weight-mult", type=float, default=3.5,
                   help="links que cruzam zona pesam Nx mais que drones")
    p.add_argument("--zone-violation-penalty", type=float, default=6.0,
                   help="peso da evacuação de zonas no potencial (0=off)")
    p.add_argument("--p-zone-violation-seed", type=float, default=0.0,
                   help="prob. do reset SEMEAR violação (zona sobre a linha)")
    p.add_argument("--seed-initial-path", type=_str2bool, default=True,
                   help="se True, o reset semeia uma cadeia dirigida 0 -> ... -> N-1")
    p.add_argument("--seed-randomness", type=float, default=0.0,
                   help="intensidade da aleatoriedade da semeadura [0..1]: 0=topologia fixa, "
                        ">0 varia A/B, num. de caminhos e jitter por episodio (reprodutivel por seed; "
                        "viabilidade dos KPIs garantida por reducao automatica da intensidade).")
    p.add_argument("--link-neighbor-radius-m", type=float, default=400.0,
                   help="raio maximo (m) para CRIAR enlace; drones alem disso ficam "
                        "mascarados para criacao de link (nao afeta remocao). "
                        "0 = sem restricao. Reduz o espaco de acao efetivo aos vizinhos plausiveis.")
    p.add_argument("--seed-num-paths", type=int, default=2,
                   help="numero base de caminhos paralelos na semeadura (FASE 2: 2 = dificuldade media)")
    p.add_argument("--seed-target-hop-m", type=float, default=330.0,
                   help="comprimento-alvo do salto na semeadura em metros (FASE 2: 330 = dificuldade media, rel inicial ~0,76)")
    p.add_argument("--seed-viability-slack", type=float, default=1.15,
                   help="folga da reducao de viabilidade: preserva saltos-alvo ate (hop*slack); 1.15 mantem ~330m sem 'consertar'")
    p.add_argument("--seed-line-min", type=_str2bool, default=False,
                   help="MODO LINHA MINIMA: semeia a menor linha reta A->B que chega proximo de "
                        "atender os KPIs (analitico via Abreu, n_atende-1 hops); demais drones "
                        "DESATIVADOS/espalhados. A politica aprende a ativar e criar rotas.")
    p.add_argument("--activate-spawn-near-line", type=_str2bool, default=True,
                   help="Ao ativar um drone, reposiciona-o perto da linha media A->B (quarto mais "
                        "vazio, +-20 celulas, celula livre) e tenta criar 1 link valido com o ativo mais proximo.")
    p.add_argument("--drone-potential-cost", type=float, default=0.15,
                   help="Custo por drone ativo dentro do potencial Phi (ativar custa; so compensa "
                        "se fechar KPIs/marcos). Ensina a ativar apenas o necessario.")
    p.add_argument("--use-hybrid-reward", type=_str2bool, default=True,
                   help="Recompensa hibrida: primario potential-based (gamma*Phi'-Phi, elimina acumulo "
                        "em subotimo) + secundarios absolutos gateados por QoS (manutencao, drones, latencia).")
    p.add_argument("--qos-maintain-bonus", type=float, default=6.0,
                   help="Bonus por passo enquanto o QoS pleno e mantido (faz 'manter o otimo' > 'pairar').")
    p.add_argument("--converge-patience", type=int, default=30,
                   help="Termina o episodio apos N passos consecutivos com |Δreward|<eps, se o QoS ja "
                        "foi atingido. 0 desliga (roda ate o horizonte). Episodios mais curtos -> mais "
                        "episodios/compute e mais amostras por tempo de treino.")
    p.add_argument("--converge-eps", type=float, default=0.01,
                   help="Limiar de |Δreward| entre passos para considerar a solucao estavel.")

    # Abreu / QoS
    p.add_argument("--abreu-enabled", type=int, default=1)
    p.add_argument("--abreu-max-paths", type=int, default=64)
    p.add_argument("--traffic-rate-mbps", type=float, default=10.0)
    p.add_argument("--frame-size-bytes", type=int, default=1500)
    p.add_argument("--buffer-size-packets", type=int, default=20)
    p.add_argument("--phy-rate-mbps", type=float, default=54.0)
    p.add_argument("--slot-time-us", type=float, default=9.0)
    p.add_argument("--difs-us", type=float, default=28.0)
    p.add_argument("--sifs-us", type=float, default=10.0)
    p.add_argument("--ack-us", type=float, default=24.0)
    p.add_argument("--ack-timeout-us", type=float, default=34.0)
    p.add_argument("--cw-min", type=int, default=15)
    p.add_argument("--cw-max", type=int, default=1023)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--carrier-sense-range-m", type=float, default=120.0)
    # Legacy Abreu solver thresholds. They no longer define the environment QoS target.
    p.add_argument("--abreu-reliability-min", type=float, default=RELIABILITY_TARGET_DEFAULT)
    p.add_argument("--abreu-delay-max-ms", type=float, default=MAX_DELAY_MS_DEFAULT)
    p.add_argument("--success-max-latency-ms", type=float, default=MAX_DELAY_MS_DEFAULT,
                   help="Legacy alias kept for compatibility; the effective latency target is --max-delay-ms.")
    # Effective QoS targets used by the environment unless explicitly overridden.
    p.add_argument("--reliability-target", type=float, default=RELIABILITY_TARGET_DEFAULT)
    p.add_argument("--delivery-ratio-target", type=float, default=0.999)
    p.add_argument("--prune-delivery-min", type=float, default=0.30)
    p.add_argument("--prune-rel-min", type=float, default=0.20)
    p.add_argument("--drone-active-cost-before-qos", type=float, default=0.02)
    p.add_argument("--bad-deactivation-penalty", type=float, default=0.80)
    p.add_argument("--max-delay-ms", type=float, default=MAX_DELAY_MS_DEFAULT)
    p.add_argument("--throughput-max-mbps", type=float, default=THROUGHPUT_MAX_MBPS_DEFAULT)
    p.add_argument("--abreu-solver-gpu", type=_str2bool, default=False)

    # Curriculum / block training helpers
    p.add_argument("--timesteps-mode", type=str, default="total", choices=["total", "incremental"],
                   help="total: --timesteps é o alvo global; incremental: roda mais --timesteps a partir do checkpoint carregado.")
    p.add_argument("--curriculum-phase-name", type=str, default="")
    p.add_argument("--curriculum-phase-idx", type=int, default=-1)
    p.add_argument("--curriculum-block-idx", type=int, default=-1)
    p.add_argument("--latest-metrics-path", type=str, default="",
                   help="Caminho opcional para salvar JSON com métricas recentes; se vazio usa results_root/latest_metrics.json e run_dir/artifacts/latest_metrics.json.")
    return p


def main():
    args = build_parser().parse_args()
    trainer = NativeHAPPOTrainer(args)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
