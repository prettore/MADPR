#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5GMLMA.py (pure RLlib / Python 3.12)

- 2 policies:
    * anchor : agent_id "anchor_*"
    * inter  : agent_id "inter_*"
- Policy shared by group + CTDE (centralized critic via obs["state"]).
- Action masking via a custom Torch model.
- Clean final metrics in TensorBoard (eval_terminal/*).

Docs:
- RLlib Multi-Agent (turn-based): https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
- Tune run(): https://docs.ray.io/en/latest/tune/api/doc/ray.tune.run.html"""

from __future__ import annotations
import os, time, math, argparse, shutil, re, json

from pathlib import Path

# -------------------------
# Default directories (without depending on parameters)
# -------------------------
_BASE_DIR = Path(__file__).resolve().parent
_WORKDIR = Path(os.environ.get("DRONE5G_WORKDIR", str(_BASE_DIR / "marl_workdir"))).resolve()
_TB_ROOT = (_WORKDIR / "tb").resolve()
_SAVES_ROOT = (_WORKDIR / "saves").resolve()
_TB_ROOT.mkdir(parents=True, exist_ok=True)
_SAVES_ROOT.mkdir(parents=True, exist_ok=True)


# Makes Ray (AIR/driver_artifacts) write inside the workdir, not in /tmp/ray.
# Important for running multiple instances in parallel without mixing logs of Ray:
# each instance gets its own RAY_TMPDIR (with a short symlink under /tmp).
def _sanitize_tag(s: str, max_len: int = 32) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    if not s:
        return ""
    return s[:max_len]


def setup_ray_tmp(instance_key: str) -> Path:
    # Uses a short tag to avoid exceeding the limit of socket path (AF_UNIX).
    tag = _sanitize_tag(instance_key, max_len=24) or f"pid{os.getpid()}"
    ray_tmp_real = (_TB_ROOT / f"ray_tmp_{tag}").resolve()
    ray_tmp_real.mkdir(parents=True, exist_ok=True)

    ray_tmp_sym = Path(f"/tmp/ray_marl_{tag}")
    try:
        # If it exists and is symlink pointing elsewhere, recreate it.
        if ray_tmp_sym.exists() or ray_tmp_sym.is_symlink():
            try:
                if ray_tmp_sym.is_symlink():
                    cur = ray_tmp_sym.resolve()
                    if cur != ray_tmp_real:
                        ray_tmp_sym.unlink()
                else:
                    # exists as a normal directory/file: leave it unchanged, use it as a fallback
                    pass
            except Exception:
                pass
        if not ray_tmp_sym.exists() and not ray_tmp_sym.is_symlink():
            ray_tmp_sym.symlink_to(ray_tmp_real, target_is_directory=True)
    except Exception:
        pass

    ray_tmp = ray_tmp_sym if (ray_tmp_sym.exists() or ray_tmp_sym.is_symlink()) else ray_tmp_real

    # Override (do not use setdefault): each instance needs its own temporary directory.
    os.environ["RAY_TMPDIR"] = str(ray_tmp)
    return ray_tmp


# Suppresses warning when num_gpus=0 (or when Ray does not detect accelerator_type)
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
from typing import Tuple

# TensorBoard (custom separate logdir)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# Avoids oversubscription on large machines
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import TBXLoggerCallback
from ray.tune.stopper import Stopper

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import torch
import torch.nn as nn

from drone_env_multi import Drone5GMAEnv


# -------------------------
# Torch model: action mask + centralized critic (state)
# -------------------------
class TorchActionMaskCCModel(TorchModelV2, nn.Module):
    '''- Policy logits: obs["obs"] (or a flat vector)
- Mask applied: obs["action_mask"] (or a slice at the end of the flat vector)
- Value function: obs["state"] (CTDE) (or a slice in the flat vector)'''

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # RLlib may flatten Dict->Box via the preprocessor.
        # We support both formats:
        #  - Dict: {"obs": ..., "state": ..., "action_mask": ...}
        # - Box : concat [obs, state, action_mask] (order defined by the environment Dict)
        cmc = dict((model_config or {}).get('custom_model_config', {}) or {})
        cmc.update(kwargs or {})  # newer RLlib may pass custom options via **kwargs
        # Limit torch thread usage inside Ray workers (prevents oversubscription at high parallelism).
        try:
            tn = int(os.environ.get('TORCH_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1')))
            torch.set_num_threads(max(1, tn))
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        self._is_dict = hasattr(obs_space, "spaces") and isinstance(getattr(obs_space, "spaces", None),
                                                                    dict) and "obs" in obs_space.spaces

        if self._is_dict:
            obs_space_obs = obs_space["obs"]
            obs_space_state = obs_space["state"]
            self._obs_dim = int(obs_space_obs.shape[0])
            self._state_dim = int(obs_space_state.shape[0])
            self._mask_dim = int(obs_space["action_mask"].shape[0])
        else:
            # Dimensions must be provided by custom_model_config (per policy).
            try:
                self._obs_dim = int(cmc["obs_dim"])
                self._state_dim = int(cmc.get("state_dim", 0))
                self._mask_dim = int(cmc["mask_dim"])
            except Exception as e:
                raise ValueError(
                    "ObsSpace chegou como Box (flatten). Forneça custom_model_config "
                    "com obs_dim/state_dim/mask_dim no policy config."
                ) from e

            obs_space_obs = gym.spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float32)
            if self._state_dim > 0:
                obs_space_state = gym.spaces.Box(-np.inf, np.inf, shape=(self._state_dim,), dtype=np.float32)
            else:
                obs_space_state = obs_space_obs

        self.policy_net = FullyConnectedNetwork(
            obs_space_obs, action_space, num_outputs, model_config, name + "_pi"
        )
        self.value_net = FullyConnectedNetwork(
            obs_space_state, action_space, 1, model_config, name + "_vf"
        )
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs_in = input_dict["obs"]

        # 1) If a Dict is received (the case that is currently failing), use the keys.
        if isinstance(obs_in, dict) and "obs" in obs_in:
            obs_vec = obs_in["obs"]
            state_vec = obs_in.get("state", obs_vec)
            mask = obs_in.get("action_mask", None)

        else:
            # 2) If it arrives flattened, prefer obs_flat (guaranteed by RLlib)
            flat = input_dict.get("obs_flat", obs_in)

            obs_vec = flat[:, : self._obs_dim]

            if self._state_dim > 0:
                s0 = self._obs_dim
                s1 = self._obs_dim + self._state_dim
                state_vec = flat[:, s0:s1]
            else:
                state_vec = obs_vec

            m0 = self._obs_dim + self._state_dim
            m1 = m0 + self._mask_dim
            mask = flat[:, m0:m1]

        logits, _ = self.policy_net({"obs": obs_vec})

        if mask is not None:
            mask = mask.to(logits.device).float()

            if mask.shape[-1] != logits.shape[-1]:
                if mask.shape[-1] > logits.shape[-1]:
                    mask = mask[..., :logits.shape[-1]]
                else:
                    pad = logits.shape[-1] - mask.shape[-1]
                    mask = torch.cat(
                        [mask, torch.ones(mask.shape[0], pad, device=mask.device, dtype=mask.dtype)],
                        dim=-1
                    )

            NEG_INF = -1e9
            inf_mask = torch.where(mask > 0.0, torch.zeros_like(mask), torch.full_like(mask, NEG_INF))

            masked_logits = logits + inf_mask
        else:
            masked_logits = logits

        vf, _ = self.value_net({"obs": state_vec})
        self._value_out = vf.squeeze(-1)
        return masked_logits, state

    def value_function(self):
        return self._value_out


# -------------------------
# Callbacks: final metrics “clean” + Best-model (evaluation-only)
# -------------------------
class TerminalMetricsCallbacks(DefaultCallbacks):
    """Collect terminal metrics from env infos and expose them in TensorBoard.

    - Stores per-episode values under episode.custom_metrics (RLlib aggregates into *_mean/min/max).
    - Stores per-episode values under episode.hist_data so we can compute percentiles (p10).
    - Adds friendly top-level aliases under result dict to create TensorBoard groups:
        rollout/*, main/*, eval_terminal/*, custom/*.
    - Saves a "best model" checkpoint (evaluation-only) whenever the selected eval metric improves:
        - project best: <project_root>/ppo_best_model   (default)
        - trial best:   <trial_logdir>/ppo_best_model
      The best dir itself is a RLlib checkpoint directory (loadable via Algorithm.from_checkpoint(best_dir)).
    """

    def __init__(self):
        super().__init__()
        self._best_inited = False
        self._best_score = None
        self._best_metric = os.environ.get("PPO_BEST_METRIC", "eval_terminal/reliability_terminal_mean")
        self._best_mode = os.environ.get("PPO_BEST_MODE", "max").strip().lower()
        self._best_min_delta = float(os.environ.get("PPO_BEST_MIN_DELTA", "1e-9") or 1e-9)
        self._best_dir_env = os.environ.get("PPO_BEST_MODEL_DIR", "").strip()
        self._best_copy_mode = "copy"  # forced: always copy (no symlinks)
        self._project_best_dir = None
        self._trial_best_dir = None
        self._best_info_key = "_BEST_INFO.json"
        self._best_bundle_json = "test_bundle.json"
        self._best_bundle_pkl = "test_bundle.pkl"

        # --- Custom TensorBoard (separate logdir) ---
        self._custom_tb_enable = int(os.environ.get("PPO_CUSTOM_TB_ENABLE", "1") or "1") != 0
        self._custom_tb_root_env = os.environ.get("PPO_CUSTOM_TB_ROOT", "").strip()
        self._custom_tb_root = None
        self._custom_tb_dir = None
        self._custom_tb_writer = None
        self._custom_tb_hparams_written = False
        self._custom_tb_step_key = os.environ.get("PPO_CUSTOM_TB_STEP_KEY",
                                                  "timesteps_total").strip() or "timesteps_total"

    # ---------- metric helpers ----------
    @staticmethod
    def _get_hist_stats(res: dict) -> dict:
        if not isinstance(res, dict):
            return {}
        hs = res.get("hist_stats")
        if isinstance(hs, dict):
            return hs
        sr = res.get("sampler_results")
        if isinstance(sr, dict) and isinstance(sr.get("hist_stats"), dict):
            return sr["hist_stats"]
        return {}

    @staticmethod
    def _get_custom_metrics(res: dict) -> dict:
        if not isinstance(res, dict):
            return {}
        cm = res.get("custom_metrics")
        if isinstance(cm, dict):
            return cm
        # Some variants nest under env_runners or sampler_results.
        for k in ("env_runners", "sampler_results"):
            sub = res.get(k)
            if isinstance(sub, dict):
                cm2 = sub.get("custom_metrics")
                if isinstance(cm2, dict):
                    return cm2
        return {}

    @staticmethod
    def _cm_mean(res: dict, base_key: str):
        cm = TerminalMetricsCallbacks._get_custom_metrics(res)
        return cm.get(f"{base_key}_mean", np.nan) if isinstance(cm, dict) else np.nan

    @staticmethod
    def _hist_p10(res: dict, key: str):
        hs = TerminalMetricsCallbacks._get_hist_stats(res)
        vals = hs.get(key)
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            return np.nan
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.percentile(arr, 10))

    @staticmethod
    def _set_scalar(result: dict, key: str, val):
        try:
            v = float(val)
        except Exception:
            return
        if math.isfinite(v):
            result[key] = v

    # ---------- best-model helpers ----------
    @staticmethod
    def _safe_rmtree(p: Path):
        try:
            if p.is_symlink():
                p.unlink()
            elif p.exists():
                shutil.rmtree(p)
        except Exception:
            pass

    @staticmethod
    def _ensure_writable_dir(p: Path, fallback: Path | None = None) -> Path:
        """Ensure directory exists and is writable. If not, fallback (if provided)."""
        p = Path(p).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            # need X_OK too to create files inside
            if os.access(str(p), os.W_OK | os.X_OK):
                # write test (covers weird FS/ACL cases)
                testf = p / ".__writetest__"
                with open(testf, "w", encoding="utf-8") as f:
                    f.write("ok")
                try:
                    testf.unlink()
                except Exception:
                    pass
                return p
            raise PermissionError(f"Sem permissão W/X em {p}")
        except Exception as e:
            if fallback is None:
                raise
            fb = Path(fallback).expanduser()
            fb.mkdir(parents=True, exist_ok=True)
            return fb

    # ---------- custom TB helpers ----------
    @staticmethod
    def _hparam_val(v):
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        try:
            return str(v)
        except Exception:
            return repr(v)

    @classmethod
    def _flatten_hparams(cls, d: dict, prefix: str = "", out: dict | None = None, max_items: int = 300) -> dict:
        """Flatten nested dict into a simple {str: scalar/str} dict (TensorBoard hparams-friendly)."""
        if out is None:
            out = {}
        if not isinstance(d, dict):
            return out
        for k, v in d.items():
            if len(out) >= max_items:
                break
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                cls._flatten_hparams(v, prefix=key + ".", out=out, max_items=max_items)
            else:
                out[key] = cls._hparam_val(v)
        return out

    def _custom_tb_init(self, algorithm):
        """Init a separate TensorBoard SummaryWriter under PPO_CUSTOM_TB_ROOT/<run>/<trial>."""
        if not self._custom_tb_enable:
            return
        if SummaryWriter is None:
            return
        if self._custom_tb_writer is not None:
            return

        # root
        root = Path(self._custom_tb_root_env).expanduser() if self._custom_tb_root_env else (_WORKDIR / "tb_custom")
        try:
            root = root.resolve()
        except Exception:
            root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self._custom_tb_root = root

        # derive run/trial names from algorithm.logdir
        try:
            logdir = Path(getattr(algorithm, "logdir", str(_SAVES_ROOT))).expanduser().resolve()
        except Exception:
            logdir = _SAVES_ROOT
        run_name = logdir.parent.name if logdir.parent is not None else "run"
        trial_name = logdir.name if logdir.name else "trial"

        tb_dir = (root / run_name / trial_name).resolve()
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._custom_tb_dir = tb_dir

        try:
            self._custom_tb_writer = SummaryWriter(str(tb_dir))
            print(f"[INFO] CustomTB: {tb_dir}")
        except Exception as e:
            self._custom_tb_writer = None
            print(f"[WARN] Falha ao criar SummaryWriter em {tb_dir}: {type(e).__name__}: {e}")
            return

        # Write hparams once
        try:
            cfg_obj = None
            if hasattr(algorithm, "get_config"):
                try:
                    cfg_obj = algorithm.get_config()
                except Exception:
                    cfg_obj = None
            cfg = None
            if isinstance(cfg_obj, dict):
                cfg = cfg_obj
            elif cfg_obj is not None and hasattr(cfg_obj, "to_dict"):
                try:
                    cfg = cfg_obj.to_dict()
                except Exception:
                    cfg = None
            if not isinstance(cfg, dict):
                cfg = getattr(algorithm, "config", None)
                if hasattr(cfg, "to_dict"):
                    try:
                        cfg = cfg.to_dict()
                    except Exception:
                        cfg = {}
                if not isinstance(cfg, dict):
                    cfg = {}
            hp = self._flatten_hparams(cfg, prefix="", out={}, max_items=400)

            # Add a few key env vars (useful)
            hp.update({
                "env.PPO_BEST_METRIC": os.environ.get("PPO_BEST_METRIC", ""),
                "env.PPO_BEST_MODE": os.environ.get("PPO_BEST_MODE", ""),
                "env.DRONE5G_WORKDIR": str(_WORKDIR),
            })

            # add_hparams needs a metric dict
            self._custom_tb_writer.add_hparams(hp, {"hparams/ok": 1.0})
            # Also store as text (truncated)
            try:
                js = json.dumps(hp, ensure_ascii=False, indent=2)
                if len(js) > 20000:
                    js = js[:20000] + "\n... (truncado) ..."
                self._custom_tb_writer.add_text("hparams/flat", js, 0)
            except Exception:
                pass
            self._custom_tb_hparams_written = True
        except Exception as e:
            print(f"[WARN] CustomTB hparams falhou: {type(e).__name__}: {e}")

    def _custom_tb_log_scalars(self, result: dict):
        """Duplicate selected scalars into the custom TB logdir (separate from Ray)."""
        if self._custom_tb_writer is None:
            return
        if not isinstance(result, dict):
            return
        step = result.get(self._custom_tb_step_key)
        if not isinstance(step, (int, float)):
            step = result.get("timesteps_total") or result.get("training_iteration") or 0
        step = int(step) if isinstance(step, (int, float)) else 0

        prefixes = ("main/", "eval_terminal/", "custom/", "rollout/", "lex/")
        for k, v in result.items():
            if not isinstance(k, str):
                continue
            if not k.startswith(prefixes):
                continue
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                try:
                    self._custom_tb_writer.add_scalar(k, float(v), step)
                except Exception:
                    pass
        try:
            self._custom_tb_writer.flush()
        except Exception:
            pass

    def _best_init(self, algorithm):
        if self._best_inited:
            return

        # Default best dir: project root (same dir as this script)
        proj_default = (_BASE_DIR / "ppo_best_model").resolve()
        proj_dir = Path(self._best_dir_env).expanduser() if self._best_dir_env else proj_default
        proj_dir = self._ensure_writable_dir(proj_dir, fallback=proj_default)

        # Also keep a per-trial best dir
        try:
            logdir = Path(getattr(algorithm, "logdir", str(_SAVES_ROOT))).expanduser().resolve()
        except Exception:
            logdir = _SAVES_ROOT
        trial_dir = self._ensure_writable_dir(logdir / "ppo_best_model", fallback=proj_dir)

        self._project_best_dir = proj_dir
        self._trial_best_dir = trial_dir

        # Load previous best (for resume/continue)
        self._best_score = None
        info_path = self._project_best_dir / self._best_info_key
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                self._best_score = float(info.get("score"))
            except Exception:
                self._best_score = None

        self._best_inited = True

    def _get_eval_metric_value(self, eval_res: dict) -> float:
        """Fetch metric value from evaluation result dict (evaluation-only)."""
        if not isinstance(eval_res, dict):
            return float("nan")

        # Allow metric in different forms:
        # - exact key (e.g., "eval_terminal/reliability_terminal_mean") in eval custom_metrics
        # - base key without suffix -> try adding _mean
        cm = self._get_custom_metrics(eval_res)

        key = (self._best_metric or "").strip()
        if not key:
            return float("nan")

        # 1) direct in custom_metrics
        if isinstance(cm, dict) and key in cm:
            try:
                return float(cm[key])
            except Exception:
                pass

        # 2) if it's a base key, try _mean
        if isinstance(cm, dict) and not key.endswith("_mean"):
            k2 = f"{key}_mean"
            if k2 in cm:
                try:
                    return float(cm[k2])
                except Exception:
                    pass

        # 3) maybe user passed "main/..." or "eval_terminal/..." already placed in result
        try:
            v = eval_res.get(key)
            if v is not None:
                return float(v)
        except Exception:
            pass

        return float("nan")

    # ---------- best-model helpers ----------
    def _mirror_checkpoint_dir(self, src_ckpt: Path, dst_dir: Path):
        """Make dst_dir become a loadable checkpoint directory pointing to src_ckpt.

Prefer a symlink if PPO_BEST_COPY_MODE=symlink; otherwise use copytree.
Fallback automatically.

Note: in some RLlib versions or wrapper return formats, `algorithm.save()`
may return a *file* inside the checkpoint directory. In that case,
normalize it to the parent directory."""
        src_ckpt = Path(src_ckpt).expanduser().resolve()
        dst_dir = Path(dst_dir).expanduser()

        # Normalizes: if a file was returned, use the parent directory.
        try:
            if src_ckpt.is_file():
                src_ckpt = src_ckpt.parent
        except OSError:
            # If stat fails, leave it unchanged so the copy operation raises a clearer error.
            pass

        if not src_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint fonte não existe: {src_ckpt}")
        if not src_ckpt.is_dir():
            raise NotADirectoryError(f"Checkpoint fonte não é diretório: {src_ckpt}")

        # Remove previous
        self._safe_rmtree(dst_dir)

        # Try symlink first (fast)
        if self._best_copy_mode == "symlink":
            try:
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                dst_dir.symlink_to(src_ckpt, target_is_directory=True)
                return
            except Exception:
                # fall back to copy
                pass

        # Copy (robust)
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_ckpt, dst_dir)

    def _write_best_sidecars(self, best_dir: Path, info: dict):
        """Write sidecars and a self-contained recovery bundle into best_dir.

`best_dir` already contains the RLlib checkpoint mirrored by `_mirror_checkpoint_dir`.
Here we:
  1) write the sidecars (_BEST_INFO/test_bundle.*),
  2) copy the code files required for restore/evaluation,
  3) write recovery_manifest.json describing what was embedded.

This makes `ppo_best_model` loadable without relying only on external references
such as `checkpoint_source`/`trial_logdir`."""
        try:
            best_dir = Path(best_dir)
            best_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        # 1) info json
        try:
            (best_dir / self._best_info_key).write_text(
                json.dumps(info, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

        # 2) test bundle json + pkl (helps Jupyter evaluator)
        bundle = {
            "saved_at": info.get("saved_at"),
            "metric": info.get("metric"),
            "mode": info.get("mode"),
            "score": info.get("score"),
            "checkpoint_source": info.get("checkpoint_source"),
            "trial_logdir": info.get("trial_logdir"),
            "env_config": info.get("env_config"),
            "run_name": info.get("run_name"),
        }
        try:
            (best_dir / self._best_bundle_json).write_text(
                json.dumps(bundle, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass
        try:
            import pickle
            with open(best_dir / self._best_bundle_pkl, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

        # 3) embeds the code files used for restore/evaluation
        embedded_files = {}
        search_roots = []
        for cand in (
            _BASE_DIR,
            Path.cwd(),
            Path(info.get("trial_logdir", "")).expanduser().parent if info.get("trial_logdir") else None,
            Path(info.get("checkpoint_source", "")).expanduser().parent if info.get("checkpoint_source") else None,
        ):
            try:
                if cand is not None:
                    cand = cand.resolve()
                    if cand.exists() and cand not in search_roots:
                        search_roots.append(cand)
            except Exception:
                pass

        # Copies the trainer script itself always with this name.
        try:
            trainer_src = Path(__file__).resolve()
            trainer_dst = best_dir / Path(__file__).name
            if trainer_src.exists():
                shutil.copy2(trainer_src, trainer_dst)
                embedded_files[trainer_dst.name] = str(trainer_src)
        except Exception as e:
            print(f"[WARN] Falha ao copiar trainer para BEST: {type(e).__name__}: {e}")

        companion_names = [
            "drone_env_multi.py",
            "calculaProbLink.py",
            "calculaProbLinkOtimizado.py",
            "calculaProbLink_otimizado.py",
        ]
        for fname in companion_names:
            src_found = None
            for root in search_roots:
                cand = root / fname
                try:
                    if cand.exists() and cand.is_file():
                        src_found = cand.resolve()
                        break
                except Exception:
                    continue
            if src_found is None:
                continue
            try:
                shutil.copy2(src_found, best_dir / fname)
                embedded_files[fname] = str(src_found)
            except Exception as e:
                print(f"[WARN] Falha ao copiar {fname} para BEST: {type(e).__name__}: {e}")

        manifest = {
            "saved_at": info.get("saved_at"),
            "checkpoint_dir": str(best_dir),
            "checkpoint_source": info.get("checkpoint_source"),
            "trial_logdir": info.get("trial_logdir"),
            "run_name": info.get("run_name"),
            "embedded_files": embedded_files,
            "required_runtime_files_present": {
                "checkpoint": bool((best_dir / "rllib_checkpoint.json").exists() or (best_dir / "algorithm_state.pkl").exists() or (best_dir / "algorithm_state.msgpack").exists() or (best_dir / "algorithm_state.msgpck").exists()),
                "trainer": bool((best_dir / Path(__file__).name).exists()),
                "env": bool((best_dir / "drone_env_multi.py").exists()),
            },
        }
        try:
            (best_dir / "recovery_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _maybe_save_best(self, algorithm, result: dict) -> bool:
        """If evaluation metric improved, save new best and mirror to best dirs."""
        eval_res = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else None
        if not eval_res:
            return False

        self._best_init(algorithm)

        score = self._get_eval_metric_value(eval_res)
        if not math.isfinite(score):
            return False

        improved = False
        if self._best_score is None or not math.isfinite(float(self._best_score)):
            improved = True
        else:
            prev = float(self._best_score)
            delta = float(self._best_min_delta)
            if self._best_mode == "min":
                improved = score < (prev - delta)
            else:
                improved = score > (prev + delta)

        # Always expose current score (even if not improved)
        self._set_scalar(result, "custom/best_eval_score", score)
        if self._best_score is not None and math.isfinite(float(self._best_score)):
            self._set_scalar(result, "custom/best_eval_score_best", float(self._best_score))

        if not improved:
            return False

        # Save new checkpoint right now
        try:
            ckpt_obj = algorithm.save()
            try:
                ckpt_path = ckpt_obj.path if hasattr(ckpt_obj, "path") else str(ckpt_obj)
            except Exception:
                ckpt_path = str(ckpt_obj)
            src_ckpt = Path(ckpt_path).expanduser().resolve()
        except Exception as e:
            print(f"[WARN] Falha ao salvar checkpoint BEST via algorithm.save(): {type(e).__name__}: {e}")
            return False

        # Mirror into both dirs
        try:
            self._mirror_checkpoint_dir(src_ckpt, self._project_best_dir)
        except Exception as e:
            # if project dir fails for any reason, keep trial best at least
            print(f"[WARN] Falha ao espelhar BEST para project_best_dir={self._project_best_dir}: {type(e).__name__}: {e}")
        try:
            self._mirror_checkpoint_dir(src_ckpt, self._trial_best_dir)
        except Exception as e:
            print(f"[WARN] Falha ao espelhar BEST para trial_best_dir={self._trial_best_dir}: {type(e).__name__}: {e}")

        # Update score + sidecars
        self._best_score = float(score)
        info = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metric": self._best_metric,
            "mode": self._best_mode,
            "min_delta": float(self._best_min_delta),
            "score": float(score),
            "checkpoint_source": str(src_ckpt),
            "trial_logdir": str(getattr(algorithm, "logdir", "")),
            "env_config": (getattr(algorithm, "config", {}) or {}).get("env_config", None),
            "run_name": (getattr(algorithm, "config", {}) or {}).get("name", None),
        }
        # Write into project best + trial best
        if self._project_best_dir is not None:
            self._write_best_sidecars(self._project_best_dir, info)
        if self._trial_best_dir is not None:
            self._write_best_sidecars(self._trial_best_dir, info)

        self._set_scalar(result, "custom/best_saved", 1.0)
        return True

    # ---------- RLlib hooks ----------
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index: int, **kwargs):
        last_info = None
        for aid in episode.get_agents():
            info = episode.last_info_for(aid)
            if info and any(k.endswith("_terminal") for k in info.keys()):
                last_info = info

        if not last_info:
            return

        rel_t = float(last_info.get("reliability_terminal", np.nan))
        zone_t = float(last_info.get("zone_violations_terminal",
                                     last_info.get("zone_violation_terminal", np.nan)))
        right_ratio_t = float(last_info.get("right_links_ratio_terminal", np.nan))
        success_t = float(last_info.get("success_terminal", last_info.get("is_success", 0)))
        tp_t = float(last_info.get("throughput_terminal", last_info.get("throughput", np.nan)))

        # Custom metrics (RLlib will aggregate into *_mean/min/max)
        episode.custom_metrics["eval_terminal/reliability_terminal"] = rel_t
        episode.custom_metrics["eval_terminal/zone_violations_terminal"] = zone_t
        episode.custom_metrics["eval_terminal/right_links_ratio_terminal"] = right_ratio_t
        episode.custom_metrics["eval_terminal/success_ratio"] = success_t
        episode.custom_metrics["eval_terminal/throughput_terminal"] = tp_t

        # Hist data (for percentiles)
        episode.hist_data["eval_terminal/reliability_terminal"] = [rel_t]
        episode.hist_data["eval_terminal/zone_violations_terminal"] = [zone_t]
        episode.hist_data["eval_terminal/right_links_ratio_terminal"] = [right_ratio_t]
        episode.hist_data["eval_terminal/success_ratio"] = [success_t]
        episode.hist_data["eval_terminal/throughput_terminal"] = [tp_t]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Liveness signal: ensures a tag in TensorBoard/CSV even if metrics do not exist yet
        result["debug/callback_on_train_result"] = 1.0

        # Custom TB (separate from Ray)
        self._custom_tb_init(algorithm)

        # --- Rollout group (training sampler) ---
        self._set_scalar(result, "rollout/ep_rew_mean", result.get("episode_reward_mean", np.nan))
        self._set_scalar(result, "rollout/success_rate", self._cm_mean(result, "eval_terminal/success_ratio"))

        eval_res = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else None

        # --- Main group (keeps compatibility: prefer evaluation, fallback training) ---
        src = eval_res if eval_res else result

        rel_mean = self._cm_mean(src, "eval_terminal/reliability_terminal")
        zone_mean = self._cm_mean(src, "eval_terminal/zone_violations_terminal")
        right_mean = self._cm_mean(src, "eval_terminal/right_links_ratio_terminal")
        succ_mean = self._cm_mean(src, "eval_terminal/success_ratio")
        tp_mean = self._cm_mean(src, "eval_terminal/throughput_terminal")

        rel_p10 = self._hist_p10(src, "eval_terminal/reliability_terminal")
        if not math.isfinite(rel_p10) and eval_res:
            rel_p10 = self._hist_p10(result, "eval_terminal/reliability_terminal")

        self._set_scalar(result, "main/reliability_terminal_mean", rel_mean)
        self._set_scalar(result, "main/reliability_terminal_p10", rel_p10)
        self._set_scalar(result, "main/terminal_rigth_links", right_mean)
        self._set_scalar(result, "main/success_rate", succ_mean)
        self._set_scalar(result, "main/violation_zone_mean", zone_mean)
        self._set_scalar(result, "main/throughput_terminal_mean", tp_mean)

        # Extra aliases (typos)
        self._set_scalar(result, "main/ralayability_terminal_mean", rel_mean)
        self._set_scalar(result, "main/relayability_terminal_p10", rel_p10)

        # --- eval_terminal group (evaluation-only; separated in TB) ---
        if eval_res:
            self._set_scalar(result, "eval_terminal/reliability_terminal_mean",
                             self._cm_mean(eval_res, "eval_terminal/reliability_terminal"))
            self._set_scalar(result, "eval_terminal/reliability_terminal_p10",
                             self._hist_p10(eval_res, "eval_terminal/reliability_terminal"))
            self._set_scalar(result, "eval_terminal/zone_violations_terminal_mean",
                             self._cm_mean(eval_res, "eval_terminal/zone_violations_terminal"))
            self._set_scalar(result, "eval_terminal/right_links_ratio_terminal_mean",
                             self._cm_mean(eval_res, "eval_terminal/right_links_ratio_terminal"))
            self._set_scalar(result, "eval_terminal/success_ratio_mean",
                             self._cm_mean(eval_res, "eval_terminal/success_ratio"))
            self._set_scalar(result, "eval_terminal/throughput_terminal_mean",
                             self._cm_mean(eval_res, "eval_terminal/throughput_terminal"))

        # --- best model (evaluation-only) ---
        saved = self._maybe_save_best(algorithm, result)
        if not saved:
            self._set_scalar(result, "custom/best_saved", 0.0)

        # Duplicates scalars into tb_custom (optional).
        self._custom_tb_log_scalars(result)

    def on_training_end(self, *, algorithm, **kwargs):
        # Close custom TB writer
        try:
            if self._custom_tb_writer is not None:
                self._custom_tb_writer.flush()
                self._custom_tb_writer.close()
        except Exception:
            pass
        self._custom_tb_writer = None


def _auto_workers(max_speed: bool) -> Tuple[int, int]:
    cpu = os.cpu_count() or 32
    reserve = 8 if cpu >= 64 else 4
    avail = max(1, cpu - reserve)
    if not max_speed:
        return min(32, avail), 1
    return min(int(os.environ.get('MAX_SPEED_WORKERS_CAP', '80')), avail), (2 if avail >= 32 else 1)



class ReliabilityPlateauStopper(Stopper):
    """Stop training when evaluation reliability stalls or collapses.

    Reads the top-level scalar `eval_terminal/reliability_terminal_mean` produced by
    TerminalMetricsCallbacks. If absent, falls back to the nested evaluation custom metric.
    """

    def __init__(self, patience_evals: int = 8, min_delta: float = 1e-3, max_drop: float = 0.05):
        self.patience_evals = max(1, int(patience_evals))
        self.min_delta = float(min_delta)
        self.max_drop = float(max_drop)
        self.best = None
        self.bad_evals = 0
        self._stopped = False

    @staticmethod
    def _extract_rel(result: dict) -> float:
        try:
            v = result.get("eval_terminal/reliability_terminal_mean", None)
            if v is not None and math.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
        try:
            ev = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else None
            if isinstance(ev, dict):
                cm = ev.get("custom_metrics") if isinstance(ev.get("custom_metrics"), dict) else {}
                v = cm.get("eval_terminal/reliability_terminal_mean", None)
                if v is not None and math.isfinite(float(v)):
                    return float(v)
        except Exception:
            pass
        return float('nan')

    def __call__(self, trial_id, result):
        if self._stopped:
            return True

        rel = self._extract_rel(result if isinstance(result, dict) else {})
        if not math.isfinite(rel):
            return False

        if self.best is None or rel > (float(self.best) + self.min_delta):
            self.best = float(rel)
            self.bad_evals = 0
            return False

        self.bad_evals += 1
        if self.max_drop > 0.0 and self.best is not None and rel < (float(self.best) - self.max_drop):
            print(f"[INFO] Early stop por queda de confiabilidade: atual={rel:.6f} best={self.best:.6f}")
            self._stopped = True
            return True

        if self.bad_evals >= self.patience_evals:
            print(f"[INFO] Early stop por estagnação de confiabilidade: atual={rel:.6f} best={self.best:.6f} evals_sem_melhora={self.bad_evals}")
            self._stopped = True
            return True

        return False

    def stop_all(self):
        return self._stopped

def build_parser():
    p = argparse.ArgumentParser()

    # compat (accepts legacy arguments without using them)
    p.add_argument("--use-model", type=str, default="")
    p.add_argument("--lr-end", type=float, default=None)
    p.add_argument("--clip-end", type=float, default=None)
    p.add_argument("--reset-ret-rms", type=int, default=0)
    p.add_argument("--vf-coef", type=float, default=1.0)

    # perf
    p.add_argument("--max-speed", type=int, default=1)
    p.add_argument("--n-envs", type=int, default=0, help="compat: total envs (opcional)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--envs-per-worker", type=int, default=0)
    p.add_argument("--num-gpus", type=float, default=1.0)

    # training
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=0)  # 0=auto
    p.add_argument("--minibatch-size", type=int, default=16384)
    p.add_argument("--n-epochs", type=int, default=4)

    p.add_argument("--lr-start", type=float, default=1.0e-4)
    p.add_argument("--clip-start", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--target-kl", type=float, default=0.03)

    # model
    p.add_argument("--policy-arch", type=str, default="384,384,384")
    p.add_argument("--activation", type=str, default="tanh",
                   choices=["tanh", "relu", "elu", "swish", "linear"])

    # model (per policy) — optional: if empty, inherits from --policy-arch/--activation
    p.add_argument("--policy-arch-anchor", type=str, default="",
                   help="Arquitetura da policy 'anchor' (ex: 128,128). Vazio => usa --policy-arch.")
    p.add_argument("--policy-arch-inter", type=str, default="",
                   help="Arquitetura da policy 'inter' (ex: 256,256,128). Vazio => usa --policy-arch.")
    p.add_argument("--activation-anchor", type=str, default="",
                   help="Ativação da policy 'anchor' (tanh|relu|elu|swish|linear). Vazio => usa --activation.")
    p.add_argument("--activation-inter", type=str, default="",
                   help="Ativação da policy 'inter' (tanh|relu|elu|swish|linear). Vazio => usa --activation.")

    # eval / ckpt
    p.add_argument("--eval-freq", type=int, default=200_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--checkpoint-freq-steps", type=int, default=1_000_000)
    p.add_argument("--run-name", type=str, default="marl_main")
    p.add_argument("--instance-tag", type=str, default="",
                   help="Tag para isolar logs/temporários ao rodar múltiplas instâncias (ex: A, B, exp1).")
    p.add_argument("--overwrite", type=int, default=0,
                   help="Se 1 e NÃO estiver em resume, apaga o diretório do experimento antes de iniciar (cuidado!).")
    p.add_argument("--run-dir", type=str, default="")  # auto => WORKDIR/saves
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--restore-path", type=str, default="")

    # env (subset)
    p.add_argument("--episode-horizon", type=int, default=2000)
    p.add_argument("--mask-actions", type=int, default=1)

    p.add_argument("--min-drones", type=int, default=10)
    p.add_argument("--max-drones", type=int, default=13)
    p.add_argument("--variable-drones", type=str, default="true")

    p.add_argument("--zonas-count", type=int, default=1)
    p.add_argument("--zona-w", type=int, default=15)
    p.add_argument("--zona-h", type=int, default=15)
    p.add_argument("--zona-single-fixed", type=int, default=1)
    p.add_argument("--p-sem-zona", type=float, default=0.0)

    p.add_argument("--obs-zone-map", type=int, default=1)
    p.add_argument("--zone-map-scale", type=int, default=4)
    p.add_argument("--obs-edge-cross", type=int, default=1)
    p.add_argument("--rel-step", type=float, default=2500.0,
                   help="Peso denso da confiabilidade por step (override de FIT_WEIGHTS_DEFAULT[REL_STEP]).")
    p.add_argument("--resume-from-best", type=int, default=1,
                   help="Se 1, em --resume/--use-model continue tenta restaurar automaticamente do melhor checkpoint salvo.")
    p.add_argument("--log-fit-terms", type=int, default=1,
                   help="Se 1, exporta os termos detalhados do fit para TensorBoard.")
    p.add_argument("--conn-quality-coef", type=float, default=200.0)
    p.add_argument("--hops-weight", type=float, default=20.0)
    p.add_argument("--layout-mix", type=float, default=800.0)
    p.add_argument("--line-align-w", type=float, default=200.0)
    p.add_argument("--disjoint-path-bonus", type=float, default=0.0)
    p.add_argument("--disjoint-gate-rff-start", type=float, default=0.97)
    p.add_argument("--disjoint-gate-rf-start", type=float, default=0.97)
    p.add_argument("--dirty-reset-prob-start", type=float, default=0.05)
    p.add_argument("--dirty-reset-prob-end", type=float, default=0.30)
    p.add_argument("--dirty-force-drone-in-zone-prob-start", type=float, default=0.05)
    p.add_argument("--dirty-force-drone-in-zone-prob-end", type=float, default=0.20)
    p.add_argument("--dirty-anneal-episodes", type=int, default=1000)
    p.add_argument("--stop-on-reliability-plateau", type=int, default=1,
                   help="Se 1, interrompe o treino quando a confiabilidade de avaliação estagna ou cai demais.")
    p.add_argument("--rel-patience-evals", type=int, default=8,
                   help="Número de avaliações sem melhora relevante antes do early-stop.")
    p.add_argument("--rel-min-delta", type=float, default=1e-3,
                   help="Melhora mínima de confiabilidade para resetar a paciência.")
    p.add_argument("--rel-max-drop", type=float, default=0.05,
                   help="Queda máxima aceitável a partir do melhor valor antes de parar o treino.")

    # best-model (evaluation-only)
    p.add_argument("--best-model-dir", type=str, default="",
                   help="Diretório do BEST model (checkpoint RLlib). Default: <repo>/ppo_best_model")
    p.add_argument("--best-metric", type=str, default="eval_terminal/reliability_terminal_mean",
                   help="Métrica de avaliação para escolher BEST (evaluation.custom_metrics).")
    p.add_argument("--best-mode", type=str, default="max", choices=["max", "min"],
                   help="max => maior é melhor, min => menor é melhor.")
    p.add_argument("--best-min-delta", type=float, default=1e-9, help="Melhora mínima para considerar novo BEST.")
    p.add_argument("--best-copy-mode", type=str, default="symlink", choices=["symlink", "copy"],
                   help="Como espelhar o checkpoint para ppo_best_model.")

    return p


def _is_valid_rllib_checkpoint_dir(d: Path) -> bool:
    try:
        d = Path(d).expanduser().resolve()
    except Exception:
        d = Path(d).expanduser()
    if not d.exists() or not d.is_dir():
        return False
    names = {p.name for p in d.iterdir()}
    if "algorithm_state.pkl" in names or "rllib_checkpoint.json" in names:
        return True
    if "algorithm_state.msgpack" in names or "algorithm_state.msgpck" in names:
        return True
    return any(re.fullmatch(r"checkpoint-\d+", n) for n in names)


def _resolve_best_checkpoint_dir(best_dir: Path) -> Path | None:
    best_dir = Path(best_dir).expanduser()
    if _is_valid_rllib_checkpoint_dir(best_dir):
        return best_dir.resolve()

    info_path = best_dir / "_BEST_INFO.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            src = info.get("checkpoint_source")
            if src:
                src_p = Path(src).expanduser()
                if _is_valid_rllib_checkpoint_dir(src_p):
                    return src_p.resolve()
        except Exception:
            pass

    try:
        candidates = sorted(
            [pp for pp in best_dir.rglob("*") if pp.is_dir() and _is_valid_rllib_checkpoint_dir(pp)],
            key=lambda pp: pp.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
    except Exception:
        pass
    return None


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Ignorando args desconhecidos:", unknown)

    # ------------------------------------------------------------
    # BEST model (evaluation-only): configure via env vars for Ray actors
    #
    # Project requirements:
    # - ALWAYS save by COPY (without symlinks)
    # - always keep the best checkpoint in a fixed directory (overwrite)
    #
    # Destination preference (when --best-model-dir is not provided):
    # 1) /ppo_best_model (if permission is available)
    #  2) <workdir>/ppo_best_model
    #  3) <repo>/ppo_best_model
    # ------------------------------------------------------------
    def _pick_writable_dir(candidates):
        for cand in candidates:
            try:
                p = Path(cand).expanduser()
                p.mkdir(parents=True, exist_ok=True)
                t = p / ".__writetest__"
                t.write_text("ok", encoding="utf-8")
                t.unlink(missing_ok=True)
                return p.resolve()
            except Exception:
                continue
        fb = (_BASE_DIR / "ppo_best_model").resolve()
        try:
            fb.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return fb

    if str(args.best_model_dir).strip():
        best_dir = _pick_writable_dir([Path(args.best_model_dir)])
    else:
        best_dir = _pick_writable_dir([
            Path("/ppo_best_model"),
            (_WORKDIR / "ppo_best_model"),
            (_BASE_DIR / "ppo_best_model"),
        ])

    print(f"[INFO] BEST model dir: {best_dir}")

    # If it is NOT resume/continue, clear the BEST directory to avoid inheriting an old best model and never "improving".
    if not bool(args.resume):
        try:
            if best_dir.exists():
                shutil.rmtree(best_dir, ignore_errors=True)
            best_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    os.environ["PPO_BEST_MODEL_DIR"] = str(best_dir)
    os.environ["PPO_BEST_METRIC"] = str(args.best_metric)
    os.environ["PPO_BEST_MODE"] = str(args.best_mode)
    os.environ["PPO_BEST_MIN_DELTA"] = str(args.best_min_delta)

    # Forces copying (without links) regardless of the CLI.
    os.environ["PPO_BEST_COPY_MODE"] = "copy"


    # ------------------------------------------------------------
    # Custom TensorBoard logdir (separate from Ray)
    # Logs under: <workdir>/tb_custom/<run>/<trial>
    custom_tb_root = Path(os.environ.get("PPO_CUSTOM_TB_ROOT", str((_WORKDIR / "tb_custom").resolve()))).resolve()
    try:
        custom_tb_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Não conseguiu criar custom_tb_root={custom_tb_root}: {type(e).__name__}: {e}")
    os.environ["PPO_CUSTOM_TB_ROOT"] = str(custom_tb_root)
    os.environ.setdefault("PPO_CUSTOM_TB_ENABLE", "1")

    # ------------------------------------------------------------
    # "continue" compat: allows --use-model continue as an alias for --resume 1
    # ------------------------------------------------------------
    use_model = (args.use_model or "").strip().lower()
    if use_model in ("continue", "resume"):
        args.resume = 1

    if bool(args.resume) and int(args.resume_from_best) != 0 and not str(args.restore_path).strip():
        best_ckpt = _resolve_best_checkpoint_dir(best_dir)
        if best_ckpt is not None:
            args.restore_path = str(best_ckpt)
            print(f"[INFO] Resume automático a partir do BEST checkpoint: {args.restore_path}")
        else:
            print(f"[WARN] Resume-from-best ativo, mas nenhum checkpoint válido foi encontrado em {best_dir}")

    torch.set_num_threads(1)

    max_speed = bool(args.max_speed)

    # ------------------------------------------------------------
    # Workers/envs: prioritizes explicit flags of the start.sh
    # ------------------------------------------------------------
    # Rules:
    # 1) If --num-workers > 0, do NOT override it. If --envs-per-worker was not provided,
    #    derive it from --n-envs (if provided) or use a safe default.
    # 2) If only --envs-per-worker was provided, derive --num-workers from --n-envs (if provided).
    # 3) If only --n-envs was provided, distribute automatically (capped by available CPU).
    # 4) Otherwise, auto.

    if args.num_workers > 0:
        # Honors the requested value (ex.: start.sh --num-workers 80).
        if args.envs_per_worker <= 0:
            if args.n_envs > 0:
                # tries to approximate the desired total, but RLlib requires >=1 env per runner
                args.envs_per_worker = max(1, int(math.ceil(args.n_envs / max(1, args.num_workers))))
            else:
                args.envs_per_worker = 2 if max_speed else 1
    elif args.envs_per_worker > 0:
        if args.num_workers <= 0:
            if args.n_envs > 0:
                args.num_workers = max(1, int(math.ceil(args.n_envs / max(1, args.envs_per_worker))))
            else:
                nw, _epw = _auto_workers(max_speed=max_speed)
                args.num_workers = nw
    elif args.n_envs > 0:
        # If you provided --n-envs and did not set workers, try to distribute
        cpu = os.cpu_count() or 32
        reserve = 8 if cpu >= 64 else 4
        avail = max(1, cpu - reserve)
        args.num_workers = min(args.n_envs, min(96, avail))
        args.envs_per_worker = max(1, int(math.ceil(args.n_envs / max(1, args.num_workers))))
    else:
        nw, epw = _auto_workers(max_speed=max_speed)
        if args.num_workers <= 0:
            args.num_workers = nw
        if args.envs_per_worker <= 0:
            args.envs_per_worker = epw

    total_envs = max(1, int(args.num_workers * args.envs_per_worker))
    if args.batch_size <= 0:
        args.batch_size = int(total_envs * args.n_steps)

    eval_interval = max(1, int(math.ceil(args.eval_freq / max(1, args.batch_size))))
    ckpt_freq = max(1, int(math.ceil(args.checkpoint_freq_steps / max(1, args.batch_size))))

    run_name_base = args.run_name.strip() or "marl_main"

    # To run 2 instances without interference:
    # - use --instance-tag A/B (or DRONE5G_INSTANCE_TAG)
    # - or leave it empty and the code will auto-disambiguate if the directory already exists.
    inst_tag = (args.instance_tag or os.environ.get("DRONE5G_INSTANCE_TAG", "")).strip()
    inst_tag = _sanitize_tag(inst_tag, max_len=24)

    run_name = f"{run_name_base}_{inst_tag}" if inst_tag else run_name_base

    if not args.run_dir or args.run_dir.strip() in ("./ray_results", "ray_results", "."):
        args.run_dir = str(_SAVES_ROOT)
    os.makedirs(args.run_dir, exist_ok=True)

    # Ray Tune (new versions) expects `storage_path` as a URI (ex: file:///...)
    from pathlib import Path as _Path
    storage_uri = _Path(args.run_dir).resolve().as_uri()

    # ------------------------------------------------------------
    # Experiment directory (Tune logs + checkpoints)
    # - does NOT delete by default (that would destroy logs from the other instance).
    # - If it exists and resume is disabled:
    # * --overwrite 1 => delete it and start from scratch
    # * otherwise => create a new run_name with a unique suffix
    # ------------------------------------------------------------
    exp_dir = _Path(args.run_dir).resolve() / run_name
    if exp_dir.exists() and not bool(args.resume):
        if bool(args.overwrite):
            print(f"[INFO] overwrite=1 (sem resume). Limpando exp_dir existente: {exp_dir}")
            shutil.rmtree(exp_dir, ignore_errors=True)
        else:
            suffix = time.strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
            run_name = f"{run_name}_{suffix}"
            exp_dir = _Path(args.run_dir).resolve() / run_name
            print(f"[INFO] exp_dir já existe. Criando novo experimento: {exp_dir}")

    if bool(args.resume) and (not exp_dir.exists()):
        print(f"[WARN] Resume/continue solicitado, mas exp_dir não existe: {exp_dir}")
        print("[WARN] Vou iniciar um treino novo (não há o que retomar).")

    exp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Ray temporary directory (Ray Core logs: session_latest/logs/)
    # Isolated per instance, to avoid mixing when running 2 processes.
    # ------------------------------------------------------------
    ray_tmp = setup_ray_tmp(run_name)

    fit_overrides = {
        "REL_STEP": float(args.rel_step),
        "CONN_QUALITY_COEF": float(args.conn_quality_coef),
        "HOPS_WEIGHT": float(args.hops_weight),
        "LAYOUT_MIX": float(args.layout_mix),
        "LINE_ALIGN_W": float(args.line_align_w),
        "DISJOINT_PATH_BONUS": float(args.disjoint_path_bonus),
        "DISJOINT_GATE_RFF_START": float(args.disjoint_gate_rff_start),
        "DISJOINT_GATE_RF_START": float(args.disjoint_gate_rf_start),
    }

    print("[INFO] Fit overrides:", json.dumps(fit_overrides, ensure_ascii=False))
    print("[INFO] Dirty curriculum:", json.dumps({
        "dirty_reset_prob_start": float(args.dirty_reset_prob_start),
        "dirty_reset_prob_end": float(args.dirty_reset_prob_end),
        "dirty_force_drone_in_zone_prob_start": float(args.dirty_force_drone_in_zone_prob_start),
        "dirty_force_drone_in_zone_prob_end": float(args.dirty_force_drone_in_zone_prob_end),
        "dirty_anneal_episodes": int(args.dirty_anneal_episodes),
    }, ensure_ascii=False))

    def env_creator(env_config):
        return Drone5GMAEnv(env_config)

    register_env("Drone5GMAEnv", env_creator)

    # gets spaces
    tmp_env = Drone5GMAEnv({
        "episode_horizon": args.episode_horizon,
        "mask_actions": int(args.mask_actions),
        "min_drones": args.min_drones,
        "max_drones": args.max_drones,
        "variable_drones": (str(args.variable_drones).lower() == "true"),
        "zonas_count": args.zonas_count,
        "zona_w": args.zona_w,
        "zona_h": args.zona_h,
        "zona_single_fixed": args.zona_single_fixed,
        "p_sem_zona": args.p_sem_zona,
        "obs_zone_map": args.obs_zone_map,
        "zone_map_scale": args.zone_map_scale,
        "obs_edge_cross": args.obs_edge_cross,
        "log_fit_terms": bool(int(args.log_fit_terms)),
        "dirty_reset_prob_start": float(args.dirty_reset_prob_start),
        "dirty_reset_prob_end": float(args.dirty_reset_prob_end),
        "dirty_force_drone_in_zone_prob_start": float(args.dirty_force_drone_in_zone_prob_start),
        "dirty_force_drone_in_zone_prob_end": float(args.dirty_force_drone_in_zone_prob_end),
        "dirty_anneal_episodes": int(args.dirty_anneal_episodes),
        "fit_weights": dict(fit_overrides),
    })
    anchor_obs_space = tmp_env.observation_space_dict["anchor_0"]
    anchor_act_space = tmp_env.action_space_dict["anchor_0"]
    inter_obs_space = tmp_env.observation_space_dict["inter_1"]
    inter_act_space = tmp_env.action_space_dict["inter_1"]
    tmp_env.close()
    ModelCatalog.register_custom_model("action_mask_cc_model", TorchActionMaskCCModel)

    def policy_mapping_fn(agent_id, *a, **k):
        return "anchor" if str(agent_id).startswith("anchor_") else "inter"

    def _parse_arch(s: str, fallback: str) -> list[int]:
        s = (s or '').strip()
        if not s:
            s = fallback
        return [int(x) for x in s.split(',') if x.strip()]

    def _parse_act(s: str, fallback: str) -> str:
        s = (s or '').strip()
        return s if s else fallback

    anchor_hiddens = _parse_arch(args.policy_arch_anchor, args.policy_arch)
    inter_hiddens = _parse_arch(args.policy_arch_inter, args.policy_arch)
    anchor_act = _parse_act(args.activation_anchor, args.activation)
    inter_act = _parse_act(args.activation_inter, args.activation)

    model_cfg_anchor_base = {
        'custom_model': 'action_mask_cc_model',
        'fcnet_hiddens': anchor_hiddens,
        'fcnet_activation': anchor_act,
        'vf_share_layers': False,
    }
    model_cfg_inter_base = {
        'custom_model': 'action_mask_cc_model',
        'fcnet_hiddens': inter_hiddens,
        'fcnet_activation': inter_act,
        'vf_share_layers': False,
    }

    # RLlib may flattenar Dict->Box at runtime; we pass dimensions via custom_model_config (per policy).
    anchor_cmc = {
        "obs_dim": int(anchor_obs_space["obs"].shape[0]),
        "state_dim": int(anchor_obs_space["state"].shape[0]),
        "mask_dim": int(anchor_obs_space["action_mask"].shape[0]),
    }
    inter_cmc = {
        "obs_dim": int(inter_obs_space["obs"].shape[0]),
        "state_dim": int(inter_obs_space["state"].shape[0]),
        "mask_dim": int(inter_obs_space["action_mask"].shape[0]),
    }

    model_cfg_anchor = dict(model_cfg_anchor_base)
    model_cfg_anchor["custom_model_config"] = anchor_cmc
    model_cfg_inter = dict(model_cfg_inter_base)
    model_cfg_inter["custom_model_config"] = inter_cmc
    env_config = {
        "episode_horizon": args.episode_horizon,
        "mask_actions": int(args.mask_actions),
        "min_drones": args.min_drones,
        "max_drones": args.max_drones,
        "variable_drones": (str(args.variable_drones).lower() == "true"),
        "zonas_count": args.zonas_count,
        "zona_w": args.zona_w,
        "zona_h": args.zona_h,
        "zona_single_fixed": args.zona_single_fixed,
        "p_sem_zona": args.p_sem_zona,
        "obs_zone_map": args.obs_zone_map,
        "zone_map_scale": args.zone_map_scale,
        "obs_edge_cross": args.obs_edge_cross,
        "log_fit_terms": bool(int(args.log_fit_terms)),
        "dirty_reset_prob_start": float(args.dirty_reset_prob_start),
        "dirty_reset_prob_end": float(args.dirty_reset_prob_end),
        "dirty_force_drone_in_zone_prob_start": float(args.dirty_force_drone_in_zone_prob_start),
        "dirty_force_drone_in_zone_prob_end": float(args.dirty_force_drone_in_zone_prob_end),
        "dirty_anneal_episodes": int(args.dirty_anneal_episodes),
        "fit_weights": dict(fit_overrides),
    }

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="Drone5GMAEnv", env_config=env_config, disable_env_checking=True)
        .framework("torch")
    )

    # GPU allocation (legacy API stack): use resources(num_gpus=...)
    config = config.resources(num_gpus=args.num_gpus)

    config = (
        config
        .env_runners(
            # (keep exactly what already exists in your file here)

            # `rollouts()` was removed in newer versions of the RLlib.
            # Equivalents:
            #   num_rollout_workers     -> num_env_runners
            #   num_envs_per_worker     -> num_envs_per_env_runner
            #   rollout_fragment_length -> rollout_fragment_length
            #   batch_mode              -> batch_mode
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.envs_per_worker,
            rollout_fragment_length=args.n_steps,
            batch_mode="truncate_episodes",
            create_local_env_runner=False,
            sample_timeout_s=300,
        )
        .training(
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            lr=args.lr_start,
            clip_param=args.clip_start,
            entropy_coeff=args.ent_coef,
            vf_loss_coeff=args.vf_coef,
            train_batch_size=args.batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.n_epochs,
            kl_coeff=0.0,
            use_kl_loss=True,
            kl_target=args.target_kl,

        )
        .callbacks(TerminalMetricsCallbacks)
        .evaluation(
            evaluation_interval=eval_interval,
            evaluation_duration=args.n_eval_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .multi_agent(
            policies={
                "anchor": (None, anchor_obs_space, anchor_act_space, {"model": model_cfg_anchor}),
                "inter": (None, inter_obs_space, inter_act_space, {"model": model_cfg_inter}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["anchor", "inter"],
        )
        .debugging(log_level="INFO")
    )

    # ------------------------------------------------------------
    # Ray init: allows honoring --num-workers even in cgroup/containers
    # (Ray resource override).
    # You can force it manually via env var: RAY_NUM_CPUS=...
    # ------------------------------------------------------------
    ray_num_cpus = None
    try:
        env_nc = os.environ.get('RAY_NUM_CPUS', '').strip()
        if env_nc:
            ray_num_cpus = int(float(env_nc))
    except Exception:
        ray_num_cpus = None
    if ray_num_cpus is None and args.num_workers > 0:
        # +2 for driver/learner overhead
        ray_num_cpus = max(os.cpu_count() or 1, int(args.num_workers) + 2)

    init_kwargs = dict(ignore_reinit_error=True, include_dashboard=False,
                       _temp_dir=str(ray_tmp), _node_ip_address='127.0.0.1')
    if ray_num_cpus is not None:
        init_kwargs['num_cpus'] = ray_num_cpus
    try:
        init_kwargs['num_gpus'] = float(args.num_gpus)
    except Exception:
        pass
    ray.init(**init_kwargs)
    ray.cluster_resources()
    # Ensures that callbacks is not lost during serialization of the config
    cfg = config.to_dict()
    cfg["callbacks"] = TerminalMetricsCallbacks

    stop_cfg = {"timesteps_total": int(args.timesteps)}
    if int(args.stop_on_reliability_plateau) != 0:
        stop_cfg = ReliabilityPlateauStopper(
            patience_evals=int(args.rel_patience_evals),
            min_delta=float(args.rel_min_delta),
            max_drop=float(args.rel_max_drop),
        )

    tune_kwargs = dict(
        name=run_name,
        config=cfg,
        stop=stop_cfg,
        storage_path=storage_uri,
        checkpoint_freq=ckpt_freq,
        checkpoint_at_end=True,
        verbose=1,
        callbacks=[TBXLoggerCallback()],
        # Keeps the same name/directory of the trial (only 1 trial): makes TensorBoard easier
        trial_name_creator=lambda t: "trial",
        trial_dirname_creator=lambda t: "trial",
    )
    if bool(args.resume):
        # Ray Tune "resume" resumes an existing experiment (based on storage_path + name).
        # use "AUTO" for Tune to decide whether to resume or start from scratch (if no state exists).
        tune_kwargs["resume"] = "AUTO"

        if args.restore_path:
            rp = Path(args.restore_path)
            if not rp.exists():
                print(f"[WARN] restore-path não encontrado: {args.restore_path}")
                print("[WARN] Ignorando restore explícito e tentando resume automático.")
            else:
                # restore points to a specific checkpoint (dir checkpoint_xxxxx).
                tune_kwargs["restore"] = str(rp.resolve())
                print(f"[INFO] Tune restore checkpoint: {tune_kwargs['restore']}")

        # Warning: if args.timesteps <= the already trained timesteps_total, Tune may terminate immediately.
        print(f"[INFO] Stop target timesteps_total = {int(args.timesteps)} (aumente --timesteps para continuar mais).")

    tune.run("PPO", **tune_kwargs)


if __name__ == "__main__":
    main()
