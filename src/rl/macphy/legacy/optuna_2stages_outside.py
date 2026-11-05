#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optuna "por fora" em 2 estágios (Stage1 screening + Stage2 confirmação)
Chamando diretamente o 5GMLMA (não usa start.sh) e variando:
- Hiperparâmetros PPO
- Arquitetura da rede (policy-arch) e ativação (activation)

Restrições:
- zone_violations_terminal_mean <= OPTUNA_ZONE_MAX (default 0.0)
- throughput_terminal_mean <= OPTUNA_TP_MAX_MBPS (default 9600 Mbps)
- reliability_terminal_mean >= OPTUNA_REL_MIN (default 0.0 => desliga)

Saídas:
- <WORKDIR>/optuna_outside/best_commands.sh
- <WORKDIR>/optuna_outside/summary.json
- logs: <WORKDIR>/optuna_outside/logs/stage1|stage2/*.log

LEGACY WARNING: this file targets an older Ray/RLlib PPO trainer and progress.csv output.
It is preserved for provenance and is not compatible with the native HAPPO trainer.
"""

import os
import json
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import optuna


# =========================
# CONFIG DO PROJETO
# =========================
PROJECT = Path(os.environ.get("OPTUNA_PROJECT", str(Path(__file__).resolve().parent.parent))).expanduser().resolve()

# AQUI: aponte direto para o 5GMLMA (o “último código”)
DEFAULT_5GMLMA = PROJECT / "5GMLMA_MA.py"
TRAIN_SCRIPT = Path(os.environ.get("OPTUNA_TRAIN_SCRIPT", str(DEFAULT_5GMLMA))).resolve()

# workdir padrão do seu projeto
WORKDIR = Path(os.environ.get("DRONE5G_WORKDIR", str(PROJECT / "marl_workdir"))).resolve()
SAVES_DIR = WORKDIR / "saves"

OUT_HPO = WORKDIR / "optuna_outside"
OUT_HPO.mkdir(parents=True, exist_ok=True)
COMMANDS_FILE = OUT_HPO / "best_commands.sh"
SUMMARY_FILE = OUT_HPO / "summary.json"

# Python do Optuna (este script)
PY_OPTUNA = os.sys.executable

# Python do TREINO (padrão: venv do projeto)
DEFAULT_TRAIN_PY = PROJECT / ".venv" / "bin" / "python"
import os
from pathlib import Path

_train_py_str = os.environ.get("OPTUNA_TRAIN_PYTHON", "").strip()
if not _train_py_str:
    _train_py_str = str(PROJECT / ".venv" / "bin" / "python")

# NÃO use .resolve() aqui (senão vira /usr/bin/python3.12)
PY_TRAIN = Path(_train_py_str).expanduser()
PY_TRAIN = Path(os.path.abspath(str(PY_TRAIN)))  # ok: abs sem resolver symlink

print("[INFO] PY_TRAIN (venv path):", PY_TRAIN)

import subprocess, sys

out = subprocess.check_output(
    [str(PY_TRAIN), "-c", "import sys; print(sys.prefix); print(sys.base_prefix)"],
    text=True
).strip().splitlines()

prefix, base_prefix = out[0], out[1]
print("[INFO] train sys.prefix     =", prefix)
print("[INFO] train sys.base_prefix=", base_prefix)

if prefix == base_prefix:
    raise RuntimeError(
        "PY_TRAIN não está usando o venv (sys.prefix == sys.base_prefix). "
        "Verifique OPTUNA_TRAIN_PYTHON apontando para .../.venv/bin/python."
    )

# =========================
# RESTRIÇÕES / MÉTRICAS
# =========================
ZONE_VIOL_MAX = float(os.environ.get("OPTUNA_ZONE_MAX", "0.0"))
THROUGHPUT_MAX_MBPS = float(os.environ.get("OPTUNA_TP_MAX_MBPS", "9600.0"))
RELIABILITY_MIN = float(os.environ.get("OPTUNA_REL_MIN", "0.0"))  # 0.0 => não impõe mínimo

# “Hints” (se existirem exatamente assim no progress.csv, usa direto; senão autodetecta)
COL_REL_HINT = os.environ.get("OPTUNA_COL_REL", "eval_terminal/reliability_terminal_mean")
COL_ZONE_HINT = os.environ.get("OPTUNA_COL_ZONE", "eval_terminal/zone_violations_terminal_mean")
COL_TP_HINT = os.environ.get("OPTUNA_COL_TP", "eval_terminal/throughput_terminal_mean")


# =========================
# STAGES
# =========================
STAGE1_TRIALS = int(os.environ.get("OPTUNA_STAGE1_TRIALS", "30"))
STAGE1_TIMESTEPS = int(os.environ.get("OPTUNA_STAGE1_TIMESTEPS", "5000000"))
STAGE1_TOPK = int(os.environ.get("OPTUNA_STAGE1_TOPK", "5"))

STAGE2_TIMESTEPS = int(os.environ.get("OPTUNA_STAGE2_TIMESTEPS", "20000000"))
STAGE2_RUNS_PER_CANDIDATE = int(os.environ.get("OPTUNA_STAGE2_RUNS", "3"))  # repeats “tipo seeds”


# =========================
# EXECUÇÃO FIXA (para comparabilidade)
# =========================
NUM_GPUS = float(os.environ.get("OPTUNA_NUM_GPUS", "1.0"))
NUM_WORKERS = int(os.environ.get("OPTUNA_NUM_WORKERS", "80"))
ENVS_PER_WORKER = int(os.environ.get("OPTUNA_ENVS_PER_WORKER", "1"))
N_STEPS_FIXED = int(os.environ.get("OPTUNA_N_STEPS", "512"))

EVAL_FREQ = int(os.environ.get("OPTUNA_EVAL_FREQ", "1000000"))
N_EVAL_EPISODES = int(os.environ.get("OPTUNA_N_EVAL_EPISODES", "10"))
CKPT_FREQ_STEPS = int(os.environ.get("OPTUNA_CKPT_FREQ_STEPS", "2000000"))

FAIL_TAIL_LINES = int(os.environ.get("OPTUNA_FAIL_TAIL_LINES", "60"))

# Penalidades (ajuste se quiser)
PEN_ZONE = float(os.environ.get("OPTUNA_PEN_ZONE", "50.0"))
PEN_TP = float(os.environ.get("OPTUNA_PEN_TP", "0.001"))


# =========================
# SEARCH SPACE (inclui rede neural)
# =========================
# =========================
# SEARCH SPACE (inclui rede neural por policy)
# =========================
# Dica: anchor costuma funcionar bem com redes menores; inter (movimento + toggle) tende a precisar de mais capacidade.

ARCH_ANCHOR_CHOICES = os.environ.get(
    "OPTUNA_ARCH_ANCHOR_CHOICES",
    "128,128|256,256"
).split("|")

ARCH_INTER_CHOICES = os.environ.get(
    "OPTUNA_ARCH_INTER_CHOICES",
    "256,256|512,512,256"
).split("|")

ACT_ANCHOR_CHOICES = os.environ.get("OPTUNA_ACT_ANCHOR_CHOICES", "tanh|relu").split("|")
ACT_INTER_CHOICES = os.environ.get("OPTUNA_ACT_INTER_CHOICES", "relu|swish").split("|")


# =========================
# UTIL
# =========================

@dataclass
class RunResult:
    run_name: str
    instance_tag: str
    trial_dir: Path
    score: float
    ok_constraints: bool
    best_row: Dict[str, Any]
    rc: int
    log_path: Path


def _safe_tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(não consegui ler log {path}: {e})"


def _train_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("RAY_DEDUP_LOGS", "1")
    return env


def run_one(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=_train_env())
    return int(p.returncode)


def run_dir_from_run(run_name: str) -> Path:
    return SAVES_DIR / run_name


def find_trial_dir_with_progress(run_name: str) -> Path:
    """
    Para o seu 5GMLMA, normalmente é:
      <run_dir>/<run_name>/trial/progress.csv
    Mas aqui buscamos de forma robusta.
    """
    base = run_dir_from_run(run_name)
    cand = base / "trial" / "progress.csv"
    if cand.exists():
        return cand.parent

    if not base.exists():
        raise FileNotFoundError(f"run_dir não existe: {base}")

    hits = list(base.rglob("progress.csv"))
    if not hits:
        raise FileNotFoundError(f"Não encontrei progress.csv dentro de: {base}")

    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0].parent


def read_best_constrained_from_progress(trial_dir: Path) -> Tuple[float, Dict[str, Any], bool]:
    """
    Se existe ponto viável -> score = max(rel) nos viáveis.
    Senão -> score penalizado:
        score = rel - PEN_ZONE*zone - PEN_TP*excesso_throughput
    Também converte throughput para Mbps se parecer bps (mediana > 1e6).
    """
    import pandas as pd
    import numpy as np

    prog = trial_dir / "progress.csv"
    if not prog.exists():
        raise FileNotFoundError(f"progress.csv não encontrado em {trial_dir}")

    df = pd.read_csv(prog)

    def find_exact(name: str) -> Optional[str]:
        return name if name in df.columns else None

    def find_col(pred) -> Optional[str]:
        cands = [c for c in df.columns if pred(c)]
        return cands[0] if cands else None

    rel_col = find_exact(COL_REL_HINT) or find_col(lambda c: "reliability_terminal" in c.lower() and c.lower().endswith("_mean"))
    zone_col = find_exact(COL_ZONE_HINT) or find_col(lambda c: ("zone_viol" in c.lower() or "zone_violation" in c.lower()) and c.lower().endswith("_mean"))
    tp_col = find_exact(COL_TP_HINT) or find_col(lambda c: "throughput_terminal" in c.lower() and c.lower().endswith("_mean"))

    if not rel_col or not zone_col or not tp_col:
        raise KeyError(
            "Não achei colunas necessárias.\n"
            f"rel_col={rel_col}, zone_col={zone_col}, tp_col={tp_col}\n"
            f"cols_exemplo={list(df.columns)[:60]}"
        )

    x = df[[rel_col, zone_col, tp_col]].dropna().copy()
    if x.shape[0] == 0:
        return -1e9, {"_error": "no_rows_after_dropna"}, False

    rel = x[rel_col].astype(float)
    zone = x[zone_col].astype(float)
    tp_raw = x[tp_col].astype(float)

    tp_med = float(tp_raw.median())
    if tp_med > 1e6:
        tp_mbps = tp_raw / 1e6
        tp_unit = "bps->Mbps"
    else:
        tp_mbps = tp_raw
        tp_unit = "Mbps"

    ok = (
        (zone <= ZONE_VIOL_MAX + 1e-9) &
        (tp_mbps <= THROUGHPUT_MAX_MBPS + 1e-6) &
        (rel >= RELIABILITY_MIN - 1e-12)
    )

    excess = np.maximum(0.0, tp_mbps - THROUGHPUT_MAX_MBPS)
    score_all = rel - PEN_ZONE * zone - PEN_TP * excess

    # melhor (penalizado) para debug
    i_pen = int(score_all.fillna(-1e18).idxmax())
    best_pen_row = x.loc[i_pen].to_dict()

    meta = {
        "_used_cols": {"rel": rel_col, "zone": zone_col, "tp": tp_col},
        "_tp_unit_assumed": tp_unit,
        "_tp_median_raw": tp_med,
        "_constraints": {
            "ZONE_VIOL_MAX": ZONE_VIOL_MAX,
            "THROUGHPUT_MAX_MBPS": THROUGHPUT_MAX_MBPS,
            "RELIABILITY_MIN": RELIABILITY_MIN,
        },
        "_penalties": {"PEN_ZONE": PEN_ZONE, "PEN_TP": PEN_TP},
    }

    if bool(ok.any()):
        xv = x.loc[ok].copy()
        j = int(xv[rel_col].fillna(-1).idxmax())
        best_row = xv.loc[j].to_dict()
        best_row.update(meta)
        return float(best_row[rel_col]), best_row, True

    best_pen_row.update(meta)
    return float(score_all.loc[i_pen]), best_pen_row, False


def build_cmd(run_name_base: str, instance_tag: str, timesteps: int, hp: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    O 5GMLMA gera run_name = f"{run_name_base}_{instance_tag}" (se instance_tag != "")
    """
    inst = instance_tag.strip()
    run_name = f"{run_name_base}_{inst}" if inst else run_name_base

    cmd = [
        str(PY_TRAIN), str(TRAIN_SCRIPT),

        "--run-name", run_name_base,
        "--instance-tag", inst,

        "--resume", "0",
        "--overwrite", "1",

        "--run-dir", str(SAVES_DIR),

        "--num-gpus", str(NUM_GPUS),
        "--num-workers", str(NUM_WORKERS),
        "--envs-per-worker", str(ENVS_PER_WORKER),

        "--n-steps", str(N_STEPS_FIXED),
        "--batch-size", "0",

        "--eval-freq", str(EVAL_FREQ),
        "--n-eval-episodes", str(N_EVAL_EPISODES),
        "--checkpoint-freq-steps", str(CKPT_FREQ_STEPS),

        "--timesteps", str(timesteps),

        # PPO HP
        "--lr-start", str(hp["lr_start"]),
        "--clip-start", str(hp["clip_start"]),
        "--ent-coef", str(hp["ent_coef"]),
        "--gamma", str(hp["gamma"]),
        "--gae-lambda", str(hp["gae_lambda"]),
        "--vf-coef", str(hp["vf_coef"]),
        "--n-epochs", str(hp["n_epochs"]),
        "--minibatch-size", str(hp["minibatch_size"]),

# Rede neural (por policy)
# Mantemos também --policy-arch/--activation como fallback global (setamos = inter) por compatibilidade.
"--policy-arch", str(hp["policy_arch_inter"]),
"--activation", str(hp["activation_inter"]),
"--policy-arch-anchor", str(hp["policy_arch_anchor"]),
"--activation-anchor", str(hp["activation_anchor"]),
"--policy-arch-inter", str(hp["policy_arch_inter"]),
"--activation-inter", str(hp["activation_inter"]),
    ]
    return run_name, cmd


# =========================
# OPTUNA
# =========================

def suggest_hp(trial: optuna.Trial) -> Dict[str, Any]:
    hp = {
        # PPO
        "lr_start": trial.suggest_float("lr_start", 1e-5, 3e-4, log=True),
        "clip_start": trial.suggest_float("clip_start", 0.05, 0.30),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.995, 0.99995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.98),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.2),
        "n_epochs": trial.suggest_categorical("n_epochs", [10, 15, 20]),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [8192, 16384, 32768]),

# Rede neural (por policy)
"policy_arch_anchor": trial.suggest_categorical("policy_arch_anchor", ARCH_ANCHOR_CHOICES),
"activation_anchor": trial.suggest_categorical("activation_anchor", ACT_ANCHOR_CHOICES),
"policy_arch_inter": trial.suggest_categorical("policy_arch_inter", ARCH_INTER_CHOICES),
"activation_inter": trial.suggest_categorical("activation_inter", ACT_INTER_CHOICES),
    }
    return hp


def objective_stage1(trial: optuna.Trial) -> float:
    hp = suggest_hp(trial)
    instance_tag = f"S1T{trial.number:04d}"  # <=24 chars, único
    run_name_base = "optuna_s1"

    run_name, cmd = build_cmd(run_name_base, instance_tag, STAGE1_TIMESTEPS, hp)
    log_path = OUT_HPO / "logs" / "stage1" / f"{instance_tag}.log"

    rc = run_one(cmd, PROJECT, log_path)
    trial.set_user_attr("run_name", run_name)
    trial.set_user_attr("log_path", str(log_path))
    trial.set_user_attr("rc", rc)
    trial.set_user_attr("hp", hp)

    if rc != 0:
        print(f"[FAIL][Stage1 {instance_tag}] rc={rc}\n--- tail ---\n{_safe_tail(log_path, FAIL_TAIL_LINES)}\n-----------")
        return -1e9

    try:
        tdir = find_trial_dir_with_progress(run_name)
        score, row, okc = read_best_constrained_from_progress(tdir)
        trial.set_user_attr("trial_dir", str(tdir))
        trial.set_user_attr("constraints_ok", bool(okc))
        trial.set_user_attr("best_row", row)
        return float(score)
    except Exception as e:
        print(f"[FAIL][Stage1 {instance_tag}] read_error={e}\n--- tail ---\n{_safe_tail(log_path, FAIL_TAIL_LINES)}\n-----------")
        trial.set_user_attr("read_error", repr(e))
        return -1e9


def run_stage2_for_candidate(rank: int, hp: Dict[str, Any]) -> Tuple[float, List[RunResult]]:
    results: List[RunResult] = []
    scores: List[float] = []
    run_name_base = "optuna_s2"

    for si in range(STAGE2_RUNS_PER_CANDIDATE):
        instance_tag = f"S2R{rank}R{si}"
        run_name, cmd = build_cmd(run_name_base, instance_tag, STAGE2_TIMESTEPS, hp)
        log_path = OUT_HPO / "logs" / "stage2" / f"{instance_tag}.log"

        rc = run_one(cmd, PROJECT, log_path)
        if rc != 0:
            print(f"[FAIL][Stage2 {instance_tag}] rc={rc}\n--- tail ---\n{_safe_tail(log_path, FAIL_TAIL_LINES)}\n-----------")
            score = -1e9
            okc = False
            row = {"_error": f"rc={rc}"}
            tdir = run_dir_from_run(run_name)
        else:
            try:
                tdir = find_trial_dir_with_progress(run_name)
                score, row, okc = read_best_constrained_from_progress(tdir)
            except Exception as e:
                print(f"[FAIL][Stage2 {instance_tag}] read_error={e}\n--- tail ---\n{_safe_tail(log_path, FAIL_TAIL_LINES)}\n-----------")
                score = -1e9
                okc = False
                row = {"_error": repr(e)}
                tdir = run_dir_from_run(run_name)

        scores.append(float(score))
        results.append(RunResult(
            run_name=run_name,
            instance_tag=instance_tag,
            trial_dir=Path(tdir),
            score=float(score),
            ok_constraints=bool(okc),
            best_row=row if isinstance(row, dict) else {},
            rc=int(rc),
            log_path=log_path,
        ))

    mean_score = sum(scores) / max(1, len(scores))
    return float(mean_score), results


def render_command(cmd: List[str]) -> str:
    parts = []
    for x in cmd:
        if " " in x or "\t" in x:
            parts.append("'" + x.replace("'", "'\"'\"'") + "'")
        else:
            parts.append(x)
    return " \\\n  ".join(parts)


def write_commands_file(best_hp: Dict[str, Any], best_mean_score: float, out_path: Path):
    final_timesteps = int(os.environ.get("OPTUNA_FINAL_TIMESTEPS", "50000000"))
    run_name_base = "marl_main"
    instance_tag = "BEST_OPTUNA"

    run_name, cmd = build_cmd(run_name_base, instance_tag, final_timesteps, best_hp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n\n")
        f.write("# ==============================\n")
        f.write("# BEST HP + REDE encontrado pelo Optuna\n")
        f.write("# ==============================\n")
        f.write(f"# Stage2 mean score: {best_mean_score:.6f}\n")
        f.write(f"# Constraints: zone<= {ZONE_VIOL_MAX}, tp<= {THROUGHPUT_MAX_MBPS} Mbps, rel>= {RELIABILITY_MIN}\n")
        f.write(f"# Train python: {PY_TRAIN}\n")
        f.write(f"# Train script: {TRAIN_SCRIPT}\n\n")
        f.write(render_command(cmd) + "\n")

    try:
        os.chmod(out_path, 0o755)
    except Exception:
        pass


def main():
    print("[INFO] PROJECT:", PROJECT)
    print("[INFO] TRAIN_SCRIPT:", TRAIN_SCRIPT)
    print("[INFO] PY_OPTUNA:", PY_OPTUNA)
    print("[INFO] PY_TRAIN :", PY_TRAIN)
    print("[INFO] WORKDIR:", WORKDIR)
    print("[INFO] SAVES  :", SAVES_DIR)
    print("[INFO] Constraints:",
          f"zone<= {ZONE_VIOL_MAX}, tp<= {THROUGHPUT_MAX_MBPS} Mbps, rel>= {RELIABILITY_MIN}")
    print("[INFO] Hints:", COL_REL_HINT, COL_ZONE_HINT, COL_TP_HINT)
    print("[INFO] Stage1:", STAGE1_TRIALS, "trials,", STAGE1_TIMESTEPS, "timesteps, topK=", STAGE1_TOPK)
    print("[INFO] Stage2: timesteps=", STAGE2_TIMESTEPS, "runs_per_candidate=", STAGE2_RUNS_PER_CANDIDATE)
    print("[INFO] Fixed exec:", f"workers={NUM_WORKERS} epw={ENVS_PER_WORKER} n_steps={N_STEPS_FIXED} gpus={NUM_GPUS}")
    print("[INFO] Arch choices (anchor):", ARCH_ANCHOR_CHOICES)
    print("[INFO] Arch choices (inter) :", ARCH_INTER_CHOICES)
    print("[INFO] Act choices  (anchor):", ACT_ANCHOR_CHOICES)
    print("[INFO] Act choices  (inter) :", ACT_INTER_CHOICES)

    # Stage 1
    study1 = optuna.create_study(direction="maximize")
    study1.optimize(objective_stage1, n_trials=STAGE1_TRIALS, n_jobs=int(os.environ.get("OPTUNA_N_JOBS", "1")))

    # TopK
    trials_sorted = sorted(
        [t for t in study1.trials if t.value is not None],
        key=lambda t: float(t.value),
        reverse=True
    )
    topk_trials = trials_sorted[:max(1, STAGE1_TOPK)]

    topk = []
    for rank, t in enumerate(topk_trials, 1):
        topk.append({
            "rank": rank,
            "stage1_score": float(t.value),
            "params": dict(t.params),
            "run_name": t.user_attrs.get("run_name"),
            "trial_dir": t.user_attrs.get("trial_dir"),
            "constraints_ok": t.user_attrs.get("constraints_ok"),
            "rc": t.user_attrs.get("rc"),
            "log_path": t.user_attrs.get("log_path"),
            "best_row": t.user_attrs.get("best_row"),
            "read_error": t.user_attrs.get("read_error"),
        })

    # Stage 2
    stage2_results = []
    best_mean = -1e18
    best_hp = None
    best_detail = None

    for item in topk:
        rank = item["rank"]
        hp = item["params"]

        mean_score, runs = run_stage2_for_candidate(rank, hp)
        entry = {
            "rank": rank,
            "stage1_score": item["stage1_score"],
            "stage2_mean_score": mean_score,
            "hp": hp,
            "runs": [{
                "run_name": r.run_name,
                "instance_tag": r.instance_tag,
                "trial_dir": str(r.trial_dir),
                "score": r.score,
                "ok_constraints": r.ok_constraints,
                "rc": r.rc,
                "log_path": str(r.log_path),
                "best_row": r.best_row,
            } for r in runs]
        }
        stage2_results.append(entry)

        if mean_score > best_mean:
            best_mean = mean_score
            best_hp = hp
            best_detail = entry

    if best_hp is None:
        raise RuntimeError("Nenhum candidato retornou score válido no Stage 2 (verifique logs).")

    write_commands_file(best_hp, best_mean_score=best_mean, out_path=COMMANDS_FILE)

    summary = {
        "project": str(PROJECT),
        "train_script": str(TRAIN_SCRIPT),
        "py_optuna": str(PY_OPTUNA),
        "py_train": str(PY_TRAIN),
        "workdir": str(WORKDIR),
        "saves_dir": str(SAVES_DIR),
        "constraints": {
            "zone_viol_max": ZONE_VIOL_MAX,
            "throughput_max_mbps": THROUGHPUT_MAX_MBPS,
            "reliability_min": RELIABILITY_MIN,
        },
        "col_hints": {"rel": COL_REL_HINT, "zone": COL_ZONE_HINT, "tp": COL_TP_HINT},
"search_space": {
    "arch_anchor_choices": ARCH_ANCHOR_CHOICES,
    "arch_inter_choices": ARCH_INTER_CHOICES,
    "act_anchor_choices": ACT_ANCHOR_CHOICES,
    "act_inter_choices": ACT_INTER_CHOICES,
},
        "fixed_exec": {
            "num_gpus": NUM_GPUS,
            "num_workers": NUM_WORKERS,
            "envs_per_worker": ENVS_PER_WORKER,
            "n_steps": N_STEPS_FIXED,
            "eval_freq": EVAL_FREQ,
            "n_eval_episodes": N_EVAL_EPISODES,
            "checkpoint_freq_steps": CKPT_FREQ_STEPS,
        },
        "penalties": {"PEN_ZONE": PEN_ZONE, "PEN_TP": PEN_TP},
        "stage1": {
            "trials": STAGE1_TRIALS,
            "timesteps": STAGE1_TIMESTEPS,
            "best_value": float(study1.best_value) if study1.best_value is not None else None,
            "best_params": dict(study1.best_trial.params) if study1.best_trial is not None else None,
            "topk": topk,
        },
        "stage2": {
            "timesteps": STAGE2_TIMESTEPS,
            "runs_per_candidate": STAGE2_RUNS_PER_CANDIDATE,
            "results": stage2_results,
        },
        "best": {
            "stage2_mean_score": best_mean,
            "hp": best_hp,
            "detail": best_detail,
        },
        "outputs": {
            "commands_file": str(COMMANDS_FILE),
            "summary_file": str(SUMMARY_FILE),
            "logs_dir": str(OUT_HPO / "logs"),
        }
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Concluído.")
    print(" - Commands:", COMMANDS_FILE)
    print(" - Summary :", SUMMARY_FILE)
    print(" - Best Stage2 mean score:", best_mean)
    print(" - Best HP:", best_hp)


if __name__ == "__main__":
    main()
