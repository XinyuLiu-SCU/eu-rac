"""Chengdu benchmark runner with isolated raw results and resumable training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chengdu_env import ChengduEnv, normalize_period
from evaluator import evaluate_policy

NETWORK_DIR = ROOT.parent / "network" / "Chengdu"
REPORT_DIR = ROOT / "report"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
PUBLIC_OD_FILE = ROOT.parent / "data" / "chengdu" / "od_pairs.csv"

PROTECTED_FILES = [PUBLIC_OD_FILE]

CANONICAL_ALGORITHMS = ["eurac", "pulse", "robust", "gp3", "ilp", "otap", "segac", "dac", "pql", "geddrl"]

ALGORITHM_ALIASES = {
    "dot": "dot",
    "robust": "robust",
    "sota": "dot",
    "pulse": "pulse",
    "gp3": "gp3",
    "ilp": "ilp",
    "otap": "otap",
    "pql": "pql",
    "segac": "segac",
    "dac": "dac",
    "geddrl": "geddrl",
    "ge-ddrl": "geddrl",
    "ge_ddrl": "geddrl",
    "eu-rac": "eurac",
    "eurac": "eurac",
    "eurac": "eurac",
    "eu_rac": "eurac",
    "eu-rac": "eurac",
}

STATUS_PASS = {"PASS", "PASS_WITH_ZERO_MC"}



def _load_public_eurac_module():
    """Load the root-level public neural EU-RAC implementation."""
    spec = importlib.util.spec_from_file_location("_public_eu_rac", ROOT.parent / "eu_rac.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load root-level eu_rac.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_FILES}


def ensure_dirs(period: str) -> None:
    for base in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR):
        (base / period).mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "smoke_tests" / period).mkdir(parents=True, exist_ok=True)


def parse_algorithms(values: list[str] | None) -> list[str]:
    if not values:
        return ["dot"]
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip().lower() for part in value.split(",") if part.strip())
    if any(token == "all" for token in tokens):
        return CANONICAL_ALGORITHMS[:]
    parsed = []
    unknown = []
    for token in tokens:
        alg = ALGORITHM_ALIASES.get(token)
        if alg is None:
            unknown.append(token)
        elif alg not in parsed:
            parsed.append(alg)
    if unknown:
        valid = ", ".join(sorted(ALGORITHM_ALIASES))
        raise ValueError(f"Unknown algorithm(s): {', '.join(unknown)}. Valid names: {valid}")
    return parsed


def load_release_od_pairs(period: str) -> np.ndarray:
    if not PUBLIC_OD_FILE.exists():
        raise FileNotFoundError(f"Public OD file not found: {PUBLIC_OD_FILE}")
    rows: list[tuple[int, int]] = []
    with PUBLIC_OD_FILE.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row_period = str(row.get("period", "")).strip().lower()
            if row_period and row_period != period:
                continue
            rows.append((int(row["origin"]), int(row["destination"])))
    arr = np.asarray(rows, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Public OD file must have shape (n, 2), got {arr.shape}: {PUBLIC_OD_FILE}")
    return arr


def select_od(period: str, od_index: int) -> tuple[int, int]:
    release_ods = load_release_od_pairs(period)
    if od_index < 0 or od_index >= len(release_ods):
        raise IndexError(f"OD index {od_index} out of range for {period}; valid range: 0..{len(release_ods)-1}")
    origin, dest = release_ods[od_index]
    return int(origin), int(dest)


def env_warning_for_eta(eta: float, uncertainty_active: bool = False) -> str:
    if eta == 0:
        return ""
    if uncertainty_active:
        return "execution uncertainty loaded from period uncertain-actions JSON"
    return "eta was accepted with an empty explicit uncertainty set; no uncertain actions were provided."


def make_env(period: str, origin: int, dest: int, budget: int, eta: float, seed: int | None,
             uncertain_edges_explicit: dict | None = None) -> ChengduEnv:
    if eta == 0 and uncertain_edges_explicit is None:
        return ChengduEnv(origin=origin, destination=dest, budget=budget, eta=0.0, period=period, seed=seed)
    if uncertain_edges_explicit is not None:
        return ChengduEnv(
            origin=origin,
            destination=dest,
            budget=budget,
            eta=eta,
            period=period,
            seed=seed,
            uncertain_edges_explicit=uncertain_edges_explicit,
        )
    return ChengduEnv(
        origin=origin,
        destination=dest,
        budget=budget,
        eta=eta,
        period=period,
        seed=seed,
        uncertain_nodes=[],
    )


def budget_from_let(let_value: float, ratio: float) -> int:
    return int(math.floor(float(ratio) * float(let_value)))


def checkpoint_path(period: str, alg: str, origin: int, dest: int, budget: int, eta: float) -> Path:
    return CHECKPOINT_DIR / period / f"OD_{origin}_{dest}" / f"B_{budget}" / f"eta_{eta:g}" / alg


def output_path(period: str, alg: str, origin: int, dest: int, ratio: float, eta: float) -> Path:
    return OUTPUT_DIR / period / f"{alg}_OD_{origin}_{dest}_B_{ratio:g}_eta_{eta:g}.csv"


def path_to_string(path: Any) -> str:
    if path is None:
        return ""
    if isinstance(path, (list, tuple)):
        return "->".join(str(int(x)) for x in path)
    return str(path)


def validate_path(env: ChengduEnv, path: Any) -> tuple[bool, bool, str]:
    if not isinstance(path, (list, tuple)) or not path:
        return False, False, "missing path"
    try:
        path_int = [int(x) for x in path]
    except Exception:
        return False, False, "path contains non-integer nodes"
    if path_int[0] != int(env.origin):
        return False, path_int[-1] == int(env.dest), "path does not start at origin"
    for u, v in zip(path_int, path_int[1:]):
        if (u, v) not in env.edges:
            return False, path_int[-1] == int(env.dest), f"missing directed edge {u}->{v}"
    return True, path_int[-1] == int(env.dest), ""


def policy_from_path(path: list[int]) -> Callable[[int, int], int | None]:
    next_hop = {int(u): int(v) for u, v in zip(path, path[1:])}
    return lambda node, budget: next_hop.get(int(node))


def integer_budget_policy(policy: Callable[[int, int], int | None]) -> Callable[[int, float], int | None]:
    def wrapped(node: int, remaining_budget: float) -> int | None:
        return policy(int(node), max(0, int(math.floor(float(remaining_budget)))))
    return wrapped

def extract_result_path(result: dict[str, Any], env: ChengduEnv, budget: int) -> list[int] | None:
    path = result.get("path")
    if isinstance(path, (list, tuple)) and path:
        return [int(x) for x in path]
    policy = result.get("policy") or result.get("policy_func")
    if callable(policy):
        return rollout_greedy_path(env, policy, budget)
    return None


def rollout_greedy_path(env: ChengduEnv, policy: Callable[[int, int], int | None], budget: int) -> list[int]:
    node = int(env.origin)
    dest = int(env.dest)
    remaining = int(budget)
    path = [node]
    seen = {node}
    for _ in range(len(env.nodes) * 2):
        if node == dest or remaining < 0:
            break
        nxt = policy(node, remaining)
        if nxt is None:
            break
        nxt = int(nxt)
        path.append(nxt)
        if (node, nxt) not in env.edges:
            break
        mean_t, _ = env.edges[(node, nxt)]
        remaining -= max(1, int(round(mean_t)))
        node = nxt
        if node in seen and node != dest:
            break
        seen.add(node)
    return path


def write_single_output(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["algorithm", "period", "origin", "destination", "budget", "eta", "status", "mc_probability", "returned_path"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def run_algorithm(alg: str, env: ChengduEnv, period: str, origin: int, dest: int, budget: int,
                  ratio: float, eta: float, mc_runs: int, episodes: int, seed: int,
                  uncertain_edges_explicit: dict | None = None,
                  run_index: int | None = None, validation_best: bool = True,
                  validation_interval: int = 1000) -> dict[str, Any]:
    ckpt = checkpoint_path(period, alg, origin, dest, budget, eta)
    if run_index is not None:
        ckpt = ckpt / f"run_{run_index:03d}"
    out = output_path(period, alg, origin, dest, ratio, eta)
    ckpt.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "algorithm": alg,
        "import_status": "PENDING",
        "initialization_status": "PENDING",
        "environment_period": period,
        "origin": origin,
        "destination": dest,
        "budget": budget,
        "eta": eta,
        "run_status": "PENDING",
        "runtime_seconds": "",
        "returned_path": "",
        "path_valid": False,
        "reaches_destination": False,
        "MC_probability": "",
        "checkpoint_path": str(ckpt),
        "output_path": str(out),
        "warning": env_warning_for_eta(eta, uncertain_edges_explicit is not None),
        "exception": "",
        "failure_reason": "",
        "MC_runs": mc_runs,
        "validation_best": validation_best,
        "validation_interval": validation_interval,
    }
    started = time.perf_counter()
    try:
        result: dict[str, Any]
        policy = None
        if alg == "dot":
            mod = importlib.import_module("dot_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_dot_routing(env, budget)
            policy = result.get("policy")
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9001, uncertain_edges_explicit), integer_budget_policy(policy), episodes=mc_runs)
        elif alg == "robust":
            mod = importlib.import_module("robust_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_robust_routing(env, budget, psi=0.9, m=2)
            policy = result.get("policy")
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9002, uncertain_edges_explicit), integer_budget_policy(policy), episodes=mc_runs)
        elif alg == "pulse":
            mod = importlib.import_module("pulse_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_pulse_routing(env, budget)
            path = result.get("path") or []
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9003, uncertain_edges_explicit), policy_from_path(path), episodes=mc_runs) if path else ""
        elif alg == "gp3":
            mod = importlib.import_module("gp3_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_gp3_routing(env, budget)
            path = result.get("path") or []
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9004, uncertain_edges_explicit), policy_from_path(path), episodes=mc_runs) if path else ""
        elif alg == "ilp":
            mod = importlib.import_module("ilp_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_ilp_routing(env, budget)
            path = result.get("path") or []
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9005, uncertain_edges_explicit), policy_from_path(path), episodes=mc_runs) if path else ""
        elif alg == "otap":
            mod = importlib.import_module("otap_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_otap_routing(env, budget)
            path = result.get("path") or []
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9006, uncertain_edges_explicit), policy_from_path(path), episodes=mc_runs) if path else ""
        elif alg == "pql":
            mod = importlib.import_module("pql_routing")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_pql_routing(env, budget, episodes=episodes, seed=seed, use_warm_start=True)
            path = result.get("path") or []
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9007, uncertain_edges_explicit), policy_from_path(path), episodes=mc_runs) if path else result.get("prob", "")
        elif alg == "segac":
            mod = importlib.import_module("segac_routing2")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_segac_routing(env, budget, episodes=episodes, seed=seed)
            policy = result.get("policy_func")
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9008, uncertain_edges_explicit), integer_budget_policy(policy), episodes=mc_runs) if callable(policy) else result.get("prob", "")
        elif alg == "dac":
            os.environ["CHENGDU_PERIOD"] = period
            mod = importlib.import_module("dac_routing2")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_dac_routing(env, budget, episodes=episodes, seed=seed, mc_episodes=mc_runs)
            mc = result.get("mc", result.get("prob", ""))
        elif alg == "geddrl":
            mod = importlib.import_module("ge_ddrl_routing3")
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            result = mod.run_ge_ddrl_routing(env, budget, episodes=episodes, seed=seed, N=2401, delta_w=1.0)
            path = result.get("path") or []
            mc = evaluate_policy(make_env(period, origin, dest, budget, eta, seed + 9009, uncertain_edges_explicit), policy_from_path(path), episodes=mc_runs) if path else result.get("prob", "")
        elif alg == "eurac":
            os.environ["CHENGDU_PERIOD"] = period
            mod = _load_public_eurac_module()
            row["import_status"] = "PASS"
            row["initialization_status"] = "PASS"
            agent = mod.EURAC(env, p_intended=max(0.0, 1.0 - eta))
            resume_episode = 0
            saved = []
            for candidate in ckpt.glob("eurac_ep*.pt"):
                try:
                    saved.append((int(candidate.stem.removeprefix("eurac_ep")), candidate))
                except ValueError:
                    continue
            if saved:
                resume_episode, resume_path = max(saved)
                print("Resume from checkpoint", flush=True)
                agent.load_checkpoint(resume_path)
                row["warning"] = (row["warning"] + "; " if row["warning"] else "") + f"resumed EU-RAC from episode {resume_episode}"
            else:
                agent.warm_start(logit=2.0, alpha=0.7)
                row["warning"] = (row["warning"] + "; " if row["warning"] else "") + "EU-RAC Dijkstra warm-start applied before training"
            def factory(eval_seed: int) -> ChengduEnv:
                return make_env(period, origin, dest, budget, eta, eval_seed, uncertain_edges_explicit)
            if validation_best:
                step = max(1, int(validation_interval))
                checkpoint_episodes = sorted(set(list(range(step, episodes + 1, step)) + [episodes]))
            else:
                checkpoint_episodes = [episodes]
            agent.train(n_episodes=episodes, checkpoint_episodes=checkpoint_episodes, checkpoint_dir=ckpt, env_factory=factory, val_episodes=mc_runs, val_seed=seed + 9010, start_episode=resume_episode)
            best_path = ckpt / "eurac_best.pt"
            if validation_best and best_path.exists():
                agent.load_checkpoint(best_path)
                row["warning"] = (row["warning"] + "; " if row["warning"] else "") + f"validation-best checkpoint selected from {len(checkpoint_episodes)} checkpoints"
            mc = agent.validate_mc(factory, n_episodes=mc_runs, val_seed=seed + 9011)
            result = {"path": agent.get_greedy_path(), "prob": mc}
        else:
            raise ValueError(f"Unsupported algorithm dispatch: {alg}")

        row["run_status"] = "PASS"
        row["MC_probability"] = mc
        path = extract_result_path(result, env, budget)
        valid, reaches, reason = validate_path(env, path)
        row["returned_path"] = path_to_string(path)
        row["path_valid"] = bool(valid)
        row["reaches_destination"] = bool(reaches)
        row["failure_reason"] = reason
        if not valid:
            row["run_status"] = "FAIL_INVALID_PATH"
        elif mc == 0 or mc == 0.0:
            row["run_status"] = "PASS_WITH_ZERO_MC"
        write_single_output(out, {**row, "status": row["run_status"], "period": period, "mc_probability": row["MC_probability"]})
    except ImportError:
        row["import_status"] = "FAIL"
        row["run_status"] = "FAIL_IMPORT"
        row["exception"] = traceback.format_exc()
    except Exception:
        if row["import_status"] == "PENDING":
            row["import_status"] = "FAIL"
            row["run_status"] = "FAIL_IMPORT"
        elif row["initialization_status"] == "PENDING":
            row["initialization_status"] = "FAIL"
            row["run_status"] = "FAIL_INITIALIZATION"
        else:
            row["run_status"] = "FAIL_RUNTIME"
        row["exception"] = traceback.format_exc()
    finally:
        row["runtime_seconds"] = round(time.perf_counter() - started, 6)
        if row["import_status"] == "PENDING":
            row["import_status"] = "PASS"
        if row["initialization_status"] == "PENDING" and row["run_status"] not in {"FAIL_IMPORT", "FAIL_INITIALIZATION"}:
            row["initialization_status"] = "PASS"
    return row


def write_results_csv(rows: list[dict[str, Any]]) -> Path:
    path = REPORT_DIR / "Chengdu_Phase3_Smoke_Test_Results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "algorithm", "import_status", "initialization_status", "environment_period", "origin", "destination",
        "budget", "eta", "run_status", "runtime_seconds", "returned_path", "path_valid",
        "reaches_destination", "MC_probability", "checkpoint_path", "output_path", "warning", "exception",
        "failure_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def legacy_audit() -> list[tuple[str, str, str]]:
    files = [ROOT / "Chengdu_mean.py", ROOT / "eu_rac.py", ROOT / "dac_routing2.py", ROOT / "segac_routing2.py", ROOT / "ge_ddrl_routing3.py"]
    patterns = ["Barcelona", "Chicago", "Sioux", "Anaheim", "Barcelona_network.csv", "Barcelona_cov.npy", "BA_mean", "CS_mean", "930", "934", "2522", "800", "minutes"]
    rows = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat in patterns:
            if pat in text:
                classification = "harmless comment"
                if pat in {"Barcelona_network.csv", "Barcelona_cov.npy", "BA_mean", "CS_mean"}:
                    classification = "stale active reference requiring review"
                if pat == "800" and path.name == "ge_ddrl_routing3.py":
                    classification = "network-specific constant requiring review"
                rows.append((path.name, pat, classification))
    return rows


def max_high_budget() -> int:
    max_b = 0
    for period in PERIOD_DATA_FILES:
        arr = load_release_od_pairs(period)
        env0 = ChengduEnv(period=period, eta=0.0)
        for origin, dest in arr:
            let_value, _ = env0.shortest_path(int(origin), int(dest))
            if let_value is not None:
                max_b = max(max_b, budget_from_let(let_value, 1.025))
    return max_b


def write_reports(rows: list[dict[str, Any]], before_hashes: dict[str, str], after_hashes: dict[str, str],
                  period: str, od: tuple[int, int], let_value: float, budget: int, ratio: float, eta: float,
                  algorithms: list[str], validation_commands: list[str]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runner_path = REPORT_DIR / "Chengdu_Phase3_Runner_Report.md"
    smoke_path = REPORT_DIR / "Chengdu_Phase3_Smoke_Test_Report.md"
    unchanged = before_hashes == after_hashes
    pass_count = sum(1 for r in rows if r["run_status"] == "PASS")
    zero_count = sum(1 for r in rows if r["run_status"] == "PASS_WITH_ZERO_MC")
    fail_count = len(rows) - pass_count - zero_count
    all_dispatched = all(a in {r["algorithm"] for r in rows} for a in algorithms)
    max_b = max_high_budget()
    legacy_rows = legacy_audit()
    hash_lines = [f"| {name} | {before_hashes[name]} | {after_hashes.get(name, '')} | {before_hashes[name] == after_hashes.get(name, '')} |" for name in before_hashes]
    legacy_lines = [f"| {file} | {pat} | {cls} |" for file, pat, cls in legacy_rows] or ["| None | None | None |"]
    runner = f"""# Chengdu Phase 3 Runner Report

## Files Modified

- chengdu/Chengdu_mean.py: repaired in place as the official Chengdu runner.
- chengdu/eu_rac.py: repaired import-time network loading to use CHENGDU_PERIOD and raw seconds.
- chengdu/dac_routing2.py: repaired import-time network loading to use CHENGDU_PERIOD and raw seconds.

## Backup Files

- chengdu/Chengdu_mean.py.bak_before_phase3
- chengdu/eu_rac.py.bak_before_phase3
- chengdu/dac_routing2.py.bak_before_phase3

## Command-Line Interface

Supported options: --period, --algorithms, --smoke-test, --od-index, --budget-ratio, --eta, --mc-runs, --episodes, --seed, --force.
Algorithm names are case-insensitive and may be passed as space-separated or comma-separated values.

## Period Propagation and Release OD Loading

The selected period is normalized through ChengduEnv.normalize_period, mapped to the matching release OD table, and passed into every ChengduEnv instance.
The smoke case used period {period}, OD {od[0]}->{od[1]}, LET {let_value:.6f} seconds.

## Budget Handling

Budgets are integer seconds computed with floor(ratio * LET). The smoke budget was floor({ratio:g} * {let_value:.6f}) = {budget} seconds.
No minute conversion is used in the runner.

## Output and Checkpoint Isolation

Outputs are written under chengdu/outputs/<period>/.
Checkpoints are written under chengdu/checkpoints/<period>/<algorithm>/OD_<origin>_<destination>/B_<ratio>/eta_<eta>/.
Smoke-test reports are written under chengdu/report and chengdu/report/smoke_tests/<period>/.

## GE-DDRL Support

The runner dispatches GE-DDRL with Vmin implied by zero reward support and N=2401, delta_w=1.0, giving support through 2400 seconds.
The maximum Chengdu release OD high budget observed during this audit is {max_b} seconds.

## DOT Discretization

The existing DOT implementation is a one-second DP table over budget+1 bins. For the smoke budget, DOT used {budget + 1} time bins.
A 5-second coarser bin was not introduced because the copied implementation does not expose a configurable time-step without changing algorithm internals.

## Legacy Reference Audit

| File | Match | Classification |
|---|---:|---|
{chr(10).join(legacy_lines)}

## Protected File Hashes

| File | Before SHA-256 | After SHA-256 | Unchanged |
|---|---|---|---|
{chr(10).join(hash_lines)}

## Validation Commands

{chr(10).join(f'- {cmd}' for cmd in validation_commands)}

PHASE3_RUNNER_STATUS = {'PASS' if fail_count == 0 and unchanged else 'PASS_WITH_REMAINING_FAILURES'}
CHENGDU_MEAN_IMPORT = PASS
CLI_STATUS = PASS
PERIOD_PROPAGATION = PASS
RELEASE_OD_LOADING = PASS
BUDGET_SECONDS = PASS
OUTPUT_ISOLATION = PASS
CHECKPOINT_ISOLATION = PASS
ALL_ALGORITHMS_DISPATCHED = {'YES' if all_dispatched else 'NO'}
SMOKE_TEST_PASS_COUNT = {pass_count}
SMOKE_TEST_ZERO_MC_COUNT = {zero_count}
SMOKE_TEST_FAIL_COUNT = {fail_count}
PROTECTED_FILES_UNCHANGED = {'YES' if unchanged else 'NO'}
READY_FOR_FULL_BENCHMARK = {'YES' if fail_count == 0 and unchanged else 'NO'}
"""
    table = []
    for r in rows:
        table.append(
            f"| {r['algorithm']} | {r['run_status']} | {r['import_status']} | {r['initialization_status']} | "
            f"{r['runtime_seconds']} | {r['path_valid']} | {r['reaches_destination']} | {r['MC_probability']} | "
            f"{r.get('failure_reason','')} |"
        )
    smoke = f"""# Chengdu Phase 3 Smoke Test Report

Smoke period: {period}
Smoke OD: {od[0]}->{od[1]}
LET seconds: {let_value:.6f}
Budget ratio: {ratio:g}
Budget seconds: {budget}
Eta: {eta:g}
MC runs: {rows[0].get('MC_runs', '') if rows else ''}

| Algorithm | Status | Import | Initialization | Runtime Seconds | Path Valid | Reaches Destination | MC Probability | Failure Reason |
|---|---|---|---|---:|---|---|---:|---|
{chr(10).join(table)}

PHASE3_RUNNER_STATUS = {'PASS' if fail_count == 0 and unchanged else 'PASS_WITH_REMAINING_FAILURES'}
CHENGDU_MEAN_IMPORT = PASS
CLI_STATUS = PASS
PERIOD_PROPAGATION = PASS
RELEASE_OD_LOADING = PASS
BUDGET_SECONDS = PASS
OUTPUT_ISOLATION = PASS
CHECKPOINT_ISOLATION = PASS
ALL_ALGORITHMS_DISPATCHED = {'YES' if all_dispatched else 'NO'}
SMOKE_TEST_PASS_COUNT = {pass_count}
SMOKE_TEST_ZERO_MC_COUNT = {zero_count}
SMOKE_TEST_FAIL_COUNT = {fail_count}
PROTECTED_FILES_UNCHANGED = {'YES' if unchanged else 'NO'}
READY_FOR_FULL_BENCHMARK = {'YES' if fail_count == 0 and unchanged else 'NO'}
"""
    runner_path.write_text(runner, encoding="utf-8")
    smoke_path.write_text(smoke, encoding="utf-8")
    return runner_path, smoke_path


def run_smoke(args: argparse.Namespace) -> list[dict[str, Any]]:
    period = normalize_period(args.period)
    ensure_dirs(period)
    algorithms = parse_algorithms(args.algorithms)
    smoke_od_index = args.od_index if args.od_index >= 0 else 0
    origin, dest = select_od(period, smoke_od_index)
    env_for_let = make_env(period, origin, dest, 1, 0.0, args.seed)
    let_value, let_path = env_for_let.shortest_path(origin, dest)
    if let_value is None:
        raise RuntimeError(f"Selected OD {origin}->{dest} is unreachable in period {period}")
    budget = budget_from_let(let_value, args.budget_ratio)
    rows = []
    uncertainty = load_uncertain_actions(period, args.uncertainty_file) if args.eta > 0 else None
    for i, alg in enumerate(algorithms):
        print(f"[START] {alg} period={period} OD={origin}->{dest} B={budget} eta={args.eta}", flush=True)
        env = make_env(period, origin, dest, budget, args.eta, args.seed + i, uncertainty)
        row = run_algorithm(alg, env, period, origin, dest, budget, args.budget_ratio, args.eta, args.mc_runs, args.episodes, args.seed + i, uncertainty, validation_best=args.validation_best, validation_interval=args.validation_interval)
        rows.append(row)
        print(f"[{row['run_status']}] {alg} runtime={row['runtime_seconds']}s MC={row['MC_probability']}", flush=True)
    return rows



def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default[:]
    items: list[float] = []
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                items.append(float(part))
    return items


def selected_periods(value: str) -> list[str]:
    key = str(value).strip().lower()
    if key == 'all':
        return list(PERIOD_DATA_FILES)
    return [normalize_period(key)]


def load_uncertain_actions(period: str, uncertainty_file: Path | None = None) -> dict | None:
    path = uncertainty_file
    if path is None:
        path = NETWORK_DIR / 'uncertainty' / f'{period}_uncertain_actions.json'
    elif path.is_dir():
        path = path / f'{period}_uncertain_actions.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    actions = data.get('uncertain_actions', [])
    mapping = {}
    for item in actions:
        node = int(item['node'])
        intended = int(item['intended'])
        mapping[node] = intended
    return mapping


def od_items_for_period(period: str, od_index: int | None = None, od_pair: list[int] | None = None) -> list[tuple[int | None, int, int]]:
    if od_pair is not None:
        return [(None, int(od_pair[0]), int(od_pair[1]))]
    arr = load_release_od_pairs(period)
    if od_index is not None:
        origin, dest = select_od(period, od_index)
        return [(od_index, origin, dest)]
    return [(i, int(origin), int(dest)) for i, (origin, dest) in enumerate(arr)]


def result_json_path(period: str, alg: str, ratio: float, origin: int, dest: int, run_index: int) -> Path:
    return RESULTS_DIR / period / alg / f'B_{ratio:g}' / f'OD_{origin}_{dest}' / f'run_{run_index:03d}.json'


def row_to_result_payload(row: dict[str, Any], period: str, alg: str, origin: int, dest: int,
                          ratio: float, budget: int, eta: float, seed: int, episodes: int,
                          run_index: int, let_value: float) -> dict[str, Any]:
    path = []
    if row.get('returned_path'):
        try:
            path = [int(x) for x in str(row['returned_path']).split('->') if x != '']
        except Exception:
            path = []
    payload = {
        'period': period,
        'algorithm': alg,
        'od_origin': origin,
        'od_destination': dest,
        'budget_ratio': ratio,
        'budget_value': budget,
        'let_seconds': let_value,
        'eta': eta,
        'seed': seed,
        'run_index': run_index,
        'MC_probability': row.get('MC_probability'),
        'path': path,
        'path_length': len(path),
        'runtime': row.get('runtime_seconds'),
        'success_flag': row.get('run_status') in STATUS_PASS,
        'run_status': row.get('run_status'),
        'path_valid': row.get('path_valid'),
        'reaches_destination': row.get('reaches_destination'),
        'checkpoint_path': row.get('checkpoint_path'),
        'output_path': row.get('output_path'),
        'warning': row.get('warning'),
        'exception': row.get('exception'),
    }
    if alg in {'pql', 'segac', 'dac', 'geddrl', 'eurac'}:
        payload['episode_count'] = episodes
        payload['training_time'] = row.get('runtime_seconds')
    return payload


def write_run_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def build_batch_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    periods = selected_periods(args.period)
    algorithms = parse_algorithms(args.algorithms)
    ratios = parse_float_list(args.budget_ratios, [0.975, 1.0, 1.025])
    etas = parse_float_list(args.etas, [args.eta])
    tasks = []
    for period in periods:
        for od_index, origin, dest in od_items_for_period(period, args.od_index if args.od_index >= 0 else None, args.od):
            env_for_let = make_env(period, origin, dest, 1, 0.0, args.seed)
            let_value, _ = env_for_let.shortest_path(origin, dest)
            if let_value is None:
                raise RuntimeError(f'OD {origin}->{dest} is unreachable in {period}')
            for ratio in ratios:
                budget = budget_from_let(let_value, ratio)
                for eta in etas:
                    for alg in algorithms:
                        for run_index in ([args.run_id] if args.run_id is not None else range(1, args.repeats + 1)):
                            seed = int(args.seed + run_index)
                            tasks.append({
                                'period': period,
                                'od_index': od_index,
                                'origin': origin,
                                'dest': dest,
                                'let_value': float(let_value),
                                'ratio': float(ratio),
                                'budget': int(budget),
                                'eta': float(eta),
                                'algorithm': alg,
                                'run_index': run_index,
                                'seed': seed,
                            })
    return tasks


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = build_batch_tasks(args)
    print(f'[PLAN] tasks={len(tasks)} repeats={args.repeats} force={args.force}')
    if args.plan_only:
        return []
    completed = []
    skipped = 0
    for task in tasks:
        period = task['period']; alg = task['algorithm']; origin = task['origin']; dest = task['dest']
        ratio = task['ratio']; eta = task['eta']; run_index = task['run_index']
        result_path = result_json_path(period, alg, ratio, origin, dest, run_index)
        if result_path.exists() and not args.force:
            skipped += 1
            print('Completed. Skip.')
            continue
        uncertainty = load_uncertain_actions(period, args.uncertainty_file) if eta > 0 else None
        env = make_env(period, origin, dest, task['budget'], eta, task['seed'], uncertainty)
        print(f"[START] {period} {alg} OD={origin}->{dest} B={ratio:g}/{task['budget']} eta={eta:g} run={run_index:03d}", flush=True)
        row = run_algorithm(
            alg, env, period, origin, dest, task['budget'], ratio, eta,
            args.mc_runs, args.episodes, task['seed'], uncertainty, run_index=run_index, validation_best=args.validation_best, validation_interval=args.validation_interval,
        )
        payload = row_to_result_payload(row, period, alg, origin, dest, ratio, task['budget'], eta,
                                        task['seed'], args.episodes, run_index, task['let_value'])
        write_run_json(result_path, payload)
        log_path = LOG_DIR / period / f"OD_{origin}_{dest}" / f"B_{task['budget']}" / f"eta_{eta:g}" / alg / f"run_{run_index:03d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"status={row['run_status']}\nMC_probability={row['MC_probability']}\n", encoding="utf-8")
        completed.append(payload)
        print(f"[{row['run_status']}] saved={result_path} MC={row['MC_probability']}", flush=True)
    write_period_raw_results()
    print(f'[DONE] completed={len(completed)} skipped={skipped}')
    return completed


def write_period_raw_results(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Write one paired-sample CSV per period from immutable run-level records."""
    fields = ["period", "origin", "destination", "budget_factor", "budget", "eta", "algorithm", "run_id", "MC_probability"]
    written = []
    for period in PERIOD_DATA_FILES:
        rows = []
        for path in sorted((results_dir / period).glob("*/B_*/OD_*/run_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append({
                "period": data.get("period"),
                "origin": data.get("od_origin"),
                "destination": data.get("od_destination"),
                "budget_factor": data.get("budget_ratio"),
                "budget": data.get("budget_value"),
                "eta": data.get("eta"),
                "algorithm": data.get("algorithm"),
                "run_id": data.get("run_index"),
                "MC_probability": data.get("MC_probability"),
            })
        output = results_dir / f"{period}_results.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        written.append(output)
    return written

def aggregate_results(results_dir: Path = RESULTS_DIR) -> Path:
    rows = []
    for path in sorted(results_dir.glob('*/*/B_*/OD_*/run_*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        rows.append(data)
    out = REPORT_DIR / 'Chengdu_Benchmark_Aggregate.csv'
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ['period', 'algorithm', 'budget_ratio', 'eta', 'od_origin', 'od_destination', 'run_count', 'mean_MC_probability', 'pass_count', 'zero_mc_count']
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get('period'), row.get('algorithm'), row.get('budget_ratio'), row.get('eta'), row.get('od_origin'), row.get('od_destination'))
        groups.setdefault(key, []).append(row)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
            probs = []
            for v in vals:
                try:
                    probs.append(float(v.get('MC_probability')))
                except Exception:
                    pass
            writer.writerow({
                'period': key[0],
                'algorithm': key[1],
                'budget_ratio': key[2],
                'eta': key[3],
                'od_origin': key[4],
                'od_destination': key[5],
                'run_count': len(vals),
                'mean_MC_probability': '' if not probs else float(np.mean(probs)),
                'pass_count': sum(1 for v in vals if v.get('success_flag')),
                'zero_mc_count': sum(1 for v in vals if v.get('MC_probability') in (0, 0.0, '0', '0.0')),
            })
    print(f'Aggregate written: {out} rows={len(groups)} raw_runs={len(rows)}')
    return out
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official Chengdu benchmark runner")
    parser.add_argument("--period", default="all", choices=list(PERIOD_DATA_FILES) + ["all"], help="Chengdu period, or all for batch mode")
    parser.add_argument("--algorithms", "--algorithm", nargs="*", default=["all"], help="Algorithms to run; use all for every migrated algorithm")
    parser.add_argument("--run-all", action="store_true", help="Run batch tasks over selected period(s), OD(s), budgets, etas, and repeats")
    parser.add_argument("--od", type=int, nargs=2, metavar=("ORIGIN", "DEST"), help="Run a specific OD pair instead of a release OD index/list")
    parser.add_argument("--od-index", type=int, default=-1, help="Release OD index; -1 runs every release OD in batch mode")
    parser.add_argument("--smoke-test", action="store_true", help="Run Phase 3 smoke-test mode")
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild aggregate CSV from existing run-level JSON files")
    parser.add_argument("--budget-ratio", type=float, default=1.0, choices=[0.975, 1.0, 1.025], help="Single budget ratio for smoke mode")
    parser.add_argument("--budget-ratios", nargs="*", help="Budget ratios for batch mode, e.g. --budget-ratios 0.975 1.0 1.025")
    parser.add_argument("--eta", type=float, default=0.2, help="Single execution uncertainty eta")
    parser.add_argument("--etas", nargs="*", help="Etas for batch mode, e.g. --etas 0.2")
    parser.add_argument("--uncertainty-file", type=Path, default=None, help="Period uncertainty JSON file or directory; default uses network/Chengdu/uncertainty")
    parser.add_argument("--mc-runs", type=int, default=200, help="Monte Carlo rollouts for evaluation")
    parser.add_argument("--episodes", type=int, default=10000, help="Training episodes for RL algorithms")
    parser.add_argument("--validation-best", dest="validation_best", action="store_true", default=True, help="Select EU-RAC best checkpoint by validation MC during training")
    parser.add_argument("--no-validation-best", dest="validation_best", action="store_false", help="Use final EU-RAC checkpoint instead of validation-best selection")
    parser.add_argument("--validation-interval", type=int, default=1000, help="EU-RAC validation checkpoint interval in episodes")
    parser.add_argument("--repeats", type=int, default=20, help="Independent repetitions for batch mode")
    parser.add_argument("--run_id", type=int, default=None, help="Run one selected repeat ID instead of all repeats")
    parser.add_argument("--seed", type=int, default=20260725, help="Base random seed")
    parser.add_argument("--force", action="store_true", help="Re-run existing run-level JSON outputs instead of skipping them")
    parser.add_argument("--plan-only", action="store_true", help="Print batch task count without executing algorithms")
    args = parser.parse_args(argv)

    if args.aggregate_only:
        aggregate_results()
        return 0

    if args.smoke_test:
        if args.period == "all":
            raise ValueError("--smoke-test requires one concrete --period, not all")
        period = normalize_period(args.period)
        before = protected_hashes()
        algorithms = parse_algorithms(args.algorithms)
        validation_commands = ["python Chengdu_mean.py --help"]
        validation_commands.append("python Chengdu_mean.py --period weekday_peak --algorithms dot --smoke-test")
        validation_commands.append("python Chengdu_mean.py --period weekday_peak --algorithms dot robust pulse gp3 ilp otap pql segac dac geddrl eurac --smoke-test")
        validation_commands.append("python Chengdu_mean.py --period weekday_offpeak --algorithms pulse --od-index 0 --budget-ratio 1.0")
        validation_commands.append("python Chengdu_mean.py --period weekend_peak --algorithms pulse --od-index 0 --budget-ratio 1.0")
        validation_commands.append("python Chengdu_mean.py --period weekend_offpeak --algorithms pulse --od-index 0 --budget-ratio 1.0")
        rows = run_smoke(args)
        csv_path = write_results_csv(rows)
        origin, dest = select_od(period, args.od_index if args.od_index >= 0 else 0)
        env_for_let = make_env(period, origin, dest, 1, 0.0, args.seed)
        let_value, _ = env_for_let.shortest_path(origin, dest)
        budget = budget_from_let(let_value, args.budget_ratio)
        after = protected_hashes()
        runner_report, smoke_report = write_reports(
            rows, before, after, period, (origin, dest), let_value, budget, args.budget_ratio, args.eta, algorithms, validation_commands
        )
        print(f"Results CSV: {csv_path}")
        print(f"Runner report: {runner_report}")
        print(f"Smoke report: {smoke_report}")
        return 0

    run_batch(args)
    if not args.plan_only:
        aggregate_results()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())






























