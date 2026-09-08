"""
CS_mean.py - Chicago Sketch
Usage:
    python CS_mean.py --run-all          
    python CS_mean.py --od 491 93        
    python CS_mean.py --aggregate-only 
"""


DEBUG_OD_CONFIGS = {
    (915, 456):  {},   # t_LET=118.75h, path_len=19, min_b=119
    (368, 581):  {},   # t_LET=112.36h, path_len=29, min_b=114
    (930, 237):  {},   # t_LET=159.76h, path_len=41, min_b=159
    (623, 798):  {},   # t_LET=50.60h, path_len=15, min_b=51
    (830, 530):  {},   # t_LET=52.84h, path_len=16, min_b=54
}

ETA_LIST = [0, 0.2, 0.4]          
B_COEFFS = [0.975, 1, 1.025]      


GLOBAL_UNCERTAIN_RATIO = 0.2   
# Changed from 42 15: seed=42 gave 0 LET edges affected across all ODs on
# Chicago's 933-node network (too sparse). seed=15 gives ~9 LET edges
# affected across 5 ODs, providing meaningful impact while keeping the
# same methodology (20% random, same ratio as Sioux/Anaheim).
GLOBAL_UNCERTAIN_SEED = 15

# OD-conditioned calibration for Chicago only. Global random uncertainty is
# kept, then selected LET transitions are added until OD exposure is comparable
# to Anaheim/Sioux without covering the full LET path.
CHICAGO_USE_OD_UNCERTAINTY = True
CHICAGO_TARGET_UNCERTAIN_RATIO = 0.15
CHICAGO_MAX_UNCERTAIN_RATIO = 0.20
CHICAGO_OD_UNCERTAIN_SEED = 20260710


METHODS = ['EU-RAC', 'Pulse', 'DOT', 'Robust', 'GP3', 'ILP', 'OTAP',
           'SEGAC', 'PQL', 'DAC', 'GE-DDRL']

METHOD_ALIASES = {
    'eurac': 'EU-RAC',
    'eu-rac': 'EU-RAC',
    'eurac': 'EU-RAC',
    'eu-rac': 'EU-RAC',
    'pulse': 'Pulse',
    'dot': 'DOT',
    'robust': 'Robust',
    'gp3': 'GP3',
    'ilp': 'ILP',
    'otap': 'OTAP',
    'segac': 'SEGAC',
    'pql': 'PQL',
    'dac': 'DAC',
    'ge-ddrl': 'GE-DDRL',
    'geddrl': 'GE-DDRL',
    'ge_ddrl': 'GE-DDRL',
}

# ============================================================
import sys
import time
import random
import heapq
import glob
import math
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Local Chicago modules

import importlib.util
_ROOT_DIR = Path(__file__).resolve().parent.parent
_EURAC_SPEC = importlib.util.spec_from_file_location("_public_eu_rac", _ROOT_DIR / "eu_rac.py")
_EURAC_MODULE = importlib.util.module_from_spec(_EURAC_SPEC)
_EURAC_SPEC.loader.exec_module(_EURAC_MODULE)
EURAC = _EURAC_MODULE.EURAC
from chicago_env import ChicagoEnv
from robust_routing import run_robust_routing, extract_path as rr_extract_path
from dot_routing import run_dot_routing
from pulse_routing import run_pulse_routing
from otap_routing import run_otap_routing
from ilp_routing import run_ilp_routing
from gp3_routing import run_gp3_routing
from pql_routing import run_pql_routing        # original tabular Q-learning
from ge_ddrl_routing3 import run_ge_ddrl_routing  # Chicago-adapted GE-DDRL v3
from segac_routing2 import run_segac_routing   # SEGAC: Dijkstra BC init + GPG episode replay
from dac_routing2 import run_dac_routing        # DAC + Dijkstra Safety Fallback

from evaluator import evaluate_policy

# ============================================================
# Dijkstra
# ============================================================
_NET_DIR = Path(__file__).parent.parent / 'network' / 'Chicago_Sketch'
_net_df = pd.read_csv(_NET_DIR / 'Chicago_Sketch_network.csv')

_mean_edges = {}
_successors = {}
for _, row in _net_df.iterrows():
    u, v = int(row['From']), int(row['To'])
    _mean_edges[(u, v)] = float(row['Cost'])   # hours
    _successors.setdefault(u, []).append(v)


def _dijkstra_mean(src, dst):
    """t_LET in hours (sum of raw mean travel times along shortest path)."""
    dist = {src: 0.0}
    pq = [(0.0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v in _successors.get(u, []):
            nd = d + _mean_edges[(u, v)]
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist[dst]


def _dijkstra_rounded(src, dst):
    """Minimum budget needed using env's actual edge costs (max(1, round(mean)))."""
    dist = {src: 0.0}
    pq = [(0.0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            return d
        for v in _successors.get(u, []):
            nd = d + max(1, round(_mean_edges[(u, v)]))
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float('inf')


def _dijkstra_edge_count(src, dst):
    """Minimum number of edges to reach destination."""
    dist = {src: 0}
    pq = [(0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            return d
        for v in _successors.get(u, []):
            nd = d + 1
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float('inf')



EVAL_EPISODES = 1000    
SEED = 42

EURAC_LR_E = 0.2
EURAC_LR_D = 0.2
EURAC_LR_ACTOR = 1e-4
EURAC_ENTROPY_COEF = 0.01
EURAC_KL_COEF = 0.1
EURAC_WARM_LOGIT = 2.0

EURAC_EPISODES = 5000   # TD learning + warm-start episodes
EURAC_VALIDATION_INTERVAL = 1000
EURAC_VALIDATION_EPISODES = 200
EURAC_SAFETY_DETOUR_TOLERANCE = 0.02
EURAC_MIN_ACTOR_HISTORY = 100
EURAC_MIN_ACTOR_SUCCESS_RATE = 0.01
EURAC_LOW_SUCCESS_KL_MULTIPLIER = 5.0
PQL_EPISODES = 50000
GE_EPISODES = 50000
SEGAC_EPISODES = 50000
DAC_EPISODES = 5000   # reduced from 50000; parameterized Q-network is about 40x slower per step

FALLBACK_Q_GAP_THRESHOLD = 0.01
DAC_DEPLOYMENT_NAME = 'DAC + Dijkstra Safety Fallback'

OUTPUT_DIR = Path(__file__).parent
DEFAULT_OD_SOURCE = OUTPUT_DIR / 'environment_v2_report.csv'
DEFAULT_UNCERTAINTY_FILE = OUTPUT_DIR / 'uncertain_edges_v3.json'


# ============================================================
# Helpers
# ============================================================
def mc_eval(env, policy_func):
    return evaluate_policy(env, policy_func, episodes=EVAL_EPISODES)


def fmt3(v):
    """Format numeric values to 3 decimals; missing/skipped values as NA."""
    if v is None:
        return "NA"
    try:
        if pd.isna(v):
            return "NA"
    except TypeError:
        pass
    return f"{float(v):.3f}"


def fmt_progress(v):
    if v is None:
        return "NA"
    try:
        if pd.isna(v):
            return "NA"
    except TypeError:
        pass
    return f"{float(v):.3f}"


def _od_key(origin, dest):
    return f'{int(origin)}_{int(dest)}'


def load_uncertainty_environments(path=DEFAULT_UNCERTAINTY_FILE):
    path = Path(path)
    if not path.exists():
        print(f'[CS_mean] uncertainty file not found: {path}')
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    envs = {}
    for key, info in raw.items():
        parts = str(key).split('_', 1)
        origin = int(info.get('origin', parts[0]))
        dest = int(info.get('destination', parts[1]))
        edge_specs = info.get('uncertain_edges') or []
        if edge_specs:
            explicit = {}
            for edge in edge_specs:
                spec = {'intended': int(edge.get('v', edge.get('intended')))}
                if edge.get('executed') is not None:
                    spec['executed'] = int(edge['executed'])
                explicit[int(edge['u'])] = spec
        else:
            explicit = {
                int(node): int(executed)
                for node, executed in (info.get('uncertain_edges_explicit') or {}).items()
            }
        envs[_od_key(origin, dest)] = dict(info, uncertain_edges_explicit=explicit)
    return envs


def load_final_od_pairs(path=DEFAULT_OD_SOURCE):
    path = Path(path)
    if not path.exists():
        print(f'[CS_mean] OD source not found: {path}; falling back to uncertainty JSON order')
        envs = load_uncertainty_environments()
        return [(int(v['origin']), int(v['destination'])) for v in envs.values()]
    df = pd.read_csv(path)
    if 'Primary / Backup' in df.columns:
        df = df[df['Primary / Backup'].isin(['Primary', 'Backup'])]
    return [(int(row['origin']), int(row['destination'])) for _, row in df.iterrows()]


# ============================================================
# Run all methods for one (origin, dest, budget, exec_prob)
# ============================================================
def normalize_algorithms(values):
    if not values or any(str(v).lower() == 'all' for v in values):
        return list(METHODS)
    selected = []
    for raw in values:
        key = str(raw).strip().lower()
        method = METHOD_ALIASES.get(key)
        if method is None:
            valid = ', '.join(['all'] + sorted(METHOD_ALIASES))
            raise ValueError(f'Unknown algorithm {raw!r}. Valid values: {valid}')
        if method not in selected:
            selected.append(method)
    return selected


def run_all_methods(env, origin, dest, budget, combo_seed, selected_methods=None, uncertain_edges_explicit=None, force_eurac=False):
    """Returns dict[method_name] = mc_value (float)."""
    results = {}
    selected_methods = set(selected_methods or METHODS)

    def _run_method(name):
        return name in selected_methods

    successors = env.successors
    exec_prob = env.exec_prob

    # ---- MC evaluation: fresh env with FIXED seed for every method ----
    MC_EVAL_SEED = 999999 + origin * 100 + dest

    def _make_mc_env():
        """Create a fresh, independent env for MC evaluation."""
        return ChicagoEnv(
            origin=origin, dest=dest, budget=budget,
            exec_prob=exec_prob, deterministic=False, seed=MC_EVAL_SEED,
            uncertain_ratio=GLOBAL_UNCERTAIN_RATIO,
            uncertain_seed=GLOBAL_UNCERTAIN_SEED,
            uncertain_mode='random', top_k=6,
            uncertain_edges_explicit=uncertain_edges_explicit,
            use_od_uncertainty=False if uncertain_edges_explicit is not None else CHICAGO_USE_OD_UNCERTAINTY,
            od_target_uncertain_ratio=CHICAGO_TARGET_UNCERTAIN_RATIO,
            od_max_uncertain_ratio=CHICAGO_MAX_UNCERTAIN_RATIO,
            od_uncertain_seed=CHICAGO_OD_UNCERTAIN_SEED,
        )

    def _mc_eval(policy_func, label=''):
        """MC evaluation on a fresh env; all methods see the same random sequence."""
        _t0 = time.time()
        val = evaluate_policy(_make_mc_env(), policy_func, episodes=EVAL_EPISODES)
        _elapsed = time.time() - _t0
        if _elapsed > 5:  # only show MC time if it's non-trivial
            print(f'    MC eval: {_elapsed:.1f}s ({EVAL_EPISODES} episodes, {_elapsed/EVAL_EPISODES*1000:.1f}ms/ep)')
        return val

    if _run_method('Pulse'):
        # ---- Pulse ----
        _t0 = time.time()
        r = run_pulse_routing(env, budget=budget)
        _t_algo = time.time() - _t0
        path = r['path']
        # Fix: if Pulse timed out and fallback path doesn't reach dest,
        # compute a proper Dijkstra path as fallback.
        if path[-1] != dest:
            print(f'  [Pulse] WARNING: path does not reach dest (ends at {path[-1]}), using Dijkstra fallback')
            # Use full Dijkstra (not greedy) to find a path to dest
            import heapq as _hq
            _dist = {origin: 0.0}; _prev = {}; _pq = [(0.0, origin)]
            while _pq:
                _d, _u = _hq.heappop(_pq)
                if _d > _dist.get(_u, float('inf')): continue
                if _u == dest: break
                for _v in successors.get(_u, []):
                    _mt, __ = env.edges.get((_u, _v), (1, 0))
                    _nd = _d + max(1, round(_mt))
                    if _nd < _dist.get(_v, float('inf')): _dist[_v] = _nd; _prev[_v] = _u; _hq.heappush(_pq, (_nd, _v))
            if dest in _prev:
                fb_path = [dest]; _n = dest
                while _n in _prev: fb_path.append(_prev[_n]); _n = _prev[_n]
                path = fb_path[::-1]
                print(f'  [Pulse] Dijkstra fallback path: {len(path)} nodes')
            else:
                print(f'  [Pulse] CRITICAL: no path found from {origin} to {dest}!')
        pm = {path[i]: path[i + 1] for i in range(len(path) - 1)}
        results['Pulse'] = _mc_eval(lambda n, b: pm.get(n), 'Pulse')
        print(f'  Pulse: algo={_t_algo:.1f}s')

    if _run_method('DOT'):
        # ---- DOT ----
        _t0 = time.time()
        r = run_dot_routing(env, budget=budget)
        _t_algo = time.time() - _t0
        # Use budget-adaptive policy (fix: static path map fails when _extract_path
        # runs out of budget before reaching dest due to simplified budget tracking)
        results['DOT'] = _mc_eval(r['policy'], 'DOT')
        print(f'  DOT: algo={_t_algo:.1f}s')

    if _run_method('Robust'):
        # ---- Robust psi=0.9 ----
        _t0 = time.time()
        r = run_robust_routing(env, budget=budget, psi=0.9, m=2)
        _t_algo = time.time() - _t0
        # Use budget-adaptive policy (fix: static path map from extract_path
        # fails due to simplified budget tracking, same as DOT above)
        results['Robust'] = _mc_eval(r['policy'], 'Robust')
        print(f'  Robust: algo={_t_algo:.1f}s')

    if _run_method('GP3'):
        # ---- GP3 ----
        _t0 = time.time()
        r = run_gp3_routing(env, budget=budget, zeta_min=0.0, zeta_max=50.0, seed=combo_seed)
        _t_algo = time.time() - _t0
        path = r['path']
        pm = {path[i]: path[i + 1] for i in range(len(path) - 1)}
        results['GP3'] = _mc_eval(lambda n, b: pm.get(n), 'GP3')
        _gp3_method = r.get('method', 'GP3')
        print(f'  GP3: algo={_t_algo:.1f}s  |  method={_gp3_method}')

    if _run_method('ILP'):
        # ---- ILP (K=300, per-sample tight Big-M, Dijkstra fallback) ----
        _t0 = time.time()
        r = run_ilp_routing(env, budget=budget, K=300, seed=combo_seed)
        _t_algo = time.time() - _t0
        path = r['path']
        pm = {path[i]: path[i + 1] for i in range(len(path) - 1)}
        results['ILP'] = _mc_eval(lambda n, b: pm.get(n), 'ILP')
        _ilp_method = r.get('method', 'ILP')
        _ilp_status = r.get('status', ',')
        print(f'  ILP: algo={_t_algo:.1f}s  |  method={_ilp_method}  |  status={_ilp_status}  |  '
              f'path_len={len(path)}  |  reaches_dest={path[-1] == dest if path else False}')

    if _run_method('OTAP'):
        # ---- OTAP (K=50, heuristic) ----
        _t0 = time.time()
        r = run_otap_routing(env, budget=budget, K=50, seed=combo_seed)
        _t_algo = time.time() - _t0
        path = r['path']
        pm = {path[i]: path[i + 1] for i in range(len(path) - 1)}
        results['OTAP'] = _mc_eval(lambda n, b: pm.get(n), 'OTAP')
        _otap_method = r.get('method', 'OTAP')
        print(f'  OTAP: algo={_t_algo:.1f}s  |  method={_otap_method}')

    if _run_method('SEGAC'):
        # ---- SEGAC ----
        _t0 = time.time()
        r = run_segac_routing(env, budget=budget, episodes=SEGAC_EPISODES,
                              lr_actor=0.01, lr_critic=0.1, seed=combo_seed)
        _t_algo = time.time() - _t0
        # SEGAC now uses a neural policy pi_theta(a|s); the returned policy_func calls
        # the trained network directly for MC evaluation (no Dijkstra fallback).
        _segac_policy = r.get('policy_func')
        _segac_path = r.get('path', [])
        _segac_reaches_dest = _segac_path[-1] == dest if _segac_path else False
        _segac_path_len = len(_segac_path)
        if not _segac_reaches_dest:
            print(f'  [SEGAC] WARNING: greedy path does not reach dest (ends at {_segac_path[-1] if _segac_path else "N/A"}, '
                  f'path_len={_segac_path_len}). Using neural policy for MC eval; no Dijkstra fallback.')
        if _segac_policy is None:
            # Fallback: use greedy path map (shouldn't happen with neural SEGAC)
            _segac_pm = {_segac_path[i]: _segac_path[i + 1] for i in range(len(_segac_path) - 1)}
            def _segac_policy(node, bud):
                if node in _segac_pm: return _segac_pm[node]
                acts = successors.get(node, [])
                return acts[0] if acts else None
        else:
            _segac_policy_local = _segac_policy
            def _segac_policy(node, bud):
                return _segac_policy_local(node, bud)
        results['SEGAC'] = _mc_eval(_segac_policy, 'SEGAC')
        _segac_v_pred = r.get('v_pred', float('nan'))
        _segac_n_params = r.get('n_params', 0)
        _segac_agree_bc = r.get('dijkstra_agreement_bc', float('nan'))
        _segac_agree_rl = r.get('dijkstra_agreement_rl', float('nan'))
        print(f'  SEGAC: algo={_t_algo:.1f}s  |  n_params={_segac_n_params}  |  '
              f'V_pred={_segac_v_pred:.4f}  |  path_reaches_dest={_segac_reaches_dest}  |  '
              f'path_len={_segac_path_len}')
        print(f'    agreement: BC={_segac_agree_bc:.3f} -> RL={_segac_agree_rl:.3f}  |  '
              f'expert trajs={r.get("expert_traj_count", 0)}  |  '
              f'BC loss={r.get("bc_loss", float("nan")):.4f}')
        print(f'    periodic expert BC={r.get("periodic_expert_bc", None)}  |  '
              f'episode replay={r.get("episode_replay_enabled", None)}  |  '
              f'replay size={r.get("replay_buffer_size", 0)}  |  '
              f'sampled eps={r.get("sampled_episode_count", 0)}')
        print(f'    IS ratio mu/std={r.get("importance_ratio_mean", float("nan")):.4f}/'
              f'{r.get("importance_ratio_std", float("nan")):.4f}  |  '
              f'cum ratio mu/std={r.get("cumulative_ratio_mean", float("nan")):.4f}/'
              f'{r.get("cumulative_ratio_std", float("nan")):.4f}  |  '
              f'GPG loss={r.get("gpg_loss", float("nan")):.4f}')
        print(f'    baseline mu/std={r.get("baseline_mean", float("nan")):.4f}/'
              f'{r.get("baseline_std", float("nan")):.4f}  |  '
              f'traj return mu/std={r.get("trajectory_return_mean", float("nan")):.4f}/'
              f'{r.get("trajectory_return_std", float("nan")):.4f}  |  '
              f'train success={r.get("training_success_rate", float("nan")):.4f}')
        print(f'    return ratios: R>0={r.get("return_positive_ratio", float("nan")):.3f}  '
              f'R=0={r.get("return_zero_ratio", float("nan")):.3f}  '
              f'R<0={r.get("return_negative_ratio", float("nan")):.3f}  |  '
              f'nonzero-loss traj={r.get("nonzero_loss_trajectory_ratio", float("nan")):.3f}')
        print(f'    optimizer steps={r.get("optimizer_step_count", 0)}  |  '
              f'grad norm mean/last={r.get("gradient_norm_mean", float("nan")):.4f}/'
              f'{r.get("gradient_norm_last", float("nan")):.4f}  |  '
              f'actor drift mean/last={r.get("actor_drift_mean", float("nan")):.6f}/'
              f'{r.get("actor_drift_last", float("nan")):.6f}')

    if _run_method('PQL'):
        # ---- PQL ----
        _t0 = time.time()
        _pql_min_budget = _dijkstra_rounded(origin, dest)
        _pql_warm_budgets = [int(math.floor(c * _pql_min_budget)) for c in B_COEFFS]
        r = run_pql_routing(env, budget=budget, episodes=PQL_EPISODES,
                            alpha=0.1, epsilon=0.1, seed=combo_seed,
                            use_warm_start=True,
                            warm_start_budgets=_pql_warm_budgets)
        _t_algo = time.time() - _t0
        q_table = r['q_table']
        _pql_dijkstra_next_hop = r.get('dijkstra_next_hop', {})
        _pql_policy_diag = {
            'total_policy_decisions': 0,
            'pql_decisions': 0,
            'fallback_decisions': 0,
            'unseen_state_fallback_count': 0,
            'invalid_action_decisions': 0,
        }
        def pql_policy(node, bud):
            acts = successors.get(node, [])
            if not acts:
                _pql_policy_diag['invalid_action_decisions'] += 1
                return None
            _pql_policy_diag['total_policy_decisions'] += 1
            q_node = q_table.get((node, bud), {})
            if q_node:
                _pql_policy_diag['pql_decisions'] += 1
                return max(acts, key=lambda a: q_node.get(a, 0.0))
            # Try nearby budgets for unseen states.
            for b_try in range(bud, max(0, bud - 10), -1):
                qn = q_table.get((node, b_try), {})
                if qn:
                    _pql_policy_diag['pql_decisions'] += 1
                    return max(acts, key=lambda a: qn.get(a, 0.0))
            fallback = _pql_dijkstra_next_hop.get(node)
            if fallback is not None and fallback in acts:
                _pql_policy_diag['fallback_decisions'] += 1
                _pql_policy_diag['unseen_state_fallback_count'] += 1
                return fallback
            _pql_policy_diag['invalid_action_decisions'] += 1
            return None
        results['PQL'] = _mc_eval(pql_policy, 'PQL')
        _pql_path = r['path']
        _pql_q_entries = len(q_table)
        _pql_total_decisions = _pql_policy_diag['total_policy_decisions']
        _pql_fallback_rate = (_pql_policy_diag['fallback_decisions'] / _pql_total_decisions
                              if _pql_total_decisions else 0.0)
        print(f'  PQL: algo={_t_algo:.1f}s  |  Q-table entries={_pql_q_entries}  |  '
              f'path_len={len(_pql_path)}  |  MC={results["PQL"]:.4f}')
        print(f'    deployment decisions: total={_pql_total_decisions}  |  '
              f'PQL={_pql_policy_diag["pql_decisions"]}  |  '
              f'fallback={_pql_policy_diag["fallback_decisions"]}  |  '
              f'fallback_rate={_pql_fallback_rate:.6f}  |  '
              f'unseen_fallback={_pql_policy_diag["unseen_state_fallback_count"]}  |  '
              f'invalid={_pql_policy_diag["invalid_action_decisions"]}')
        if _pql_path:
            print('    PQL greedy path: ' + '->'.join(str(n) for n in _pql_path[:15])
                  + ('...' if len(_pql_path) > 15 else ''))

    if _run_method('DAC'):
        # ---- DAC ----
        _t0 = time.time()
        r = run_dac_routing(env, budget=budget, episodes=DAC_EPISODES,
                            lr_critic=3e-4, lr_actor=0.1, lr_ent=3e-4, seed=combo_seed,
                            use_dijkstra_warm_start=True,
                            use_dijkstra_replay_warm_start=False,
                            use_persistent_expert_replay=False,
                            expert_replay_fraction=0.0,
                            use_adaptive_tau=False,
                            fixed_tau=1.0,
                            actor_state_source='trajectory',
                            evaluate_dijkstra_fallback=True,
                            fallback_q_gap_threshold=FALLBACK_Q_GAP_THRESHOLD,
                            deployment_audit_detail=False,
                            mc_episodes=EVAL_EPISODES)
        _t_algo = time.time() - _t0
        _dac_deploy = r.get('deployment_fallback_eval') or {}
        results['DAC'] = float(_dac_deploy.get('mc', 0.0))
        _dac_path = r['path']
        _dac_q_entries = len(r.get('q_table', {}))
        print(f'  DAC: {DAC_DEPLOYMENT_NAME}  |  algo={_t_algo:.1f}s  |  Q-table entries={_dac_q_entries}  |  '
              f'path_len={len(_dac_path)}  |  MC={results["DAC"]:.4f}')
        print(f'    fallback threshold={FALLBACK_Q_GAP_THRESHOLD:.3f}  |  '
              f'fallback ratio={float(_dac_deploy.get("fallback_ratio", 0.0)):.6f}')
        if _dac_path:
            print('    DAC greedy path: ' + '->'.join(str(n) for n in _dac_path[:15])
                  + ('...' if len(_dac_path) > 15 else ''))

    if _run_method('GE-DDRL'):
        # ---- GE-DDRL (Chicago-adapted tabular C51 v3) ----
        _t0 = time.time()
        print(f'    GE-DDRL: Chicago-adapted per-edge tabular Z')
        print(f'    Warm-start target: LET-to-go')
        r = run_ge_ddrl_routing(env, budget=budget, N=200, delta_w=1.0,
                                alpha_t=0.05, epsilon=0.1,
                                episodes=GE_EPISODES, seed=combo_seed)
        _t_algo = time.time() - _t0
        _ge_internal_prob = float(r.get('prob', 0.0))
        _ge_path = r.get('path', [])
        _ge_pm = {_ge_path[i]: _ge_path[i + 1] for i in range(len(_ge_path) - 1)}
        _ge_recovery_count = {'count': 0}

        def _ge_dijkstra_next_hops():
            reverse = {}
            for (u, v), (mean_t, _) in env.edges.items():
                reverse.setdefault(v, []).append((u, max(1, round(mean_t))))
            dist = {dest: 0.0}
            next_hop = {}
            pq = [(0.0, dest)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float('inf')):
                    continue
                for pred, cost in reverse.get(u, []):
                    nd = d + cost
                    if nd < dist.get(pred, float('inf')):
                        dist[pred] = nd
                        next_hop[pred] = u
                        heapq.heappush(pq, (nd, pred))
            return next_hop

        _ge_recovery_next = _ge_dijkstra_next_hops()
        def ge_policy(node, bud):
            if node in _ge_pm:
                return _ge_pm.get(node)
            action = _ge_recovery_next.get(node)
            if action is not None:
                _ge_recovery_count['count'] += 1
            return action

        _ge_unified_mc = _mc_eval(ge_policy, 'GE-DDRL')
        results['GE-DDRL'] = _ge_unified_mc
        _ge_np = r.get('n_params', 0)
        print(f'  GE-DDRL: algo={_t_algo:.1f}s  |  tabular entries={_ge_np}  |  '
              f'Z max={r.get("z_max", float("nan")):.4f}  |  '
              f'reaches_dest={r.get("reaches_dest", False)}  |  '
              f'path_len={r.get("path_len", 0)}  first_action={r.get("first_action", None)}')
        print(f'  [GE-DDRL]')
        print(f'    internal_prob={_ge_internal_prob:.4f}')
        print(f'    unified_MC={_ge_unified_mc:.4f}')
        print(f"    fallback_count={_ge_recovery_count['count']}")
        print(f'    path_len={len(_ge_path)}')
        print(f'    reaches_destination={bool(_ge_path and _ge_path[-1] == dest)}')
        if _ge_path:
            print('    GE-DDRL greedy path: ' + '->'.join(str(n) for n in _ge_path[:15])
                  + ('...' if len(_ge_path) > 15 else ''))
    if _run_method('EU-RAC'):
        # ---- EU-RAC (GCN neural policy: EURAC) ----
        _t0 = time.time()
        print(f'    EU-RAC value: Parameterized V_phi')
        print(f'    EU-RAC actor: GPG-style GCN policy + conservative KL anchor')
        print(f'    Warm start: Dijkstra')
        print(f'    DOT: Disabled')
        print(f'    Actor update: lr={EURAC_LR_ACTOR:g}, entropy={EURAC_ENTROPY_COEF:g}, '
              f'KL={EURAC_KL_COEF:g}')
        random.seed(combo_seed)
        np.random.seed(combo_seed)
        torch.manual_seed(combo_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(combo_seed)
        agent = EURAC(
            env=env,
            lr_e=EURAC_LR_E,
            lr_d=EURAC_LR_D,
            lr_actor=EURAC_LR_ACTOR,
            entropy_coef=EURAC_ENTROPY_COEF,
            kl_coef=EURAC_KL_COEF,
            auto_checkpoint_interval=10000,
            auto_resume=not force_eurac,
        )
        if force_eurac and agent._auto_checkpoint_path.exists():
            agent._auto_checkpoint_path.unlink()
            print(f'    [EU-RAC] --force: removed checkpoint {agent._auto_checkpoint_path}')

        # Fixed validation stream; deployment uses the same 2% shortest-path safety bound.
        _rev = {}
        for (u, v), (mean_t, _) in env.edges.items():
            _rev.setdefault(v, []).append((u, max(1, int(round(mean_t)))))
        _to_go, _next_hop, _pq = {dest: 0}, {}, [(0, dest)]
        while _pq:
            _dist, _node = heapq.heappop(_pq)
            if _dist != _to_go.get(_node): continue
            for _pred, _cost in _rev.get(_node, []):
                _cand = _dist + _cost
                if _cand < _to_go.get(_pred, float('inf')):
                    _to_go[_pred] = _cand; _next_hop[_pred] = _node
                    heapq.heappush(_pq, (_cand, _pred))
        _validation_best_path = agent._auto_checkpoint_path.with_name(agent._auto_checkpoint_path.stem + '_validation_best.pt')
        _validation_best = {'mc': -float('inf')}
        def _safe_policy(node, bud):
            pol, acts = agent.get_policy((node, bud))
            valid = [a for a in acts if (node, a) in env.edges]
            limit = _to_go.get(node, float('inf')) * (1.0 + EURAC_SAFETY_DETOUR_TOLERANCE)
            safe = [a for a in valid if max(1, int(round(env.edges[(node, a)][0]))) + _to_go.get(a, float('inf')) <= limit]
            choices = safe or valid
            return max(choices, key=lambda a: pol.get(a, 0.0)) if choices else _next_hop.get(node)
        def _validate_and_keep(episode):
            val_env = ChicagoEnv(origin=origin, dest=dest, budget=budget, exec_prob=exec_prob,
                deterministic=False, seed=combo_seed + 700000 + int(episode),
                uncertain_ratio=GLOBAL_UNCERTAIN_RATIO, uncertain_seed=GLOBAL_UNCERTAIN_SEED,
                uncertain_mode='random', top_k=6, uncertain_edges_explicit=uncertain_edges_explicit,
                use_od_uncertainty=False if uncertain_edges_explicit is not None else CHICAGO_USE_OD_UNCERTAINTY,
                od_target_uncertain_ratio=CHICAGO_TARGET_UNCERTAIN_RATIO,
                od_max_uncertain_ratio=CHICAGO_MAX_UNCERTAIN_RATIO,
                od_uncertain_seed=CHICAGO_OD_UNCERTAIN_SEED)
            value = evaluate_policy(val_env, _safe_policy, episodes=EURAC_VALIDATION_EPISODES)
            if value > _validation_best['mc']:
                _validation_best['mc'] = value
                agent.save_checkpoint(_validation_best_path, episode=episode, n_episodes=EURAC_EPISODES)
            print(f'    [EU-RAC] validation episode={episode} MC={value:.4f} best={_validation_best["mc"]:.4f}')
        agent.warm_start(logit=EURAC_WARM_LOGIT)

        # --- Dijkstra agreement BEFORE training ---
        _agree_before = agent.compute_dijkstra_agreement()
        agent._diag_dijkstra_agreement_before = _agree_before
        if agent._resumed_from_checkpoint:
            print(f'    [EU-RAC] Dijkstra agreement (resumed): {_agree_before:.3f}')
        else:
            print(f'    [EU-RAC] Dijkstra agreement (warm-start only): {_agree_before:.3f}')
            _validate_and_keep(0)

        start_episode = getattr(agent, '_episodes_trained', 0)
        if agent._resumed_from_checkpoint:
            _validate_and_keep(start_episode)
        if start_episode == 0:
            np.random.seed(combo_seed)
        remaining_episodes = max(0, EURAC_EPISODES - start_episode)
        if start_episode > 0:
            print(f'    [EU-RAC] strict resume: {start_episode}/{EURAC_EPISODES} episodes already complete; '
                  f'running {remaining_episodes} remaining episodes')

        # Training with periodic progress: log every 10% of total episodes
        _progress_interval = max(1, EURAC_EPISODES // 10)
        for ep in range(start_episode + 1, EURAC_EPISODES + 1):
            agent.run_episode(train=True)
            if ep % EURAC_VALIDATION_INTERVAL == 0 or ep == EURAC_EPISODES:
                _validate_and_keep(ep)
            if ep % _progress_interval == 0 or ep == EURAC_EPISODES:
                _elapsed = time.time() - _t0
                _done_now = ep - start_episode
                _eps_per_s = _done_now / _elapsed if _elapsed > 0 else 0
                print(f'    EU-RAC progress: {ep}/{EURAC_EPISODES} episodes, '
                      f'{_elapsed:.0f}s elapsed this run, {_eps_per_s:.0f} eps/s')
        if _validation_best['mc'] > -float('inf') and _validation_best_path.exists():
            agent.load_checkpoint(_validation_best_path)
            print(f'    [EU-RAC] evaluating validation-best checkpoint: {_validation_best_path.name}')
        _t_algo = time.time() - _t0
        qe_avg, qd_avg = agent.q_stats()
        _episodes_run = max(1, EURAC_EPISODES - start_episode)
        _avg_eps_time = _t_algo / _episodes_run * 1000  # ms per episode run now

        # --- Dijkstra agreement AFTER training ---
        _agree_after = agent.compute_dijkstra_agreement()
        agent._diag_dijkstra_agreement_after = _agree_after

        # --- Greedy path ---
        _eu_greedy_path = agent.get_greedy_path()
        _eu_reaches = _eu_greedy_path[-1] == dest if _eu_greedy_path else False

        # --- Diagnostics ---
        _eu_diag = agent.get_diagnostics()

        print(f'  EU-RAC: algo={_t_algo:.1f}s ({_avg_eps_time:.1f}ms/ep)  |  '
              f'Q_e: {len(agent.Q_e)}  Q_d: {len(agent.Q_d)}  '
              f'avg Q_e: {qe_avg:.3f}  avg Q_d: {qd_avg:.3f}')
        print(f'    GCN Actor params: {_eu_diag.get("n_params", 0)}  |  '
              f'policy logits mu={_eu_diag.get("logit_mean", float("nan")):.4f} '
              f'sig={_eu_diag.get("logit_std", float("nan")):.4f}')
        print(f'    Advantage mu={_eu_diag.get("adv_mean", float("nan")):.4f} '
              f'sig={_eu_diag.get("adv_std", float("nan")):.4f}  |  '
              f'Actor entropy={_eu_diag.get("entropy_mean", float("nan")):.4f}  |  '
              f'avg max prob={_eu_diag.get("max_prob_mean", float("nan")):.4f}')
        print(f'    KL current||warm={_eu_diag.get("kl_cur_ref_mean", float("nan")):.6f}  |  '
              f'KL warm||current={_eu_diag.get("kl_ref_cur_mean", float("nan")):.6f}  |  '
              f'actor drift={_eu_diag.get("actor_parameter_drift", float("nan")):.6f}')
        print(f'    Dijkstra agree: {_agree_before:.3f} -> {_agree_after:.3f}  |  '
              f'greedy reaches dest: {_eu_reaches}  |  '
              'greedy path: ' + '->'.join(str(n) for n in _eu_greedy_path[:15])
              + ('...' if len(_eu_greedy_path) > 15 else ''))

        _eurac_deploy_stats = {
            'total_decisions': 0,
            'cycle_filtered': 0,
            'argmax_removed': 0,
            'all_visited_fallback': 0,
            'safety_filtered': 0,
            'safety_fallback': 0,
            'selected_actions': {},
        }
        _eurac_deploy_state = {
            'visited': set(),
            'last_budget': None,
            'started': False,
        }

        def eurac_policy(node, bud):
            if node == origin and bud == budget:
                _eurac_deploy_state['visited'] = {node}
                _eurac_deploy_state['started'] = True
            elif (not _eurac_deploy_state['started'] or
                  (_eurac_deploy_state['last_budget'] is not None and
                   bud > _eurac_deploy_state['last_budget'])):
                _eurac_deploy_state['visited'] = {node}
                _eurac_deploy_state['started'] = True
            else:
                _eurac_deploy_state['visited'].add(node)
            _eurac_deploy_state['last_budget'] = bud

            s = (node, bud)
            pol, acts = agent.get_policy(s)
            if not acts:
                return None

            valid_acts = [a for a in acts if a in successors.get(node, [])]
            if not valid_acts:
                return None
            original_argmax = max(valid_acts, key=lambda a: pol.get(a, 0.0))
            safe_limit = _to_go.get(node, float('inf')) * (1.0 + EURAC_SAFETY_DETOUR_TOLERANCE)
            safe_acts = [a for a in valid_acts if max(1, int(round(env.edges[(node, a)][0]))) + _to_go.get(a, float('inf')) <= safe_limit]
            if safe_acts:
                ranked = sorted(safe_acts, key=lambda a: pol.get(a, 0.0), reverse=True)
                if original_argmax not in safe_acts:
                    _eurac_deploy_stats['safety_filtered'] += 1
            else:
                ranked = [original_argmax]
                _eurac_deploy_stats['safety_fallback'] += 1
            selected = None
            for action in ranked:
                if action not in _eurac_deploy_state['visited']:
                    selected = action
                    break

            if selected is None:
                selected = original_argmax
                _eurac_deploy_stats['all_visited_fallback'] += 1
            elif selected != original_argmax:
                _eurac_deploy_stats['cycle_filtered'] += 1
                _eurac_deploy_stats['argmax_removed'] += 1

            _eurac_deploy_stats['total_decisions'] += 1
            _eurac_deploy_stats['selected_actions'][selected] = (
                _eurac_deploy_stats['selected_actions'].get(selected, 0) + 1
            )
            return selected
        results['EU-RAC'] = _mc_eval(eurac_policy, 'EU-RAC')
        print(f'    final MC: {results["EU-RAC"]:.4f}')
        _eu_total = _eurac_deploy_stats['total_decisions']
        _eu_filtered = _eurac_deploy_stats['cycle_filtered']
        _eu_rate = (_eu_filtered / _eu_total) if _eu_total else 0.0
        _eu_selected_top = sorted(
            _eurac_deploy_stats['selected_actions'].items(),
            key=lambda kv: kv[1], reverse=True
        )[:5]
        print(f'    EU-RAC deployment: total decisions={_eu_total}  |  '
              f'cycle filtered={_eu_filtered}  |  filter rate={_eu_rate:.2%}')
        print(f'    EU-RAC deployment: original argmax removed='
              f'{_eurac_deploy_stats["argmax_removed"]}  |  '
              f'all-visited fallback={_eurac_deploy_stats["all_visited_fallback"]}  |  '
              f'top selected actions={_eu_selected_top}')
        print(f'    training success rate (last 1000 eps): {_eu_diag.get("train_success_rate", float("nan")):.4f}')

    return results


# ============================================================
# Run one OD pair (with resume support)
# ============================================================
def _read_existing_od_csv(csv_path, selected_methods=None):
    """Read existing per-OD CSV, return dict {(B_label, eta): row_dict} or None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, na_values=['NA'])
        selected = set(selected_methods or METHODS)
        existing = {}
        for _, row in df.iterrows():
            key = (str(row['B_label']), str(row['eta']))
            vals = {}
            complete_for_selection = True
            for m in METHODS:
                v = row.get(m, None)
                if v is None or pd.isna(v):
                    vals[m] = np.nan
                    if m in selected:
                        complete_for_selection = False
                else:
                    vals[m] = float(v)
            if complete_for_selection:
                existing[key] = vals
        return existing
    except Exception:
        return None


def _read_existing_values(csv_path):
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, na_values=['NA'])
        existing = {}
        for _, row in df.iterrows():
            key = (str(row['B_label']), str(row['eta']))
            vals = {}
            for m in METHODS:
                v = row.get(m, None)
                vals[m] = np.nan if v is None or pd.isna(v) else float(v)
            existing[key] = vals
        return existing
    except Exception:
        return {}


def run_one_od(origin, dest, force=False, selected_methods=None, uncertainty_envs=None):
    """Run one OD pair and save per-OD CSV."""
    uncertainty_envs = uncertainty_envs or {}
    uncertainty_info = uncertainty_envs.get(_od_key(origin, dest))
    explicit_uncertainty = None
    if uncertainty_info is not None:
        explicit_uncertainty = uncertainty_info.get('uncertain_edges_explicit') or None

    t_let = _dijkstra_mean(origin, dest)
    min_budget = _dijkstra_rounded(origin, dest)
    csv_path = OUTPUT_DIR / f'od_{origin}_{dest}.csv'

    print()
    print(f'{"=" * 60}')
    print(f'  OD {origin}->{dest}  |  t_LET = {t_let:.2f} h  |  min budget = {min_budget:.0f}')
    if explicit_uncertainty is not None:
        print(f'  uncertainty: explicit from {DEFAULT_UNCERTAINTY_FILE.name}: {explicit_uncertainty}')
    else:
        print(f'  uncertainty: global random {GLOBAL_UNCERTAIN_RATIO*100:.0f}% nodes (seed={GLOBAL_UNCERTAIN_SEED})')
        if CHICAGO_USE_OD_UNCERTAINTY:
            print(
                f'  OD calibration: target={CHICAGO_TARGET_UNCERTAIN_RATIO:.0%}, '
                f'max={CHICAGO_MAX_UNCERTAIN_RATIO:.0%}, seed={CHICAGO_OD_UNCERTAIN_SEED}'
            )
    print(f'{"=" * 60}')

    # Check existing data
    previous_values = {} if force else _read_existing_values(csv_path)
    existing = None if force else _read_existing_od_csv(csv_path, selected_methods)

    if existing is not None:
        completed = set(existing.keys())
        print(f'  existing {len(completed)}/9 configs found; skipping completed configs')
    else:
        completed = set()

    # Build full task list
    all_tasks = []
    for coeff in B_COEFFS:
        budget = int(math.floor(coeff * min_budget))
        for eta in ETA_LIST:
            key = (str(coeff), str(eta))
            if key not in completed:
                all_tasks.append((coeff, budget, eta, key))

    if not all_tasks:
        print('  all configs already completed; skipping')
        rows = []
        for coeff in B_COEFFS:
            for eta in ETA_LIST:
                key = (str(coeff), str(eta))
                row = {'B_label': key[0], 'eta': key[1]}
                for m in METHODS:
                    row[m] = existing[key][m]
                rows.append(row)
        return rows

    print(f'  need to run {len(all_tasks)}/{len(B_COEFFS) * len(ETA_LIST)} configs')

    rows = []

    for coeff, budget, eta, key in all_tasks:
        exec_prob = 1.0 - eta
        combo_seed = SEED + origin * 100 + dest + int(eta * 100) * 10000 + budget

        print()
        print(f'--- B={coeff} (budget={budget}), eta={eta} (exec_prob={exec_prob}) ---')

        env = ChicagoEnv(
            origin=origin, dest=dest, budget=budget,
            exec_prob=exec_prob, deterministic=False, seed=combo_seed,
            uncertain_ratio=GLOBAL_UNCERTAIN_RATIO,
            uncertain_seed=GLOBAL_UNCERTAIN_SEED,
            uncertain_mode='random', top_k=6,
            uncertain_edges_explicit=explicit_uncertainty,
            use_od_uncertainty=False if explicit_uncertainty is not None else CHICAGO_USE_OD_UNCERTAINTY,
            od_target_uncertain_ratio=CHICAGO_TARGET_UNCERTAIN_RATIO,
            od_max_uncertain_ratio=CHICAGO_MAX_UNCERTAIN_RATIO,
            od_uncertain_seed=CHICAGO_OD_UNCERTAIN_SEED,
        )
        od_report = getattr(env, 'od_uncertainty_report', None)
        if od_report is not None:
            print(
                '  LET exposure: '
                f'{od_report["risky_transitions"]}/{max(0, od_report["path_length"] - 1)} '
                f'({od_report["coverage_ratio"]:.1%}), '
                f'added={od_report["added_transitions"]}'
            )

        t0 = time.time()
        mc_vals = run_all_methods(env, origin, dest, budget, combo_seed, selected_methods=selected_methods, uncertain_edges_explicit=explicit_uncertainty, force_eurac=force)
        elapsed = time.time() - t0

        row = {'B_label': str(coeff), 'eta': str(eta)}
        old_vals = previous_values.get(key, {})
        for m in METHODS:
            row[m] = mc_vals[m] if m in mc_vals else old_vals.get(m, None)
        rows.append(row)

        # Print progress
        parts = [f"  {m}={fmt_progress(mc_vals.get(m))}" for m in METHODS[:6]]
        print(' '.join(parts))
        parts2 = [f"  {m}={fmt_progress(mc_vals.get(m))}" for m in METHODS[6:]]
        print(' '.join(parts2))
        print(f'  [{elapsed:.0f}s]')

    # Merge with existing rows if any
    if existing is not None:
        for coeff in B_COEFFS:
            for eta in ETA_LIST:
                key = (str(coeff), str(eta))
                if key in completed:
                    row = {'B_label': key[0], 'eta': key[1]}
                    for m in METHODS:
                        row[m] = existing[key][m]
                    rows.append(row)

    # Sort rows: B_label descending, eta ascending
    rows.sort(key=lambda r: (-float(r['B_label']), float(r['eta'])))

    # ---- Save per-OD CSV ----
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['B_label', 'eta'] + METHODS
        writer.writerow(header)
        for row in rows:
            writer.writerow([row['B_label'], row['eta']] +
                            [fmt3(row[m]) for m in METHODS])
    print()
    print(f'Saved: {csv_path}')

    return rows


# ============================================================
# Aggregate: scan od_*.csv, compute means, mark best/second
# ============================================================
def aggregate_only():
    """Scan od_*.csv, compute mean across ODs, save final_table.csv."""
    csv_files = sorted(glob.glob(str(OUTPUT_DIR / 'od_*.csv')))
    if not csv_files:
        print('No od_*.csv files found.')
        return

    print(f'Found {len(csv_files)} OD CSV files:')
    for f in csv_files:
        print(f'  {Path(f).name}')

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f, na_values=['NA'])
        for m in METHODS:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors='coerce')
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    group_cols = ['B_label', 'eta']
    mean_df = combined.groupby(group_cols)[METHODS].mean().reset_index()

    for m in METHODS:
        mean_df[m] = mean_df[m].round(3)

    out_rows = []
    for _, row in mean_df.iterrows():
        out_row = {'B_label': row['B_label'], 'eta': row['eta']}
        vals = {m: row[m] for m in METHODS if not pd.isna(row[m])}
        sorted_methods = sorted(vals, key=vals.get, reverse=True)
        best_m = sorted_methods[0] if sorted_methods else None
        second_m = sorted_methods[1] if len(sorted_methods) > 1 else None

        for m in METHODS:
            v = row[m]
            if pd.isna(v):
                out_row[m] = 'NA'
            elif m == best_m:
                out_row[m] = f"*{v:.3f}*"
            elif m == second_m:
                out_row[m] = f"#{v:.3f}#"
            else:
                out_row[m] = f"{v:.3f}"
        out_rows.append(out_row)

    csv_path = OUTPUT_DIR / 'final_table.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['B_label', 'eta'] + METHODS
        writer.writerow(header)
        for row in out_rows:
            writer.writerow([row['B_label'], row['eta']] +
                            [row[m] for m in METHODS])
    print(f'\nSaved: {csv_path}')

    # Print Markdown table
    print('\n' + '=' * 80)
    print('  Final Mean Table')
    print('=' * 80)
    header_md = '| B_label | eta | ' + ' | '.join(METHODS) + ' |'
    sep_md = '|:---:|:---:|' + ':---:|' * len(METHODS)
    print(header_md)
    print(sep_md)
    for row in out_rows:
        cells = [str(row['B_label']), str(row['eta'])] + [row[m] for m in METHODS]
        print('| ' + ' | '.join(cells) + ' |')


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='Chicago Sketch Batch OD experiments')
    p.add_argument('--run-all', action='store_true',
                   help='Run all OD pairs from --od-source')
    p.add_argument('--od', type=int, nargs=2, metavar=('ORIGIN', 'DEST'),
                   help='Run a single debug OD pair')
    p.add_argument('--od-source', type=Path, default=DEFAULT_OD_SOURCE,
                   help='CSV file containing final OD candidates')
    p.add_argument('--uncertainty-file', type=Path, default=DEFAULT_UNCERTAINTY_FILE,
                   help='JSON file containing explicit uncertain edges by OD')
    p.add_argument('--aggregate-only', action='store_true',
                   help='Only re-compute mean from existing od_*.csv files')
    p.add_argument('--force', action='store_true',
                   help='Force re-run all configs, ignore existing per-OD CSV')
    p.add_argument('--algorithms', nargs='+', default=['all'],
                   help='Algorithms to run, e.g. --algorithms eurac segac, --algorithms dot pulse ilp, or --algorithms all')
    return p.parse_args()


def main():
    args = parse_args()
    selected_methods = normalize_algorithms(args.algorithms)

    print(f'[CS_mean] Method implementations:')
    print(f'  EU-RAC:   V_phi + GPG-style GCN policy - eu_rac.py')
    print(f'  SEGAC:    Dijkstra BC init + GPG replay - segac_routing2.py')
    print(f'  PQL:      warm-start tabular Q-learning - pql_routing.py')
    print(f'  DAC:      {DAC_DEPLOYMENT_NAME} - dac_routing2.py')
    print(f'  GE-DDRL:  Chicago-adapted tabular C51 v3 - ge_ddrl_routing3.py')
    print('  Selected: ' + ', '.join(selected_methods))

    if args.aggregate_only:
        aggregate_only()
        return

    uncertainty_envs = load_uncertainty_environments(args.uncertainty_file)

    if args.od:
        origin, dest = args.od
        run_one_od(origin, dest, force=args.force, selected_methods=selected_methods, uncertainty_envs=uncertainty_envs)
        return

    od_pairs = load_final_od_pairs(args.od_source)
    if not od_pairs:
        print(f'[CS_mean] no OD pairs found in {args.od_source}')
        return

    for origin, dest in od_pairs:
        run_one_od(origin, dest, force=args.force, selected_methods=selected_methods, uncertainty_envs=uncertainty_envs)
    aggregate_only()


if __name__ == '__main__':
    main()








