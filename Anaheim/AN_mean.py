"""Anaheim experiment runner for selected OD pairs.

Usage:
    python AN_mean.py --run-all
    python AN_mean.py --od 37 200
    python AN_mean.py --aggregate-only
"""


OD_CONFIGS = {
    (22, 260): {},
    (22, 268): {},
    (120, 12): {},
    (168, 14): {},
    (168, 78): {},
    (172, 15): {},
    (401, 13): {},
    (405, 66): {},
    (408, 14): {},
    (411, 70): {},
    (37, 13): {},
    (120, 41): {},
    (209, 139): {},
    (21, 270): {},
    (409, 254): {},
    (414, 260): {},
    (401, 262): {},
    (395, 16): {},
    (415, 268): {},
    (210, 256): {},
}
ETA_LIST = [0, 0.2, 0.4]          
B_COEFFS = [0.975, 1.0, 1.025]  
TIME_STEP = 0.25


GLOBAL_UNCERTAIN_RATIO = 0.3
GLOBAL_UNCERTAIN_SEED = 42
USE_OD_SPECIFIC_UNCERTAINTY = True
METHODS = ['EU-RAC', 'Pulse', 'DOT', 'Robust', 'GP3', 'ILP', 'OTAP',
           'SEGAC', 'PQL', 'DAC', 'GE-DDRL']

METHOD_ALIASES = {
    'eurac': 'EU-RAC', 'eu-rac': 'EU-RAC', 'eurac': 'EU-RAC', 'eu-rac': 'EU-RAC',
    'pulse': 'Pulse', 'dot': 'DOT', 'robust': 'Robust',
    'gp3': 'GP3', 'ilp': 'ILP', 'otap': 'OTAP',
    'segac': 'SEGAC', 'pql': 'PQL', 'dac': 'DAC',
    'ge-ddrl': 'GE-DDRL', 'geddrl': 'GE-DDRL', 'ge_ddrl': 'GE-DDRL',
}

ILP_META_COLUMNS = ['ILP_status', 'ILP_runtime']
SEGAC_META_COLUMNS = [
    'SEGAC_warm_start', 'warm_start_steps', 'warm_start_agreement',
    'warm_start_loss_initial', 'warm_start_loss_final', 'SEGAC_runtime'
]

# ============================================================
import sys
import time
import heapq
import glob
import math
import argparse
import csv
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd

# Local Anaheim modules
import importlib.util
_ROOT_DIR = Path(__file__).resolve().parent.parent
_EURAC_SPEC = importlib.util.spec_from_file_location("_public_eu_rac", _ROOT_DIR / "eu_rac.py")
_EURAC_MODULE = importlib.util.module_from_spec(_EURAC_SPEC)
_EURAC_SPEC.loader.exec_module(_EURAC_MODULE)
EURAC = _EURAC_MODULE.EURAC
from anaheim_env import AnaheimEnv
from robust_routing import run_robust_routing, extract_path as rr_extract_path
from dot_routing import run_dot_routing
from pulse_routing import run_pulse_routing
from otap_routing import run_otap_routing
from ilp_routing import run_ilp_routing
from gp3_routing import run_gp3_routing
from pql_routing import run_pql_routing
from ge_ddrl_routing3 import run_ge_ddrl_routing
from segac_routing2 import run_segac_routing
from dac_routing2 import run_dac_routing

from evaluator import evaluate_policy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Dijkstra
# ============================================================
_NET_DIR = Path(__file__).parent.parent / 'network' / 'Anaheim'
_net_df = pd.read_csv(_NET_DIR / 'Anaheim_network.csv')

_mean_edges = {}
_successors = {}
for _, row in _net_df.iterrows():
    u, v = int(row['From']), int(row['To'])
    _mean_edges[(u, v)] = float(row['Cost'])   # hours
    _successors.setdefault(u, []).append(v)


UNCERTAINTY_CONFIG_PATH = Path(__file__).with_name('uncertain_edges_v3_pruned.json')


def _od_key(origin, dest):
    return f'{int(origin)}_{int(dest)}'


def _load_od_uncertainty_configs():
    if not UNCERTAINTY_CONFIG_PATH.exists():
        return {}
    with open(UNCERTAINTY_CONFIG_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    configs = {}
    for key, entries in raw.items():
        configs[key] = {int(e['node']): int(e['intended']) for e in entries}
    return configs


OD_UNCERTAIN_EDGES = _load_od_uncertainty_configs()


def _uncertainty_edges_for(origin, dest, eta):
    if not USE_OD_SPECIFIC_UNCERTAINTY or eta <= 0:
        return None
    return OD_UNCERTAIN_EDGES.get(_od_key(origin, dest))


def _dijkstra_mean(src, dst):
    """t_LET in minutes (sum of raw mean travel times along shortest path)."""
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
            nd = d + max(1, round(_mean_edges[(u, v)] / TIME_STEP))
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return float('inf')


def _dijkstra_next_hop(dst):
    """Precompute Dijkstra next-hop from every node toward dst (dt units)."""
    rev = {}
    for (u, v), cost in _mean_edges.items():
        rev.setdefault(v, []).append((u, max(1, round(cost / TIME_STEP))))
    dist = {dst: 0.0}
    next_hop = {}
    pq = [(0.0, dst)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for pred, c in rev.get(u, []):
            nd = d + c
            if nd < dist.get(pred, float('inf')):
                dist[pred] = nd
                next_hop[pred] = u
                heapq.heappush(pq, (nd, pred))
    return next_hop


EVAL_EPISODES = 10000
SEED = 42

EURAC_LR_E = 0.2
EURAC_LR_D = 0.2
EURAC_LR_ACTOR = 1e-4
EURAC_ENTROPY_COEF = 0.01
EURAC_KL_COEF = 0.1
EURAC_WARM_LOGIT = 2.0

EURAC_EPISODES = 10000
EURAC_SELECTION_INTERVAL = 1000
EURAC_SELECTION_EVAL_EPISODES = 1000
PQL_EPISODES = 50000
GE_EPISODES = 50000
SEGAC_EPISODES = 50000
DAC_EPISODES = 50000
FALLBACK_Q_GAP_THRESHOLD = 0.01
DAC_DEPLOYMENT_NAME = 'DAC + Dijkstra Safety Fallback'

RESULT_ROOT = Path(__file__).parent / 'results' / 'dt025_seedfixed_segac_warmstart'
CSV_DIR = RESULT_ROOT / 'csv'
PATHS_DIR = RESULT_ROOT / 'paths'
LOGS_DIR = RESULT_ROOT / 'logs'
CONFIG_DIR = RESULT_ROOT / 'config'
OUTPUT_DIR = CSV_DIR
CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
CHECKPOINT_INTERVAL = 10000  # save EU-RAC checkpoint every N episodes
for _out_dir in (CSV_DIR, PATHS_DIR, LOGS_DIR, CONFIG_DIR):
    _out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def mc_eval(env, policy_func):
    return evaluate_policy(env, policy_func, episodes=EVAL_EPISODES)


def fmt3(v):
    """Format float to 3 decimal places, NA for missing."""
    if v is None: return 'NA'
    try:
        if pd.isna(v): return 'NA'
    except TypeError: pass
    return f'{float(v):.3f}'


def _write_od_csv(csv_path, rows):
    rows = sorted(rows, key=lambda r: (-float(r['B_label']), float(r['eta'])))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['B_label', 'eta'] + METHODS + ILP_META_COLUMNS + SEGAC_META_COLUMNS
        writer.writerow(header)
        for row in rows:
            writer.writerow([row['B_label'], row['eta']] +
                            [fmt3(row[m]) for m in METHODS] +
                            [row.get('ILP_status', ''),
                             fmt3(float(row.get('ILP_runtime', 0.0) or 0.0))] +
                            [str(row.get('SEGAC_warm_start', '')),
                             row.get('warm_start_steps', ''),
                             fmt3(float(row.get('warm_start_agreement', 0.0) or 0.0)),
                             fmt3(float(row.get('warm_start_loss_initial', 0.0) or 0.0)),
                             fmt3(float(row.get('warm_start_loss_final', 0.0) or 0.0)),
                             fmt3(float(row.get('SEGAC_runtime', 0.0) or 0.0))])


def _task_key(coeff, eta):
    return (round(float(coeff), 6), round(float(eta), 6))


# ============================================================
# Run all methods for one (origin, dest, budget, exec_prob)
# ============================================================
# Run all methods for one (origin, dest, budget, exec_prob)
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


def run_all_methods(env, origin, dest, budget, combo_seed, selected_methods=None, static_plan_cache=None, coeff=None, eta=None):
    """Returns dict[method_name] = mc_value (float).

    Static baselines do not use eta/execution uncertainty during planning, so
    their paths are cached once per budget and re-evaluated for each eta.
    """
    results = {}
    _selected = set(selected_methods or METHODS)
    successors = env.successors
    exec_prob = env.exec_prob
    if static_plan_cache is None:
        static_plan_cache = {}
    log_path = LOGS_DIR / f"od_{origin}_{dest}.log"

    def _run(name, fn):
        if name not in _selected:
            return None
        _diag(f"[START] {name}")
        t0 = time.time()
        try:
            val = fn()
        except Exception as e:
            _diag(f"[FAILED] {name}: {e}")
            raise
        _diag(f"[DONE] {name} | {time.time()-t0:.1f}s")
        return val

    def _diag(msg):
        print(msg, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    # ---- MC evaluation: fresh env with FIXED seed for every method ----
    MC_EVAL_SEED = 999999 + origin * 100 + dest

    def _make_mc_env():
        """Create a fresh, independent env for MC evaluation."""
        uncertain_edges = dict(getattr(env, 'uncertain_edges', {})) if exec_prob < 1.0 else None
        return AnaheimEnv(
            origin=origin, dest=dest, budget=budget,
            exec_prob=exec_prob, deterministic=False, seed=MC_EVAL_SEED,
            uncertain_ratio=0.0 if uncertain_edges else (GLOBAL_UNCERTAIN_RATIO if exec_prob < 1.0 else 0.0),
            uncertain_seed=GLOBAL_UNCERTAIN_SEED,
            uncertain_mode='random', top_k=6,
            time_step=TIME_STEP,
            uncertain_edges_explicit=uncertain_edges,
        )
    def _mc_eval_n(policy_func, episodes):
        """MC evaluation on a fresh env - all methods see the same random sequence."""
        return evaluate_policy(_make_mc_env(), policy_func, episodes=episodes)

    def _mc_eval(policy_func):
        return _mc_eval_n(policy_func, EVAL_EPISODES)
    def _eval_path(path):
        pm = {path[i]: path[i + 1] for i in range(len(path) - 1)}
        return _mc_eval(lambda n, b: pm.get(n))

    def _eval_static_path(name, path):
        _diag(f"[MC START] {name}")
        t_eval = time.time()
        value = _eval_path(path)
        _diag(f"[MC DONE] {name} | runtime={time.time() - t_eval:.3f} s")
        return value

    # ---- eta-invariant static planners: compute once per budget ----
    static_paths = static_plan_cache.setdefault(budget, {})

    def _pulse():
        if 'Pulse' not in static_paths:
            _diag(f"[CACHE MISS] Pulse {budget}")
            t_plan = time.time()
            static_paths['Pulse'] = run_pulse_routing(env, budget=budget)['path']
            _diag(f"[PLANNER DONE] Pulse | runtime={time.time() - t_plan:.3f} s")
        else:
            _diag(f"[CACHE HIT] Pulse {budget}")
        return _eval_static_path('Pulse', static_paths['Pulse'])
    results['Pulse'] = _run('Pulse', _pulse)

    def _dot():
        if 'DOT' not in static_paths:
            _diag(f"[CACHE MISS] DOT {budget}")
            t_plan = time.time()
            static_paths['DOT'] = run_dot_routing(env, budget=budget)
            _diag(f"[PLANNER DONE] DOT | runtime={time.time() - t_plan:.3f} s")
        else:
            _diag(f"[CACHE HIT] DOT {budget}")
        _diag(f"[MC START] DOT")
        t_eval = time.time()
        value = _mc_eval(static_paths['DOT']['policy'])
        _diag(f"[MC DONE] DOT | runtime={time.time() - t_eval:.3f} s")
        return value
    results['DOT'] = _run('DOT', _dot)

    def _robust():
        if 'Robust' not in static_paths:
            _diag(f"[CACHE MISS] Robust {budget}")
            t_plan = time.time()
            static_paths['Robust'] = run_robust_routing(env, budget=budget, psi=0.9, m=2)
            _diag(f"[PLANNER DONE] Robust | runtime={time.time() - t_plan:.3f} s")
        else:
            _diag(f"[CACHE HIT] Robust {budget}")
        _diag(f"[MC START] Robust")
        t_eval = time.time()
        value = _mc_eval(static_paths['Robust']['policy'])
        _diag(f"[MC DONE] Robust | runtime={time.time() - t_eval:.3f} s")
        return value
    results['Robust'] = _run('Robust', _robust)

    def _gp3():
        if 'GP3' not in static_paths:
            _diag(f"[CACHE MISS] GP3 {budget}")
            t_plan = time.time()
            static_paths['GP3'] = run_gp3_routing(
                env, budget=budget, zeta_min=0.0, zeta_max=50.0
            )['path']
            _diag(f"[PLANNER DONE] GP3 | runtime={time.time() - t_plan:.3f} s")
        else:
            _diag(f"[CACHE HIT] GP3 {budget}")
        return _eval_static_path('GP3', static_paths['GP3'])
    results['GP3'] = _run('GP3', _gp3)

    def _ilp():
        if 'ILP' not in static_paths:
            _diag(f"[CACHE MISS] ILP {budget}")
            t_plan = time.time()
            ilp_result = run_ilp_routing(env, budget=budget, K=100)
            static_paths['ILP'] = ilp_result['path']
            static_paths['ILP_status'] = ilp_result.get('status', 'unknown')
            static_paths['ILP_runtime'] = ilp_result.get('runtime', time.time() - t_plan)
            _diag(f"[PLANNER DONE] ILP | runtime={static_paths['ILP_runtime']:.3f} s | status={static_paths['ILP_status']}")
        else:
            _diag(f"[CACHE HIT] ILP {budget}")
        results['ILP_status'] = static_paths.get('ILP_status', '')
        results['ILP_runtime'] = static_paths.get('ILP_runtime', 0.0)
        return _eval_static_path('ILP', static_paths['ILP'])
    results['ILP'] = _run('ILP', _ilp)

    def _otap():
        if 'OTAP' not in static_paths:
            _diag(f"[CACHE MISS] OTAP {budget}")
            t_plan = time.time()
            static_paths['OTAP'] = run_otap_routing(env, budget=budget, K=30)['path']
            _diag(f"[PLANNER DONE] OTAP | runtime={time.time() - t_plan:.3f} s")
        else:
            _diag(f"[CACHE HIT] OTAP {budget}")
        return _eval_static_path('OTAP', static_paths['OTAP'])
    results['OTAP'] = _run('OTAP', _otap)

    # ---- SEGAC ----
    def _segac():
        t_segac = time.time()
        r = run_segac_routing(
            env, budget=budget, episodes=SEGAC_EPISODES,
            lr_actor=0.01, lr_critic=0.1, seed=combo_seed,
        )
        value = _mc_eval(r['policy_func'])
        results['SEGAC_warm_start'] = True
        results['warm_start_steps'] = r.get('expert_traj_count', 0)
        results['warm_start_agreement'] = r.get('bc_agreement', 0.0)
        results['warm_start_loss_initial'] = r.get('bc_loss_start', 0.0)
        results['warm_start_loss_final'] = r.get('bc_loss', 0.0)
        results['SEGAC_runtime'] = time.time() - t_segac
        _diag(
            f"[SEGAC WARM] experts={results['warm_start_steps']} "
            f"agreement={results['warm_start_agreement']:.3f} "
            f"loss={results['warm_start_loss_initial']:.6g}->{results['warm_start_loss_final']:.6g}"
        )
        return value
    results['SEGAC'] = _run('SEGAC', _segac)

    # ---- PQL ----
    def _pql():
        _pql_min_budget = _dijkstra_rounded(origin, dest)
        _pql_warm_budgets = [int(round(c * _pql_min_budget)) for c in B_COEFFS]
        r = run_pql_routing(env, budget=budget, episodes=PQL_EPISODES,
                            alpha=0.1, epsilon=0.1, seed=combo_seed,
                            use_warm_start=True,
                            warm_start_budgets=_pql_warm_budgets)
        q_table = r['q_table']
        _pql_dijkstra_next_hop = r.get('dijkstra_next_hop', {})
        def pql_policy(node, bud):
            acts = successors.get(node, [])
            if not acts:
                return None
            q_node = q_table.get((node, bud), {})
            if q_node:
                return max(acts, key=lambda a: q_node.get(a, 0.0))
            for b_try in range(bud, max(0, bud - 10), -1):
                qn = q_table.get((node, b_try), {})
                if qn:
                    return max(acts, key=lambda a: qn.get(a, 0.0))
            fallback = _pql_dijkstra_next_hop.get(node)
            if fallback is not None and fallback in acts:
                return fallback
            return None
        return _mc_eval(pql_policy)
    results['PQL'] = _run('PQL', _pql)

    # ---- DAC ----
    def _dac():
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
        _dac_deploy = r.get('deployment_fallback_eval') or {}
        return float(_dac_deploy.get('mc', r.get('prob', 0.0)))
    results['DAC'] = _run('DAC', _dac)

    # ---- GE-DDRL ----
    def _ge_ddrl():
        r = run_ge_ddrl_routing(env, budget=budget, N=200, delta_w=1.0,
                                alpha_t=0.05, epsilon=0.1,
                                episodes=GE_EPISODES, seed=combo_seed)
        q_dist = r.get('q_dist', {})
        def ge_policy(node, bud, visited=None):
            acts = [a for a in successors.get(node, []) if (node, a) in q_dist]
            if visited is not None:
                unvisited = [a for a in acts if a not in visited]
                if unvisited:
                    acts = unvisited
            if not acts:
                return None
            idx = max(0, min(199, int(bud)))
            return max(acts, key=lambda a: float(np.sum(q_dist[(node, a)][:idx + 1])))
        return _mc_eval(ge_policy)
    results['GE-DDRL'] = _run('GE-DDRL', _ge_ddrl)

    # ---- EU-RAC ----
    def _eurac():
        agent = EURAC(
            env=env,
            lr_e=EURAC_LR_E,
            lr_d=EURAC_LR_D,
            lr_actor=EURAC_LR_ACTOR,
            entropy_coef=EURAC_ENTROPY_COEF,
            kl_coef=EURAC_KL_COEF,
        )
        #  Checkpoint path 
        ckpt_dir = CHECKPOINT_DIR / f'OD_{origin}_{dest}' / f'B_{coeff if coeff is not None else budget}' / f'eta_{eta if eta is not None else (1.0 - exec_prob):g}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / 'eurac.pt'
        start_ep = 1
        resumed = False

        #  Resume if checkpoint exists 
        if ckpt_path.exists():
            try:
                state = agent.load_checkpoint(ckpt_path)
                start_ep = state.get('episode', 1) + 1
                resumed = True
                print(f'    [EU-RAC] Resumed from checkpoint: episode {start_ep}/{EURAC_EPISODES}')
            except Exception as e:
                print(f'    [EU-RAC] Checkpoint load failed ({e}), starting fresh')

        if not resumed:
            agent.warm_start(logit=EURAC_WARM_LOGIT, alpha=0.7)

        def eurac_policy(node, bud, visited=None):
            return agent.deployment_action(node, bud, visited)

        best_mc = -float('inf')
        best_state = None
        best_ep = None

        def _snapshot_agent():
            return {
                'gcn': copy.deepcopy(agent.gcn.state_dict()),
                'action_scorer': copy.deepcopy(agent.action_scorer.state_dict()),
                'v_net': copy.deepcopy(agent.v_net.state_dict()),
                'v_target': copy.deepcopy(agent.v_target.state_dict()),
                'Q_e': dict(agent.Q_e),
                'Q_d': dict(agent.Q_d),
            }

        def _restore_agent(state):
            agent.gcn.load_state_dict(state['gcn'])
            agent.action_scorer.load_state_dict(state['action_scorer'])
            agent.v_net.load_state_dict(state['v_net'])
            agent.v_target.load_state_dict(state['v_target'])
            agent.Q_e.clear(); agent.Q_e.update(state['Q_e'])
            agent.Q_d.clear(); agent.Q_d.update(state['Q_d'])
            agent._cached_embs = None

        np.random.seed(combo_seed)
        for ep in range(start_ep, EURAC_EPISODES + 1):
            agent.run_episode(train=True)
            if ep % CHECKPOINT_INTERVAL == 0:
                agent.save_checkpoint(ckpt_path, episode=ep)
            if ep % EURAC_SELECTION_INTERVAL == 0 or ep == EURAC_EPISODES:
                val_mc = _mc_eval_n(eurac_policy, EURAC_SELECTION_EVAL_EPISODES)
                _diag(f"[EU-RAC SELECT] ep={ep} val_mc={val_mc:.3f}")
                if val_mc > best_mc:
                    best_mc = val_mc
                    best_ep = ep
                    best_state = _snapshot_agent()

        if best_state is not None:
            _restore_agent(best_state)
            _diag(f"[EU-RAC SELECT] using best_ep={best_ep} best_val_mc={best_mc:.3f}")

        # Final checkpoint stores the selected deployment state.
        agent.save_checkpoint(ckpt_path, episode=best_ep or EURAC_EPISODES)
        return _mc_eval(eurac_policy)
    results['EU-RAC'] = _run('EU-RAC', _eurac)

    return results
# ============================================================
# Run one OD pair (with resume support)
# ============================================================
def _read_existing_od_csv(csv_path, selected_methods=None):
    """Read existing per-OD CSV, return dict {(B_label, eta): row_dict} or None.
    Only returns completed configs where ALL selected methods have values."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, na_values=['NA'])
        selected = set(selected_methods or METHODS)
        existing = {}
        for _, row in df.iterrows():
            key = _task_key(row['B_label'], row['eta'])
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
                for col in ILP_META_COLUMNS + SEGAC_META_COLUMNS:
                    vals[col] = row.get(col, '')
                existing[key] = vals
        return existing
    except Exception:
        return None


def _read_existing_values(csv_path):
    """Read all existing values from CSV (including NA)."""
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, na_values=['NA'])
        existing = {}
        for _, row in df.iterrows():
            key = _task_key(row['B_label'], row['eta'])
            vals = {}
            for m in METHODS:
                v = row.get(m, None)
                vals[m] = np.nan if v is None or pd.isna(v) else float(v)
            existing[key] = vals
        return existing
    except Exception:
        return {}


def run_one_od(origin, dest, force=False, selected_methods=None):
    """Run one OD pair and save per-OD CSV."""
    t_let = _dijkstra_mean(origin, dest)
    min_budget = _dijkstra_rounded(origin, dest)
    csv_path = OUTPUT_DIR / f'od_{origin}_{dest}.csv'

    print(f'\n{"=" * 60}')
    print(f'  OD {origin}->{dest}  |  t_LET = {t_let:.2f} min  |  min budget = {min_budget:.0f}')
    if USE_OD_SPECIFIC_UNCERTAINTY and _od_key(origin, dest) in OD_UNCERTAIN_EDGES:
        print(f'  uncertainty: OD-specific explicit risky edges ({len(OD_UNCERTAIN_EDGES[_od_key(origin, dest)])} edges)')
    else:
        print(f'  uncertainty: random {GLOBAL_UNCERTAIN_RATIO*100:.0f}% nodes (seed={GLOBAL_UNCERTAIN_SEED})')
    print(f'{"=" * 60}')

    # Check existing data
    previous_values = {} if force else _read_existing_values(csv_path)
    existing = None if force else _read_existing_od_csv(csv_path, selected_methods)

    if existing is not None:
        completed = set(existing.keys())
        print(f'  existing rows: {len(completed)}/9, skipping completed tasks')
    else:
        completed = set()

    # Build full task list
    all_tasks = []
    for coeff in B_COEFFS:
        budget = int(math.floor(coeff * min_budget))
        for eta in ETA_LIST:
            key = _task_key(coeff, eta)
            if key not in completed:
                all_tasks.append((coeff, budget, eta, key))

    if not all_tasks:
        print('  all configs already completed; skipping')
        rows = []
        for coeff in B_COEFFS:
            for eta in ETA_LIST:
                key = _task_key(coeff, eta)
                row = {'B_label': str(coeff), 'eta': str(eta)}
                for m in METHODS:
                    row[m] = existing[key][m]
                rows.append(row)
        return rows

    print(f'  tasks to run: {len(all_tasks)}/{len(B_COEFFS) * len(ETA_LIST)}')

    rows = []
    static_plan_cache = {}

    for coeff, budget, eta, key in all_tasks:
        exec_prob = 1.0 - eta
        combo_seed = SEED + origin * 100 + dest + budget

        print(f'\n--- B={coeff} (budget={budget}), eta={eta} (exec_prob={exec_prob}) ---')

        uncertain_edges = _uncertainty_edges_for(origin, dest, eta)
        env = AnaheimEnv(
            origin=origin, dest=dest, budget=budget,
            exec_prob=exec_prob, deterministic=False, seed=combo_seed,
            uncertain_ratio=0.0 if uncertain_edges else (GLOBAL_UNCERTAIN_RATIO if eta > 0 else 0.0),
            uncertain_seed=GLOBAL_UNCERTAIN_SEED,
            uncertain_mode='random', top_k=6,
            time_step=TIME_STEP,
            uncertain_edges_explicit=uncertain_edges,
        )
        t0 = time.time()
        mc_vals = run_all_methods(env, origin, dest, budget, combo_seed, selected_methods=selected_methods, static_plan_cache=static_plan_cache, coeff=coeff, eta=eta)
        elapsed = time.time() - t0

        row = {'B_label': str(coeff), 'eta': str(eta)}
        old_vals = previous_values.get(key, {})
        for m in METHODS:
            row[m] = mc_vals.get(m) if m in mc_vals and mc_vals[m] is not None else old_vals.get(m, np.nan)
        for col in ILP_META_COLUMNS + SEGAC_META_COLUMNS:
            row[col] = mc_vals.get(col, '')
        rows.append(row)

        # Print progress
        parts = [f"  {m}={fmt3(mc_vals.get(m))}" for m in METHODS[:6]]
        print(' '.join(parts))
        parts2 = [f"  {m}={fmt3(mc_vals.get(m))}" for m in METHODS[6:]]
        print(' '.join(parts2))
        print(f'  [{elapsed:.0f}s]')

        checkpoint_rows = list(rows)
        if existing is not None:
            for c2 in B_COEFFS:
                for e2 in ETA_LIST:
                    existing_key = _task_key(c2, e2)
                    if existing_key in completed:
                        existing_row = {'B_label': str(c2), 'eta': str(e2)}
                        for m in METHODS:
                            existing_row[m] = existing[existing_key][m]
                        for col in ILP_META_COLUMNS + SEGAC_META_COLUMNS:
                            existing_row[col] = existing[existing_key].get(col, '')
                        checkpoint_rows.append(existing_row)
        _write_od_csv(csv_path, checkpoint_rows)
        print(f'  Checkpoint saved: {csv_path}')

    # Merge with existing rows if any
    if existing is not None:
        for coeff in B_COEFFS:
            for eta in ETA_LIST:
                key = _task_key(coeff, eta)
                if key in completed:
                    row = {'B_label': str(coeff), 'eta': str(eta)}
                    for m in METHODS:
                        row[m] = existing[key][m]
                    for col in ILP_META_COLUMNS + SEGAC_META_COLUMNS:
                        row[col] = existing[key].get(col, '')
                    rows.append(row)

    # ---- Save per-OD CSV ----
    _write_od_csv(csv_path, rows)
    print(f'\nSaved: {csv_path}')

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
    p = argparse.ArgumentParser(description='Anaheim Batch OD x eta experiments')
    p.add_argument('--run-all', action='store_true',
                   help='Run all OD pairs in OD_CONFIGS')
    p.add_argument('--od', type=int, nargs=2, metavar=('ORIGIN', 'DEST'),
                   help='Run a single OD pair')
    p.add_argument('--aggregate-only', action='store_true',
                   help='Only re-compute mean from existing od_*.csv files')
    p.add_argument('--force', action='store_true',
                   help='Force re-run all configs, ignore existing per-OD CSV')
    p.add_argument('--algorithms', nargs='+', default=['all'],
                   help='Algorithms to run, e.g. --algorithms eurac segac, or --algorithms all')
    return p.parse_args()


def main():
    args = parse_args()
    selected_methods = normalize_algorithms(args.algorithms)
    print('Selected: ' + ', '.join(selected_methods))

    if args.aggregate_only:
        aggregate_only()
        return

    if args.run_all:
        for (origin, dest) in OD_CONFIGS:
            run_one_od(origin, dest, force=args.force, selected_methods=selected_methods)
        aggregate_only()
        return

    if args.od:
        origin, dest = args.od
        run_one_od(origin, dest, force=args.force, selected_methods=selected_methods)
        return

    # Default: run all
    for (origin, dest) in OD_CONFIGS:
        run_one_od(origin, dest, force=args.force, selected_methods=selected_methods)
    aggregate_only()


if __name__ == '__main__':
    main()

