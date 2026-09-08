"""Run the Sioux Falls all-method comparison for a selected OD pair.

Only `ORIGIN`, `DEST`, and `UNCERTAIN_EDGES` need to be changed when
reusing this script for another OD pair. Budgets are derived from t_LET.
"""

# ============================================================
# OD and uncertainty settings
# ============================================================
ORIGIN = 11
DEST   = 20

# Explicit uncertain edges as {node: risky_next_node}.
# Set to None to use random uncertainty with uncertain_ratio=0.2.
UNCERTAIN_EDGES = {14: 15}
# ============================================================

import sys
import time
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util
_ROOT_DIR = Path(__file__).resolve().parent.parent
_EURAC_SPEC = importlib.util.spec_from_file_location("_public_eu_rac_tabular", _ROOT_DIR / "eu_rac_tabular.py")
_EURAC_MODULE = importlib.util.module_from_spec(_EURAC_SPEC)
_EURAC_SPEC.loader.exec_module(_EURAC_MODULE)
EURAC = _EURAC_MODULE.EURAC

sys.path.insert(0, str(Path(__file__).parent.parent / 'simple'))

from sioux_env import SiouxEnv
from robust_routing import run_robust_routing, extract_path as rr_extract_path
from dot_routing import run_dot_routing
from pulse_routing import run_pulse_routing
from otap_routing import run_otap_routing
from ilp_routing import run_ilp_routing
from gp3_routing import run_gp3_routing
from pql_routing import run_pql_routing
from ge_ddrl_routing import run_ge_ddrl_routing
from segac_routing import run_segac_routing
from dac_routing import run_dac_routing
from evaluator import evaluate_policy

# ------------------------------------------------------------------
#  t_LET Dijkstra
# ------------------------------------------------------------------
_NET_DIR = Path(__file__).parent.parent / 'Networks' / 'Networks' / 'SiouxFalls'
_net_df  = pd.read_csv(_NET_DIR / 'SiouxFalls_network.csv')

_mean_edges = {}
_successors = {}
for _, row in _net_df.iterrows():
    u, v = int(row['From']), int(row['To'])
    _mean_edges[(u, v)] = float(row['Cost'])
    _successors.setdefault(u, []).append(v)

def _dijkstra_mean(src, dst):
    dist = {src: 0.0}
    pq   = [(0.0, src)]
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

T_LET = _dijkstra_mean(ORIGIN, DEST)
BUDGET_FACTORS = [0.975, 1.000, 1.025]
BUDGETS        = [round(f * T_LET) for f in BUDGET_FACTORS]
BUDGET_LABELS  = [str(f) for f in BUDGET_FACTORS]

print(f't_LET({ORIGIN}{DEST}) = {T_LET:.4f} min')
print(f': {list(zip(BUDGET_LABELS, BUDGETS))}')

EVAL_EPISODES = 10000
SEED = 42

# EU-RAC hyperparameters
EURAC_LR_E = 0.2
EURAC_LR_D = 0.2
EURAC_LR_ACTOR = 0.1
EURAC_ENTROPY_COEF = 0.02
EURAC_WARM_LOGIT = 2.0

# RL training episodes
EURAC_EPISODES = 100000
PQL_EPISODES   = 50000
GE_EPISODES    = 50000
SEGAC_EPISODES = 50000
DAC_EPISODES   = 50000


def make_env(budget):
    if UNCERTAIN_EDGES is not None:
        return SiouxEnv(
            origin=ORIGIN, dest=DEST, budget=budget,
            exec_prob=0.8, deterministic=False, seed=SEED,
            uncertain_edges_explicit=UNCERTAIN_EDGES,
        )
    return SiouxEnv(
        origin=ORIGIN, dest=DEST, budget=budget,
        exec_prob=0.8, uncertain_ratio=0.2,
        deterministic=False, seed=SEED,
        uncertain_seed=42, uncertain_mode='random', top_k=6,
    )


def mc_eval(env, policy_func):
    return evaluate_policy(env, policy_func, episodes=EVAL_EPISODES)


def path_str(path):
    return ''.join(map(str, path))


def run_all(budget):
    print(f'\n{"="*60}')
    print(f'  Budget = {budget} min')
    print(f'{"="*60}')
    env = make_env(budget)
    results = {}

    # ------------------------------------------------------------------
    # DOT
    # ------------------------------------------------------------------
    print('\n[DOT]', end=' ', flush=True)
    t0 = time.time()
    r = run_dot_routing(env, budget=budget)
    dp_prob = r['f_origin']
    path = r['path']
    policy = r['policy']
    mc = mc_eval(env, lambda n, b: policy.get(n))
    results['DOT'] = dict(dp=dp_prob, mc=mc, path=path_str(path))
    print(f'DP={dp_prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # Robust =0.9
    # ------------------------------------------------------------------
    print('[Robust =0.9]', end=' ', flush=True)
    t0 = time.time()
    r = run_robust_routing(env, budget=budget, psi=0.9, m=2)
    dp_prob = r['u_origin']
    path = rr_extract_path(r['policy'], ORIGIN, DEST, env, budget, u_table=r['u_table'])
    policy = r['policy']
    mc = mc_eval(env, lambda n, b: policy.get(n))
    results['Robust'] = dict(dp=dp_prob, mc=mc, path=path_str(path))
    print(f'DP={dp_prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # Pulse
    # ------------------------------------------------------------------
    print('[Pulse]', end=' ', flush=True)
    t0 = time.time()
    r = run_pulse_routing(env, budget=budget)
    prob = r['prob']
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(env, lambda n, b: pm.get(n))
    results['Pulse'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # OTAP (K=100)
    # ------------------------------------------------------------------
    print('[OTAP K=100]', end=' ', flush=True)
    t0 = time.time()
    r = run_otap_routing(env, budget=budget, K=100)
    prob = r['prob']
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(env, lambda n, b: pm.get(n))
    results['OTAP'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # ILP (K=100)
    # ------------------------------------------------------------------
    print('[ILP K=100]', end=' ', flush=True)
    t0 = time.time()
    r = run_ilp_routing(env, budget=budget, K=100)
    prob = r['prob']
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(env, lambda n, b: pm.get(n))
    results['ILP'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # GP3
    # ------------------------------------------------------------------
    print('[GP3]', end=' ', flush=True)
    t0 = time.time()
    r = run_gp3_routing(env, budget=budget, zeta_min=0.0, zeta_max=50.0)
    prob = r['prob']
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(env, lambda n, b: pm.get(n))
    results['GP3'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # PQL (50k episodes)
    # ------------------------------------------------------------------
    print(f'[PQL {PQL_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_pql_routing(env, budget=budget, episodes=PQL_EPISODES,
                        alpha=0.1, epsilon=0.1, seed=SEED)
    prob = r['prob']
    path = r['path']
    q_table = r['q_table']
    successors = env.successors
    def pql_policy(node, bud):
        acts = successors.get(node, [])
        if not acts: return None
        q_node = q_table.get((node, bud), {})
        return max(acts, key=lambda a: q_node.get(a, 0.0))
    mc = mc_eval(env, pql_policy)
    results['PQL'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # GE-DDRL (50k episodes)
    # ------------------------------------------------------------------
    print(f'[GE-DDRL {GE_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_ge_ddrl_routing(env, budget=budget, N=200, delta_w=1.0,
                            alpha_t=0.05, epsilon=0.1,
                            episodes=GE_EPISODES, seed=SEED)
    prob = r['prob']
    path = r['path']
    Z = r['q_dist']
    sota_k = min(int(budget / 1.0), 200 - 1)
    def ge_policy(node, bud):
        acts = successors.get(node, [])
        if not acts: return None
        def obj(a):
            d = Z.get((node, a))
            return float(d[:sota_k+1].sum()) if d is not None else 0.0
        return max(acts, key=obj)
    mc = mc_eval(env, ge_policy)
    results['GE-DDRL'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # SEGAC (50k episodes)
    # ------------------------------------------------------------------
    print(f'[SEGAC {SEGAC_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_segac_routing(env, budget=budget, episodes=SEGAC_EPISODES,
                          lr_actor=0.01, lr_critic=0.1, seed=SEED)
    prob = r['prob']
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    def segac_policy(node, bud):
        if node in pm: return pm[node]
        acts = successors.get(node, [])
        return acts[0] if acts else None
    mc = mc_eval(env, segac_policy)
    results['SEGAC'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # DAC (50k episodes)
    # ------------------------------------------------------------------
    print(f'[DAC {DAC_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_dac_routing(env, budget=budget, episodes=DAC_EPISODES,
                        lr_critic=3e-4, lr_actor=0.1, lr_ent=3e-4, seed=SEED)
    prob = r['prob']
    path = r['path']
    q_table = r['q_table']
    successors = env.successors
    def dac_policy(node, bud):
        acts = successors.get(node, [])
        if not acts: return None
        return max(acts, key=lambda a: q_table.get((node, bud, a), 0.0))
    mc = mc_eval(env, dac_policy)
    results['DAC'] = dict(dp=prob, mc=mc, path=path_str(path))
    print(f'prob={prob:.4f}  MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # EU-RAC (100k episodes)
    # ------------------------------------------------------------------
    print(f'[EU-RAC {EURAC_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    agent = EURAC(
        env=env,
        lr_e=EURAC_LR_E,
        lr_d=EURAC_LR_D,
        lr_actor=EURAC_LR_ACTOR,
        entropy_coef=EURAC_ENTROPY_COEF,
    )
    agent.warm_start(logit=EURAC_WARM_LOGIT)
    np.random.seed(SEED)
    for ep in range(1, EURAC_EPISODES + 1):
        agent.run_episode(train=True)
        if ep % 20000 == 0:
            print(f'.', end='', flush=True)
    sota = agent.evaluate(n_episodes=5000)
    # greedy path
    state = (ORIGIN, budget)
    gpath = [ORIGIN]
    visited = set()
    for _ in range(len(env.nodes) * 2):
        node, bud = state
        if node == DEST or bud < 0: break
        if node in visited: break
        visited.add(node)
        policy_d, actions = agent.get_policy(state)
        if not actions: break
        unvisited = [a for a in actions if a not in visited]
        if not unvisited: break
        best = max(unvisited, key=lambda a: policy_d.get(a, 0))
        mean_t, _ = env.edges[(node, best)]
        state = (best, bud - round(mean_t))
        gpath.append(best)
    # MC eval with execution uncertainty
    def eurac_policy(node, bud):
        s = (node, bud)
        pol, acts = agent.get_policy(s)
        if not acts: return None
        return max(pol, key=pol.get)
    mc = mc_eval(env, eurac_policy)
    results['EU-RAC'] = dict(dp=sota, mc=mc, path=path_str(gpath))
    print(f'\n  EU-RAC: eval={sota:.4f}  MC={mc:.4f}  path={path_str(gpath)}  ({time.time()-t0:.1f}s)')

    return results


# ============================================================
# Main
# ============================================================
all_results = {}
for budget in BUDGETS:
    all_results[budget] = run_all(budget)

# ============================================================
# Print Markdown table
# ============================================================
METHODS = ['EU-RAC', 'DOT', 'Robust', 'Pulse', 'OTAP', 'ILP', 'GP3', 'PQL', 'GE-DDRL', 'SEGAC', 'DAC']
METHOD_LABELS = {
    'EU-RAC':  'EU-RAC (ours)',
    'DOT':     'DOT (Prakash 2020)',
    'Robust':  'Robust =0.9 (Manseur 2020)',
    'Pulse':   'Pulse (Leiva 2026)',
    'OTAP':    'OTAP K=100 (Yang 2017)',
    'ILP':     'ILP K=100 (Cao 2020)',
    'GP3':     'GP3 (Guo 2022)',
    'PQL':     'PQL (Cao 2020)',
    'GE-DDRL': 'GE-DDRL (Guo 2023)',
    'SEGAC':   'SEGAC (Guo 2024)',
    'DAC':     'DAC (Chen 2024)',
}
METHOD_TYPE = {
    'DOT': 'DP', 'Robust': 'DP', 'Pulse': 'DP',
    'OTAP': 'MP', 'ILP': 'MP', 'GP3': 'MP',
    'PQL': 'RL', 'GE-DDRL': 'RL', 'SEGAC': 'RL', 'DAC': 'RL', 'EU-RAC': 'RL',
}

print('\n\n' + '='*80)
print(f'12   OD={ORIGIN}{DEST}, t_LET={T_LET:.2f} min, uncertain-ratio=0.2, uncertain-seed=42')
print('='*80)

B0, B1, B2 = BUDGETS[0], BUDGETS[1], BUDGETS[2]
L0, L1, L2 = BUDGET_LABELS[0], BUDGET_LABELS[1], BUDGET_LABELS[2]

# Header
header = f'|  |  | T={B0} ({L0}) MC | T={B1} ({L1}) MC | T={B2} ({L2}) MC |  (T={B1}) |'
sep    =  '|------|:----:|:---:|:---:|:---:|:---|'
print(header)
print(sep)

for m in METHODS:
    mc0   = all_results[B0][m]['mc']
    mc1   = all_results[B1][m]['mc']
    mc2   = all_results[B2][m]['mc']
    path1 = all_results[B1][m]['path']
    mtype = METHOD_TYPE[m]
    label = METHOD_LABELS[m]
    print(f'| {label} | {mtype} | {mc0:.4f} | {mc1:.4f} | {mc2:.4f} | {path1} |')

print()
print('MC  10000  Monte Carlo ')
print('    DP=MP=/RL=')
print('    EU-RAC  100000 episodesPQL/GE-DDRL/SEGAC/DAC  50000 episodes')

# Also save to CSV
import csv
_csv_name = f'results_OD{ORIGIN}_{DEST}.csv'
with open(Path(__file__).parent / _csv_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'type', 'budget', 'dp_prob', 'mc_prob', 'path'])
    for budget in BUDGETS:
        for m in METHODS:
            r = all_results[budget][m]
            writer.writerow([m, METHOD_TYPE[m], budget, f"{r['dp']:.4f}", f"{r['mc']:.4f}", r['path']])
print(f'\n sioux/{_csv_name}')
