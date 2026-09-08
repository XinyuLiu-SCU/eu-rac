"""Run the Sioux Falls generalization experiment.

EU-RAC and the baselines are solved under random execution uncertainty and
evaluated under an explicit or fixed-node uncertainty setting.

Usage:
    python run_generalization.py
    python run_generalization.py --uncertain-nodes 2,11,16,18
    python run_generalization.py --origin 3 --dest 19 --uncertain-nodes 2,11,16,18
"""

import sys
import argparse
import time
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

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
from evaluator import evaluate_policy
import importlib.util
_ROOT_DIR = Path(__file__).resolve().parent.parent
_EURAC_SPEC = importlib.util.spec_from_file_location("_public_eu_rac_tabular", _ROOT_DIR / "eu_rac_tabular.py")
_EURAC_MODULE = importlib.util.module_from_spec(_EURAC_SPEC)
_EURAC_SPEC.loader.exec_module(_EURAC_MODULE)
EURAC = _EURAC_MODULE.EURAC

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--origin', type=int, default=2)
parser.add_argument('--dest',   type=int, default=15)
parser.add_argument('--uncertain-nodes', type=str, default=None,
                    help='Comma-separated fixed uncertain node IDs, e.g. 2,11,16,18')
parser.add_argument('--uncertain-edges', type=str, default='3:4',
                    help='Explicit edge pairs node:next,node:next, e.g. 3:4 or 3:4,9:10')
args = parser.parse_args()

ORIGIN = args.origin
DEST   = args.dest

# uncertain_edges_explicit  --uncertain-nodes 
if args.uncertain_nodes is not None:
    UNCERTAIN_NODES = [int(x) for x in args.uncertain_nodes.split(',')]
    UNCERTAIN_EDGES_EXPLICIT = None
else:
    UNCERTAIN_NODES = None
    UNCERTAIN_EDGES_EXPLICIT = {
        int(pair.split(':')[0]): int(pair.split(':')[1])
        for pair in args.uncertain_edges.split(',')
    }

SEED           = 42
EVAL_EPISODES  = 10000
EURAC_EPISODES = 100000
PQL_EPISODES   = 50000
GE_EPISODES    = 50000
SEGAC_EPISODES = 50000

# Best EU-RAC hyperparameters from tuning
BEST_CFG = dict(lr_e=0.2, lr_d=0.2, lr_actor=0.1, entropy_coef=0.02, warm_logit=2.0)

# ============================================================
# t_LET and budgets
# ============================================================
_NET_DIR = Path(__file__).parent.parent / 'network' / 'SiouxFalls'
_net_df  = pd.read_csv(_NET_DIR / 'SiouxFalls_network.csv')

_mean_edges = {}
_successors_d = {}
for _, row in _net_df.iterrows():
    u, v = int(row['From']), int(row['To'])
    _mean_edges[(u, v)] = float(row['Cost'])
    _successors_d.setdefault(u, []).append(v)

def _dijkstra_mean(src, dst):
    dist = {src: 0.0}
    pq = [(0.0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == dst: break
        for v in _successors_d.get(u, []):
            nd = d + _mean_edges[(u, v)]
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist[dst]

T_LET  = _dijkstra_mean(ORIGIN, DEST)
BUDGETS       = [round(f * T_LET) for f in [0.975, 1.000, 1.025]]
BUDGET_LABELS = ['0.975', '1.000', '1.025']

print(f't_LET({ORIGIN}{DEST}) = {T_LET:.4f}')
print(f': {list(zip(BUDGET_LABELS, BUDGETS))}')
print(f': random uncertain_ratio=0.2')
if UNCERTAIN_EDGES_EXPLICIT is not None:
    print(f': explicit uncertain_edges={UNCERTAIN_EDGES_EXPLICIT}')
else:
    print(f': fixed uncertain_nodes={UNCERTAIN_NODES}')

# ============================================================
# Env factories
# ============================================================
def make_train_env(budget):
    """Random-mode env for training/solving."""
    return SiouxEnv(
        origin=ORIGIN, dest=DEST, budget=budget,
        exec_prob=0.8, uncertain_ratio=0.2,
        deterministic=False, seed=SEED,
        uncertain_seed=42, uncertain_mode='random', top_k=6,
    )

def make_test_env(budget):
    """Test env: explicit edges take priority over node list."""
    if UNCERTAIN_EDGES_EXPLICIT is not None:
        return SiouxEnv(
            origin=ORIGIN, dest=DEST, budget=budget,
            exec_prob=0.8, deterministic=False, seed=SEED,
            uncertain_edges_explicit=UNCERTAIN_EDGES_EXPLICIT,
        )
    return SiouxEnv(
        origin=ORIGIN, dest=DEST, budget=budget,
        exec_prob=0.8, deterministic=False, seed=SEED,
        uncertain_nodes=UNCERTAIN_NODES,
    )

def mc_eval(test_env, policy_func):
    return evaluate_policy(test_env, policy_func, episodes=EVAL_EPISODES)

def path_str(path):
    return ''.join(map(str, path))


# ============================================================
# Per-budget runner
# ============================================================
def run_all(budget):
    print(f'\n{"="*60}')
    print(f'  Budget = {budget} min')
    print(f'{"="*60}')
    train_env = make_train_env(budget)
    test_env  = make_test_env(budget)
    results = {}
    successors = train_env.successors

    # ------------------------------------------------------------------
    # DOT  solve on train_env, eval on test_env
    # ------------------------------------------------------------------
    print('[DOT]', end=' ', flush=True)
    t0 = time.time()
    r = run_dot_routing(train_env, budget=budget)
    policy = r['policy']
    path   = r['path']
    mc = mc_eval(test_env, lambda n, b: policy.get(n))
    results['DOT'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # Robust =0.9
    # ------------------------------------------------------------------
    print('[Robust =0.9]', end=' ', flush=True)
    t0 = time.time()
    r = run_robust_routing(train_env, budget=budget, psi=0.9, m=2)
    policy = r['policy']
    path   = rr_extract_path(policy, ORIGIN, DEST, train_env, budget, u_table=r['u_table'])
    mc = mc_eval(test_env, lambda n, b: policy.get(n))
    results['Robust'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # Pulse
    # ------------------------------------------------------------------
    print('[Pulse]', end=' ', flush=True)
    t0 = time.time()
    r = run_pulse_routing(train_env, budget=budget)
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(test_env, lambda n, b: pm.get(n))
    results['Pulse'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # OTAP
    # ------------------------------------------------------------------
    print('[OTAP K=200]', end=' ', flush=True)
    t0 = time.time()
    r = run_otap_routing(train_env, budget=budget, K=200)
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(test_env, lambda n, b: pm.get(n))
    results['OTAP'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # ILP
    # ------------------------------------------------------------------
    print('[ILP K=200]', end=' ', flush=True)
    t0 = time.time()
    r = run_ilp_routing(train_env, budget=budget, K=200)
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(test_env, lambda n, b: pm.get(n))
    results['ILP'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # GP3
    # ------------------------------------------------------------------
    print('[GP3]', end=' ', flush=True)
    t0 = time.time()
    r = run_gp3_routing(train_env, budget=budget, zeta_min=0.0, zeta_max=50.0)
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    mc = mc_eval(test_env, lambda n, b: pm.get(n))
    results['GP3'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # PQL
    # ------------------------------------------------------------------
    print(f'[PQL {PQL_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_pql_routing(train_env, budget=budget, episodes=PQL_EPISODES,
                        alpha=0.1, epsilon=0.1, seed=SEED)
    path    = r['path']
    q_table = r['q_table']
    def pql_policy(node, bud):
        acts = successors.get(node, [])
        if not acts: return None
        q_node = q_table.get((node, bud), {})
        return max(acts, key=lambda a: q_node.get(a, 0.0))
    mc = mc_eval(test_env, pql_policy)
    results['PQL'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # GE-DDRL
    # ------------------------------------------------------------------
    print(f'[GE-DDRL {GE_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_ge_ddrl_routing(train_env, budget=budget, N=200, delta_w=1.0,
                            alpha_t=0.05, epsilon=0.1,
                            episodes=GE_EPISODES, seed=SEED)
    path = r['path']
    Z    = r['q_dist']
    sota_k = min(int(budget / 1.0), 200 - 1)
    def ge_policy(node, bud):
        acts = successors.get(node, [])
        if not acts: return None
        def obj(a):
            d = Z.get((node, a))
            return float(d[:sota_k+1].sum()) if d is not None else 0.0
        return max(acts, key=obj)
    mc = mc_eval(test_env, ge_policy)
    results['GE-DDRL'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # SEGAC
    # ------------------------------------------------------------------
    print(f'[SEGAC {SEGAC_EPISODES}ep]', end=' ', flush=True)
    t0 = time.time()
    r = run_segac_routing(train_env, budget=budget, episodes=SEGAC_EPISODES,
                          lr_actor=0.01, lr_critic=0.1, seed=SEED)
    path = r['path']
    pm = {path[i]: path[i+1] for i in range(len(path)-1)}
    def segac_policy(node, bud):
        if node in pm: return pm[node]
        acts = successors.get(node, [])
        return acts[0] if acts else None
    mc = mc_eval(test_env, segac_policy)
    results['SEGAC'] = dict(mc=mc, path=path_str(path))
    print(f'MC={mc:.4f}  path={path_str(path)}  ({time.time()-t0:.1f}s)')

    # ------------------------------------------------------------------
    # EU-RAC  train on random env, eval on fixed-node env
    # ------------------------------------------------------------------
    print(f'[EU-RAC {EURAC_EPISODES}ep (best cfg)]', end=' ', flush=True)
    t0 = time.time()
    agent = EURAC(
        env=train_env,
        lr_e=BEST_CFG['lr_e'],
        lr_d=BEST_CFG['lr_d'],
        lr_actor=BEST_CFG['lr_actor'],
        entropy_coef=BEST_CFG['entropy_coef'],
    )
    agent.warm_start(logit=BEST_CFG['warm_logit'])
    np.random.seed(SEED)
    for ep in range(1, EURAC_EPISODES + 1):
        agent.run_episode(train=True)
        if ep % 20000 == 0:
            print('.', end='', flush=True)

    # greedy path (on train env for path display)
    state = (ORIGIN, budget)
    gpath = [ORIGIN]
    visited = set()
    for _ in range(len(train_env.nodes) * 2):
        node, bud = state
        if node == DEST or bud < 0: break
        if node in visited: break
        visited.add(node)
        policy_d, actions = agent.get_policy(state)
        if not actions: break
        unvisited = [a for a in actions if a not in visited]
        if not unvisited: break
        best = max(unvisited, key=lambda a: policy_d.get(a, 0))
        mean_t, _ = train_env.edges[(node, best)]
        state = (best, bud - round(mean_t))
        gpath.append(best)

    # MC eval on test_env (fixed uncertain nodes)
    def eurac_policy(node, bud):
        pol, acts = agent.get_policy((node, bud))
        if not acts: return None
        return max(pol, key=pol.get)
    mc = mc_eval(test_env, eurac_policy)
    results['EU-RAC'] = dict(mc=mc, path=path_str(gpath))
    print(f'\n  EU-RAC: MC={mc:.4f}  path={path_str(gpath)}  ({time.time()-t0:.1f}s)')

    return results


# ============================================================
# Main
# ============================================================
all_results = {}
for budget in BUDGETS:
    all_results[budget] = run_all(budget)

# ============================================================
# Markdown table
# ============================================================
METHODS = ['EU-RAC', 'DOT', 'Robust', 'Pulse', 'OTAP', 'ILP', 'GP3', 'PQL', 'GE-DDRL', 'SEGAC']
METHOD_LABELS = {
    'EU-RAC':  'EU-RAC (ours)',
    'DOT':     'DOT (Prakash 2020)',
    'Robust':  'Robust =0.9 (Manseur 2020)',
    'Pulse':   'Pulse (Leiva 2026)',
    'OTAP':    'OTAP K=200 (Yang 2017)',
    'ILP':     'ILP K=200 (Cao 2020)',
    'GP3':     'GP3 (Guo 2022)',
    'PQL':     'PQL (Cao 2020)',
    'GE-DDRL': 'GE-DDRL (Guo 2023)',
    'SEGAC':   'SEGAC (Guo 2024)',
}
METHOD_TYPE = {
    'DOT': 'DP', 'Robust': 'DP', 'Pulse': 'DP',
    'OTAP': 'MP', 'ILP': 'MP', 'GP3': 'MP',
    'PQL': 'RL', 'GE-DDRL': 'RL', 'SEGAC': 'RL', 'EU-RAC': 'RL',
}

B0, B1, B2 = BUDGETS
L0, L1, L2 = BUDGET_LABELS

print('\n\n' + '='*80)
print(f'  OD={ORIGIN}{DEST}, t_LET={T_LET:.2f}')
print(f': random uncertain_ratio=0.2  |  : fixed nodes={UNCERTAIN_NODES}')
print('='*80)

header = f'|  |  | T={B0} ({L0}) MC | T={B1} ({L1}) MC | T={B2} ({L2}) MC |  (T={B1}) |'
sep    =  '|------|:----:|:---:|:---:|:---:|:---|'
print(header)
print(sep)

for m in METHODS:
    mc0   = all_results[B0][m]['mc']
    mc1   = all_results[B1][m]['mc']
    mc2   = all_results[B2][m]['mc']
    path1 = all_results[B1][m]['path']
    print(f'| {METHOD_LABELS[m]} | {METHOD_TYPE[m]} | {mc0:.4f} | {mc1:.4f} | {mc2:.4f} | {path1} |')

print()
print(f'EU-RAC  random  fixed nodes={UNCERTAIN_NODES} ')
print('     baseline  random  fixed ')

# Save CSV
import csv
_csv = Path(__file__).parent / f'results_generalization_OD{ORIGIN}_{DEST}.csv'
with open(_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'type', 'budget', 'mc_prob', 'path',
                     'train_mode', 'test_nodes'])
    for budget in BUDGETS:
        for m in METHODS:
            r = all_results[budget][m]
            writer.writerow([m, METHOD_TYPE[m], budget,
                             f"{r['mc']:.4f}", r['path'],
                             'random', str(UNCERTAIN_NODES)])
print(f'\n {_csv.name}')
