"""
run_sioux.py  Train EU-RAC on the Sioux Falls network.

Usage:
    python run_sioux.py [options]
    python run_sioux.py --help
"""
import sys
import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

import importlib.util
_ROOT_DIR = Path(__file__).resolve().parent.parent
_EURAC_SPEC = importlib.util.spec_from_file_location("_public_eu_rac_tabular", _ROOT_DIR / "eu_rac_tabular.py")
_EURAC_MODULE = importlib.util.module_from_spec(_EURAC_SPEC)
_EURAC_SPEC.loader.exec_module(_EURAC_MODULE)
EURAC = _EURAC_MODULE.EURAC

# Add simple/ to path for other imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'simple'))

from sioux_env import SiouxEnv  # noqa: E402
from robust_routing import run_robust_routing, extract_path as rr_extract_path  # noqa: E402
from dot_routing import run_dot_routing  # noqa: E402
from pulse_routing import run_pulse_routing  # noqa: E402
from otap_routing import run_otap_routing  # noqa: E402
from ilp_routing import run_ilp_routing  # noqa: E402
from gp3_routing import run_gp3_routing  # noqa: E402
from pql_routing import run_pql_routing  # noqa: E402
from ge_ddrl_routing import run_ge_ddrl_routing  # noqa: E402
from segac_routing import run_segac_routing  # noqa: E402
from dac_routing import run_dac_routing  # noqa: E402
from evaluator import evaluate_policy  # noqa: E402


def plot_training_curve(episodes, avg_rewards, overall_avgs, avg_q_e, avg_q_d):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(episodes, avg_rewards, marker='o', label='avg reward (window)')
    ax[0].plot(episodes, overall_avgs, marker='x', label='overall avg')
    ax[0].set_ylabel('Reward')
    ax[0].set_title('EU-RAC training curve')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(episodes, avg_q_e, marker='o', label='avg Q_e')
    ax[1].plot(episodes, avg_q_d, marker='x', label='avg Q_d')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Average Q value')
    ax[1].set_title('Q_e / Q_d learning curves')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    default_origin, default_dest = SiouxEnv.default_od()
    p = argparse.ArgumentParser(
        description='Train EU-RAC on the Sioux Falls network.')
    p.add_argument('--origin', type=int, default=default_origin,
                   help=f'Origin node (default: {default_origin})')
    p.add_argument('--dest', type=int, default=default_dest,
                   help=f'Destination node (default: {default_dest})')
    p.add_argument('--budget', type=int, default=60,
                   help='Time budget in minutes (default: 60)')
    p.add_argument('--exec-prob', type=float, default=0.8,
                   help='Execution probability at uncertain nodes (default: 0.8)')
    p.add_argument('--uncertain-ratio', type=float, default=0.2,
                   help='Fraction of intersections with execution uncertainty (default: 0.2)')
    p.add_argument('--uncertain-seed', type=int, default=42,
                   help='Random seed for selecting uncertain nodes (default: 42)')
    p.add_argument('--uncertain-mode', type=str, default='random',
                   choices=['random', 'important'],
                   help='How to select uncertain nodes: random (default) or important (betweenness)')
    p.add_argument('--top-k', type=int, default=6,
                   help='Number of high-centrality nodes to mark uncertain in important mode (default: 6)')
    p.add_argument('--deterministic', action='store_true',
                   help='Use mean travel times only (no stochasticity)')
    p.add_argument('--n-train', type=int, default=10000,
                   help='Training episodes (default: 10000)')
    p.add_argument('--n-eval', type=int, default=5000,
                   help='Evaluation episodes (default: 5000)')
    p.add_argument('--lr-e', type=float, default=0.2,
                   help='Execution critic learning rate (default: 0.2)')
    p.add_argument('--lr-d', type=float, default=0.2,
                   help='Decision critic learning rate (default: 0.2)')
    p.add_argument('--lr-actor', type=float, default=0.1,
                   help='Actor learning rate (default: 0.1)')
    p.add_argument('--entropy-coef', type=float, default=0.02,
                   help='Entropy regularization coefficient (default: 0.02)')
    p.add_argument('--warm-logit', type=float, default=2.0,
                   help='Warm-start logit for LET path initialization (default: 2.0)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--report-every', type=int, default=2000,
                   help='Print progress every N episodes (default: 2000)')
    p.add_argument('--save-csv', type=str, default=None,
                   help='Save training curve to this CSV file')
    p.add_argument('--plot', action='store_true',
                   help='Show training curve plot when finished')
    p.add_argument('--baseline', type=str, default=None,
                   choices=['robust_routing', 'dot', 'pulse', 'otap', 'ilp', 'gp3', 'pql', 'ge_ddrl', 'segac', 'dac'],
                   help='Baseline to run alongside (or instead of) EU-RAC')
    p.add_argument('--psi', type=float, default=0.9,
                   help='Robustness weight  for robust_routing baseline (default: 0.9)')
    p.add_argument('--m', type=int, default=2,
                   help='Number of successors in robust criterion (default: 2)')
    p.add_argument('--baseline-only', action='store_true',
                   help='Run only the baseline, skip EU-RAC training')
    p.add_argument('--otap-k', type=int, default=30,
                   help='Number of samples for OTAP baseline (default: 30)')
    p.add_argument('--ilp-k', type=int, default=200,
                   help='Number of samples for ILP baseline (default: 200)')
    p.add_argument('--gp3-zeta-min', type=float, default=0.0,
                   help='Lower bound of zeta search for GP3 (default: 0.0)')
    p.add_argument('--gp3-zeta-max', type=float, default=50.0,
                   help='Upper bound of zeta search for GP3 (default: 50.0)')
    p.add_argument('--pql-episodes', type=int, default=50000,
                   help='Training episodes for PQL baseline (default: 50000)')
    p.add_argument('--pql-alpha', type=float, default=0.1,
                   help='Learning rate for PQL baseline (default: 0.1)')
    p.add_argument('--pql-epsilon', type=float, default=0.1,
                   help='Epsilon-greedy rate for PQL baseline (default: 0.1)')
    p.add_argument('--ge-n', type=int, default=200,
                   help='Number of distribution atoms for GE-DDRL (default: 200)')
    p.add_argument('--ge-delta-w', type=float, default=1.0,
                   help='Atom width (minutes) for GE-DDRL (default: 1.0)')
    p.add_argument('--ge-alpha', type=float, default=0.05,
                   help='Learning rate for GE-DDRL (default: 0.05)')
    p.add_argument('--ge-epsilon', type=float, default=0.1,
                   help='Epsilon-greedy rate for GE-DDRL (default: 0.1)')
    p.add_argument('--ge-episodes', type=int, default=50000,
                   help='Training episodes for GE-DDRL (default: 50000)')
    p.add_argument('--segac-episodes', type=int, default=50000,
                   help='Training episodes for SEGAC (default: 50000)')
    p.add_argument('--segac-lr-actor', type=float, default=0.01,
                   help='Actor learning rate for SEGAC (default: 0.01)')
    p.add_argument('--segac-lr-critic', type=float, default=0.1,
                   help='Critic learning rate for SEGAC (default: 0.1)')
    p.add_argument('--dac-episodes', type=int, default=50000,
                   help='Training episodes for DAC baseline (default: 50000)')
    p.add_argument('--dac-lr-critic', type=float, default=3e-4,
                   help='Critic learning rate for DAC baseline (default: 3e-4)')
    p.add_argument('--dac-lr-actor', type=float, default=0.1,
                   help='Actor learning rate (eta) for DAC baseline (default: 0.1)')
    p.add_argument('--dac-lr-ent', type=float, default=3e-4,
                   help='Entropy coefficient learning rate for DAC baseline (default: 3e-4)')
    p.add_argument('--eval-robust', action='store_true',
                   help='Evaluate Robust Routing policy via Monte Carlo under execution uncertainty')
    p.add_argument('--eval-episodes', type=int, default=10000,
                   help='Number of Monte Carlo rollouts for policy evaluation (default: 10000)')
    p.add_argument('--uncertain-edges-explicit', type=str, default=None,
                   help='Explicit uncertain edges as "u:v u:v ..." (e.g. "3:4 9:10"). '
                        'Overrides --uncertain-mode/--uncertain-ratio.')
    return p.parse_args()


def greedy_path(agent, env):
    """Compute greedy path using mean travel times."""
    state = (env.origin, env.budget)
    path = [env.origin]
    visited_nodes = set()
    max_steps = len(env.nodes) * 2

    for _ in range(max_steps):
        node, budget = state
        if node == env.dest or budget < 0:
            break
        if node in visited_nodes:
            break
        visited_nodes.add(node)

        policy, actions = agent.get_policy(state)
        if not actions:
            break
        unvisited = [a for a in actions if a not in visited_nodes]
        if not unvisited:
            break
        best = max(unvisited, key=lambda a: policy.get(a, 0))
        mean_t, _ = env.edges[(node, best)]
        state = (best, budget - round(mean_t))
        path.append(best)

    return path


def run_baseline_robust(env, args):
    """Run Nadir Robust Routing DP and print results."""
    print(f'\n--- Baseline: Robust Routing  (={args.psi}, m={args.m}) ---')
    result = run_robust_routing(env, budget=args.budget, psi=args.psi, m=args.m)
    u_origin = result['u_origin']
    path = rr_extract_path(result['policy'], env.origin, env.dest, env, args.budget,
                           u_table=result['u_table'])
    print(f'Robust SOTA probability (DP)    : {u_origin:.4f}')
    print(f'Robust policy path              : {"  ".join(map(str, path))}')

    if args.eval_robust:
        policy = result['policy']

        def policy_func(node, budget):
            return policy.get(node)

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'Robust SOTA probability (MC)    : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return u_origin, mc_sota

    return u_origin, None


def run_baseline_dot(env, args):
    """Run DOT routing DP and print results."""
    print('\n--- Baseline: DOT Routing ---')
    result = run_dot_routing(env, budget=args.budget)
    f_origin = result['f_origin']
    path = result['path']
    print(f'DOT SOTA probability (DP)       : {f_origin:.4f}')
    print(f'DOT policy path                 : {"  ".join(map(str, path))}')

    if args.eval_robust:
        policy = result['policy']

        def policy_func(node, budget):
            return policy.get(node)

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'DOT SOTA probability (MC)       : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return f_origin, mc_sota

    return f_origin, None


def run_baseline_pulse(env, args):
    """Run Pulse routing and print results."""
    print('\n--- Baseline: Pulse Routing (MPOAP) ---')
    result = run_pulse_routing(env, budget=args.budget)
    prob = result['prob']
    path = result['path']
    print(f'Pulse SOTA probability (exact)  : {prob:.4f}')
    print(f'Pulse optimal path              : {"  ".join(map(str, path))}')
    print(f'Path mean / std                 : {result["path_mean"]:.2f} min / {result["path_std"]:.2f} min')

    if args.eval_robust:
        policy_map = {}
        for i in range(len(path) - 1):
            policy_map[path[i]] = path[i + 1]

        def policy_func(node, budget):
            return policy_map.get(node)

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'Pulse SOTA probability (MC)     : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_otap(env, args):
    """Run sample-based OTAP routing and print results."""
    print(f'\n--- Baseline: OTAP Routing (K={args.otap_k}) ---')
    result = run_otap_routing(env, budget=args.budget, K=args.otap_k)
    prob = result['prob']
    path = result['path']
    print(f'OTAP SOTA probability (sample)  : {prob:.4f}')
    print(f'OTAP optimal path               : {"  ".join(map(str, path))}')

    if args.eval_robust:
        policy_map = {}
        for i in range(len(path) - 1):
            policy_map[path[i]] = path[i + 1]

        def policy_func(node, budget):
            return policy_map.get(node)

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'OTAP SOTA probability (MC)      : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_ilp(env, args):
    """Run ILP-based OTAP routing and print results."""
    print(f'\n--- Baseline: ILP Routing (Cao et al. 2020, K={args.ilp_k}) ---')
    result = run_ilp_routing(env, budget=args.budget, K=args.ilp_k)
    prob = result['prob']
    path = result['path']
    print(f'ILP SOTA probability            : {prob:.4f}')
    print(f'ILP optimal path                : {"  ".join(map(str, path))}')

    if args.eval_robust:
        policy_map = {path[i]: path[i + 1] for i in range(len(path) - 1)}

        def policy_func(node, budget):
            return policy_map.get(node)

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'ILP SOTA probability (MC)       : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_gp3(env, args):
    """Run GP3 routing and print results."""
    print(f'\n--- Baseline: GP3 (Guo et al. 2022, [{args.gp3_zeta_min},{args.gp3_zeta_max}]) ---')
    result = run_gp3_routing(env, budget=args.budget,
                             zeta_min=args.gp3_zeta_min,
                             zeta_max=args.gp3_zeta_max)
    prob = result['prob']
    path = result['path']
    print(f'GP3 SOTA probability (normal)   : {prob:.4f}')
    print(f'GP3 optimal path                : {"  ".join(map(str, path))}')
    print(f'Best zeta                       : {result["best_zeta"]:.4f}')

    if args.eval_robust:
        policy_map = {path[i]: path[i + 1] for i in range(len(path) - 1)}

        def policy_func(node, budget):
            return policy_map.get(node)

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'GP3 SOTA probability (MC)       : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_pql(env, args):
    """Run PQL (Practical Q-Learning) and print results."""
    print(f'\n--- Baseline: PQL (Cao et al. 2020, episodes={args.pql_episodes}) ---')
    result = run_pql_routing(env, budget=args.budget,
                             episodes=args.pql_episodes,
                             alpha=args.pql_alpha,
                             epsilon=args.pql_epsilon,
                             seed=args.seed)
    prob = result['prob']
    path = result['path']
    print(f'PQL SOTA probability (MC)       : {prob:.4f}')
    print(f'PQL greedy path                 : {"  ".join(map(str, path))}')

    if args.eval_robust:
        q_table = result['q_table']
        successors = env.successors

        def policy_func(node, budget):
            acts = successors.get(node, [])
            if not acts:
                return None
            q_node = q_table.get((node, budget), {})
            return max(acts, key=lambda a: q_node.get(a, 0.0))

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'PQL SOTA probability (MC+exec)  : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_ge_ddrl(env, args):
    """Run GE-DDRL (Tabular DRL, SOTA mode) and print results."""
    print(f'\n--- Baseline: GE-DDRL (Guo et al. 2023, episodes={args.ge_episodes}, N={args.ge_n}) ---')
    result = run_ge_ddrl_routing(
        env, budget=args.budget,
        N=args.ge_n,
        delta_w=args.ge_delta_w,
        alpha_t=args.ge_alpha,
        epsilon=args.ge_epsilon,
        episodes=args.ge_episodes,
        seed=args.seed,
    )
    prob = result['prob']
    path = result['path']
    print(f'GE-DDRL SOTA probability        : {prob:.4f}')
    print(f'GE-DDRL greedy path             : {"  ".join(map(str, path))}')

    if args.eval_robust:
        Z = result['q_dist']
        successors = env.successors
        sota_k = min(int(args.budget / args.ge_delta_w), args.ge_n - 1)

        def _sota_obj(u, v):
            dist = Z.get((u, v))
            if dist is None:
                return 0.0
            return float(dist[:sota_k + 1].sum())

        def policy_func(node, budget):
            acts = successors.get(node, [])
            if not acts:
                return None
            return max(acts, key=lambda a: _sota_obj(node, a))

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'GE-DDRL SOTA probability (MC)   : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_segac(env, args):
    """Run SEGAC (On-Policy GAC) and print results."""
    print(f'\n--- Baseline: SEGAC (Guo et al. 2024, episodes={args.segac_episodes}) ---')
    result = run_segac_routing(
        env, budget=args.budget,
        episodes=args.segac_episodes,
        lr_actor=args.segac_lr_actor,
        lr_critic=args.segac_lr_critic,
        seed=args.seed,
    )
    prob = result['prob']
    path = result['path']
    print(f'SEGAC SOTA probability          : {prob:.4f}')
    print(f'SEGAC greedy path               : {"  ".join(map(str, path))}')

    if args.eval_robust:
        from collections import defaultdict
        successors = env.successors

        # Rebuild softmax policy from the returned path for MC eval.
        # Since run_segac_routing doesn't expose theta, we use a fixed-path
        # policy derived from the greedy path, then fall back to any action.
        path_map = {path[i]: path[i + 1] for i in range(len(path) - 1)}

        def policy_func(node, budget):
            if node in path_map:
                return path_map[node]
            acts = successors.get(node, [])
            return acts[0] if acts else None

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'SEGAC SOTA probability (MC)     : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def run_baseline_dac(env, args):
    """Run DAC (Discrete Actor-Critic) and print results."""
    print(f'\n--- Baseline: DAC (Chen et al. 2024, episodes={args.dac_episodes}) ---')
    result = run_dac_routing(
        env, budget=args.budget,
        episodes=args.dac_episodes,
        lr_critic=args.dac_lr_critic,
        lr_actor=args.dac_lr_actor,
        lr_ent=args.dac_lr_ent,
        seed=args.seed,
    )
    prob = result['prob']
    path = result['path']
    print(f'DAC SOTA probability            : {prob:.4f}')
    print(f'DAC greedy path                 : {"  ".join(map(str, path))}')

    if args.eval_robust:
        q_table = result['q_table']
        successors = env.successors

        def policy_func(node, budget):
            acts = successors.get(node, [])
            if not acts:
                return None
            q_node = {(node, budget, a): q_table.get((node, budget, a), 0.0) for a in acts}
            return max(acts, key=lambda a: q_node.get((node, budget, a), 0.0))

        mc_sota = evaluate_policy(env, policy_func, episodes=args.eval_episodes)
        print(f'DAC SOTA probability (MC+exec)  : {mc_sota:.4f}  '
              f'[{args.eval_episodes} episodes, with execution uncertainty]')
        return prob, mc_sota

    return prob, None


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Parse explicit uncertain edges if provided
    uncertain_edges_explicit = None
    if args.uncertain_edges_explicit:
        uncertain_edges_explicit = {}
        for pair in args.uncertain_edges_explicit.split():
            u, v = pair.split(':')
            uncertain_edges_explicit[int(u)] = int(v)

    env = SiouxEnv(
        origin=args.origin,
        dest=args.dest,
        budget=args.budget,
        exec_prob=args.exec_prob,
        uncertain_ratio=args.uncertain_ratio,
        deterministic=args.deterministic,
        seed=args.seed,
        uncertain_seed=args.uncertain_seed,
        uncertain_mode=args.uncertain_mode,
        top_k=args.top_k,
        uncertain_edges_explicit=uncertain_edges_explicit,
    )

    agent = EURAC(
        env=env,
        lr_e=args.lr_e,
        lr_d=args.lr_d,
        lr_actor=args.lr_actor,
        entropy_coef=args.entropy_coef,
    )
    agent.warm_start(logit=args.warm_logit)

    print(f'Network : Sioux Falls  |  nodes={len(env.nodes)}  edges={len(env.edges)}')
    print(f'OD      : {args.origin}  {args.dest}  |  budget={args.budget} min')
    print(f'Uncertain nodes: {len(env.uncertain_edges)}  '
          f'(mode={args.uncertain_mode}, ratio={args.uncertain_ratio}, exec_prob={args.exec_prob})'
          + (f'  top_k={args.top_k}' if args.uncertain_mode == 'important' else ''))

    # ------------------------------------------------------------------
    # Baseline only
    # ------------------------------------------------------------------
    if args.baseline_only:
        if args.baseline == 'robust_routing':
            run_baseline_robust(env, args)
        elif args.baseline == 'dot':
            run_baseline_dot(env, args)
        elif args.baseline == 'pulse':
            run_baseline_pulse(env, args)
        elif args.baseline == 'otap':
            run_baseline_otap(env, args)
        elif args.baseline == 'ilp':
            run_baseline_ilp(env, args)
        elif args.baseline == 'gp3':
            run_baseline_gp3(env, args)
        elif args.baseline == 'pql':
            run_baseline_pql(env, args)
        elif args.baseline == 'ge_ddrl':
            run_baseline_ge_ddrl(env, args)
        elif args.baseline == 'segac':
            run_baseline_segac(env, args)
        elif args.baseline == 'dac':
            run_baseline_dac(env, args)
        return

    # ------------------------------------------------------------------
    # EU-RAC training
    # ------------------------------------------------------------------
    print(f'Training for {args.n_train} episodes ...\n')

    rewards = []
    csv_rows = []

    for ep in range(1, args.n_train + 1):
        r = agent.run_episode(train=True)
        rewards.append(r)

        if ep % args.report_every == 0:
            window = rewards[ep - args.report_every:ep]
            avg = float(np.mean(window))
            overall_avg = float(np.mean(rewards))
            avg_q_e, avg_q_d = agent.q_stats()
            print(f'  ep {ep:7d}  avg reward (last {args.report_every}) = {avg:.4f}  overall avg = {overall_avg:.4f}  avg_q_e={avg_q_e:.4f}  avg_q_d={avg_q_d:.4f}')
            csv_rows.append((ep, avg, overall_avg, avg_q_e, avg_q_d))

    # Final evaluation
    sota = agent.evaluate(n_episodes=args.n_eval)
    print(f'\nEU-RAC SOTA probability : {sota:.4f}')

    # Greedy path
    path = greedy_path(agent, env)
    print(f'Greedy path             : {"  ".join(map(str, path))}')

    # ------------------------------------------------------------------
    # Optional baseline comparison
    # ------------------------------------------------------------------
    if args.baseline == 'robust_routing':
        rr_sota, rr_mc = run_baseline_robust(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  RobustRouting(DP)={rr_sota:.4f}', end='')
        if rr_mc is not None:
            print(f'  RobustRouting(MC)={rr_mc:.4f}', end='')
        print()
    elif args.baseline == 'dot':
        dot_sota, dot_mc = run_baseline_dot(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  DOT(DP)={dot_sota:.4f}', end='')
        if dot_mc is not None:
            print(f'  DOT(MC)={dot_mc:.4f}', end='')
        print()
    elif args.baseline == 'pulse':
        pulse_sota, pulse_mc = run_baseline_pulse(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  Pulse={pulse_sota:.4f}', end='')
        if pulse_mc is not None:
            print(f'  Pulse(MC)={pulse_mc:.4f}', end='')
        print()
    elif args.baseline == 'otap':
        otap_sota, otap_mc = run_baseline_otap(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  OTAP={otap_sota:.4f}', end='')
        if otap_mc is not None:
            print(f'  OTAP(MC)={otap_mc:.4f}', end='')
        print()
    elif args.baseline == 'ilp':
        ilp_sota, ilp_mc = run_baseline_ilp(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  ILP={ilp_sota:.4f}', end='')
        if ilp_mc is not None:
            print(f'  ILP(MC)={ilp_mc:.4f}', end='')
        print()
    elif args.baseline == 'gp3':
        gp3_sota, gp3_mc = run_baseline_gp3(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  GP3={gp3_sota:.4f}', end='')
        if gp3_mc is not None:
            print(f'  GP3(MC)={gp3_mc:.4f}', end='')
        print()
    elif args.baseline == 'pql':
        pql_sota, pql_mc = run_baseline_pql(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  PQL={pql_sota:.4f}', end='')
        if pql_mc is not None:
            print(f'  PQL(MC+exec)={pql_mc:.4f}', end='')
        print()
    elif args.baseline == 'ge_ddrl':
        ge_sota, ge_mc = run_baseline_ge_ddrl(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  GE-DDRL={ge_sota:.4f}', end='')
        if ge_mc is not None:
            print(f'  GE-DDRL(MC)={ge_mc:.4f}', end='')
        print()
    elif args.baseline == 'segac':
        segac_sota, segac_mc = run_baseline_segac(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  SEGAC={segac_sota:.4f}', end='')
        if segac_mc is not None:
            print(f'  SEGAC(MC)={segac_mc:.4f}', end='')
        print()
    elif args.baseline == 'dac':
        dac_sota, dac_mc = run_baseline_dac(env, args)
        print(f'\nComparison  EU-RAC={sota:.4f}  vs  DAC={dac_sota:.4f}', end='')
        if dac_mc is not None:
            print(f'  DAC(MC+exec)={dac_mc:.4f}', end='')
        print()

    # Optional CSV save
    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward', 'overall_avg', 'avg_q_e', 'avg_q_d'])
            writer.writerows(csv_rows)
        print(f'Training curve saved to {args.save_csv}')

    # Optional training curve plot
    if args.plot:
        if not _HAS_MATPLOTLIB:
            print('matplotlib not installed; cannot show plot.')
        else:
            episodes = [row[0] for row in csv_rows]
            avg_rewards = [row[1] for row in csv_rows]
            overall_avgs = [row[2] for row in csv_rows]
            avg_q_e = [row[3] for row in csv_rows]
            avg_q_d = [row[4] for row in csv_rows]
            plot_training_curve(episodes, avg_rewards, overall_avgs, avg_q_e, avg_q_d)


if __name__ == '__main__':
    main()