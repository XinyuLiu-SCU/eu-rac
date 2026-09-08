"""
run_ablation_td_pg.py - EU-TD / EU-PG ablation runner aligned with sioux_OD.py.

Protocol:
  - ODs and explicit uncertain edges come from sioux_OD.OD_CONFIGS.
  - Only eta=0.2, B_label=1.0 are run by default.
  - budget = floor(1.0 * t_LET), matching sioux_OD.py.
  - Training env seed and MC eval seed match sioux_OD.py.
  - Each OD/method learning curve is saved as episode,MC.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from sioux_env import SiouxEnv
from evaluator import evaluate_policy
from eu_td import EUTD
from eu_pg import EUPG
from sioux_OD import (
    OD_CONFIGS,
    GLOBAL_UNCERTAIN_RATIO,
    SEED,
    _GLOBAL_UNCERTAIN_EDGES,
    _dijkstra_mean,
)

OUTPUT_DIR = Path(__file__).parent
CURVE_DIR = OUTPUT_DIR / "ablation_curves"
DEFAULT_ETA = 0.2
DEFAULT_B_COEFF = 1.0
DEFAULT_TD_EPISODES = 200000
DEFAULT_PG_EPISODES = 200000
DEFAULT_EVAL_EPISODES = 10000
DEFAULT_REPORT_EVERY = 10000


def _apply_global_uncertainty(env, eta):
    if eta > 0 and GLOBAL_UNCERTAIN_RATIO > 0:
        for node, edge in _GLOBAL_UNCERTAIN_EDGES.items():
            if node not in env.uncertain_edges:
                env.uncertain_edges[node] = edge


def make_train_env(origin, dest, budget, eta, uncertain_edges, combo_seed):
    env = SiouxEnv(
        origin=origin,
        dest=dest,
        budget=budget,
        exec_prob=1.0 - eta,
        deterministic=False,
        seed=combo_seed,
        uncertain_edges_explicit=uncertain_edges,
    )
    _apply_global_uncertainty(env, eta)
    return env


def make_mc_env(origin, dest, budget, eta, uncertain_edges):
    mc_eval_seed = 999999 + origin * 100 + dest
    env = SiouxEnv(
        origin=origin,
        dest=dest,
        budget=budget,
        exec_prob=1.0 - eta,
        deterministic=False,
        seed=mc_eval_seed,
        uncertain_edges_explicit=uncertain_edges,
    )
    _apply_global_uncertainty(env, eta)
    return env


def mc_eval(origin, dest, budget, eta, uncertain_edges, policy_func, episodes):
    return evaluate_policy(
        make_mc_env(origin, dest, budget, eta, uncertain_edges),
        policy_func,
        episodes=episodes,
    )


def save_curve(path, curve):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "MC"])
        writer.writeheader()
        writer.writerows(curve)


def train_td(origin, dest, budget, eta, uncertain_edges, combo_seed, episodes, eval_episodes, report_every):
    env = make_train_env(origin, dest, budget, eta, uncertain_edges, combo_seed)
    agent = EUTD(env, alpha_e=0.2, alpha_d=0.2, epsilon=0.05, seed=combo_seed)
    rewards = []
    lengths = []
    curve = []

    for ep in range(1, episodes + 1):
        reward, steps = agent.run_episode(train=True)
        rewards.append(reward)
        lengths.append(steps)
        if ep % report_every == 0 or ep == episodes:
            mc = mc_eval(origin, dest, budget, eta, uncertain_edges, agent.policy, eval_episodes)
            curve.append({"episode": ep, "MC": float(mc)})

    final_mc = mc_eval(origin, dest, budget, eta, uncertain_edges, agent.policy, eval_episodes)
    return {
        "MC": float(final_mc),
        "training_success_rate": float(np.mean(rewards)) if rewards else 0.0,
        "average_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "has_nan": agent.has_nan(),
        "learning_curve": curve,
    }


def train_pg(origin, dest, budget, eta, uncertain_edges, combo_seed, episodes, eval_episodes, report_every):
    env = make_train_env(origin, dest, budget, eta, uncertain_edges, combo_seed)
    agent = EUPG(env, lr=0.05, seed=combo_seed)
    rewards = []
    lengths = []
    curve = []

    for ep in range(1, episodes + 1):
        reward, steps = agent.run_episode(train=True)
        rewards.append(reward)
        lengths.append(steps)
        if ep % report_every == 0 or ep == episodes:
            mc = mc_eval(origin, dest, budget, eta, uncertain_edges, agent.policy, eval_episodes)
            curve.append({"episode": ep, "MC": float(mc)})

    final_mc = mc_eval(origin, dest, budget, eta, uncertain_edges, agent.policy, eval_episodes)
    return {
        "MC": float(final_mc),
        "training_success_rate": float(np.mean(rewards)) if rewards else 0.0,
        "average_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "has_nan": agent.has_nan(),
        "learning_curve": curve,
    }


def run_one_od(origin, dest, uncertain_edges, args):
    t_let = _dijkstra_mean(origin, dest)
    budget = int(math.floor(args.b_coeff * t_let))
    eta_tag = str(args.eta).replace(".", "p")
    b_tag = str(args.b_coeff).replace(".", "p")
    combo_seed = SEED + origin * 100 + dest + int(args.eta * 100) * 10000 + budget

    print(f"\nOD {origin}->{dest} | t_LET={t_let:.4f} | B={budget} ({args.b_coeff}x) | eta={args.eta}")
    print(f"uncertain_edges={uncertain_edges} | combo_seed={combo_seed}")

    t0 = time.time()
    td = train_td(
        origin, dest, budget, args.eta, uncertain_edges, combo_seed,
        args.td_episodes, args.eval_episodes, args.report_every,
    )
    td_curve = CURVE_DIR / f"eu_td_OD{origin}_{dest}_B{b_tag}_eta{eta_tag}.csv"
    save_curve(td_curve, td["learning_curve"])
    print(f"  EU-TD MC={td['MC']:.4f} train={td['training_success_rate']:.4f} len={td['average_episode_length']:.2f} nan={td['has_nan']} ({time.time() - t0:.0f}s)")

    t1 = time.time()
    pg = train_pg(
        origin, dest, budget, args.eta, uncertain_edges, combo_seed,
        args.pg_episodes, args.eval_episodes, args.report_every,
    )
    pg_curve = CURVE_DIR / f"eu_pg_OD{origin}_{dest}_B{b_tag}_eta{eta_tag}.csv"
    save_curve(pg_curve, pg["learning_curve"])
    print(f"  EU-PG MC={pg['MC']:.4f} train={pg['training_success_rate']:.4f} len={pg['average_episode_length']:.2f} nan={pg['has_nan']} ({time.time() - t1:.0f}s)")

    return {
        "origin": origin,
        "dest": dest,
        "B_label": args.b_coeff,
        "eta": args.eta,
        "t_LET": t_let,
        "budget": budget,
        "EU-TD": td["MC"],
        "EU-PG": pg["MC"],
        "TD_train_success": td["training_success_rate"],
        "PG_train_success": pg["training_success_rate"],
        "TD_avg_len": td["average_episode_length"],
        "PG_avg_len": pg["average_episode_length"],
        "TD_nan": td["has_nan"],
        "PG_nan": pg["has_nan"],
        "TD_curve": str(td_curve.relative_to(OUTPUT_DIR)),
        "PG_curve": str(pg_curve.relative_to(OUTPUT_DIR)),
    }


def save_rows(path, rows):
    fieldnames = [
        "origin", "dest", "B_label", "eta", "t_LET", "budget",
        "EU-TD", "EU-PG", "TD_train_success", "PG_train_success",
        "TD_avg_len", "PG_avg_len", "TD_nan", "PG_nan", "TD_curve", "PG_curve",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_mean(path, rows):
    mean_row = {
        "B_label": rows[0]["B_label"] if rows else "",
        "eta": rows[0]["eta"] if rows else "",
        "n_od": len(rows),
        "EU-TD_mean": float(np.mean([r["EU-TD"] for r in rows])) if rows else 0.0,
        "EU-PG_mean": float(np.mean([r["EU-PG"] for r in rows])) if rows else 0.0,
        "TD_train_success_mean": float(np.mean([r["TD_train_success"] for r in rows])) if rows else 0.0,
        "PG_train_success_mean": float(np.mean([r["PG_train_success"] for r in rows])) if rows else 0.0,
        "TD_avg_len_mean": float(np.mean([r["TD_avg_len"] for r in rows])) if rows else 0.0,
        "PG_avg_len_mean": float(np.mean([r["PG_avg_len"] for r in rows])) if rows else 0.0,
        "TD_nan_any": any(r["TD_nan"] for r in rows),
        "PG_nan_any": any(r["PG_nan"] for r in rows),
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mean_row.keys()))
        writer.writeheader()
        writer.writerow(mean_row)
    return mean_row


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sioux EU-TD/EU-PG ablations aligned with sioux_OD.py.")
    parser.add_argument("--od", type=int, nargs=2, metavar=("ORIGIN", "DEST"), help="Run one OD only.")
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--b-coeff", type=float, default=DEFAULT_B_COEFF)
    parser.add_argument("--td-episodes", type=int, default=DEFAULT_TD_EPISODES)
    parser.add_argument("--pg-episodes", type=int, default=DEFAULT_PG_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--report-every", type=int, default=DEFAULT_REPORT_EVERY)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "ablation_td_pg_eta02_B1.csv"))
    parser.add_argument("--mean-output", type=str, default=str(OUTPUT_DIR / "ablation_td_pg_eta02_B1_mean.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    if args.od:
        od_key = tuple(args.od)
        od_items = [(od_key, OD_CONFIGS.get(od_key, {}))]
    else:
        od_items = list(OD_CONFIGS.items())

    rows = []
    for (origin, dest), uncertain_edges in od_items:
        rows.append(run_one_od(origin, dest, uncertain_edges, args))

    output_path = Path(args.output)
    mean_path = Path(args.mean_output)
    save_rows(output_path, rows)
    mean_row = save_mean(mean_path, rows)

    print(f"\nSaved per-OD ablation results: {output_path}")
    print(f"Saved mean ablation results: {mean_path}")
    print(f"Mean EU-TD={mean_row['EU-TD_mean']:.4f} | Mean EU-PG={mean_row['EU-PG_mean']:.4f}")


if __name__ == "__main__":
    main()