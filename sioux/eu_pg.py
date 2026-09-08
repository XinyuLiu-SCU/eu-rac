"""
Trajectory-level Execution-Uncertain Policy Gradient (EU-PG) for Sioux Falls.

This is the policy-gradient ablation of EU-RAC. It uses the existing execution
kernel and travel-time sampling, and deliberately has no critic, baseline,
advantage estimate, warm start, shortest-path repair, or expert data.
"""

import argparse
import csv
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluator import evaluate_policy
from sioux_env import SiouxEnv


class EUPG:
    def __init__(self, env, lr=0.05, seed=42):
        self.env = env
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.theta = defaultdict(float)

    def _softmax(self, node, budget, actions=None):
        if actions is None:
            actions = self.env.get_actions(node)
        if not actions:
            return [], np.asarray([], dtype=float)
        logits = np.asarray([self.theta[(node, budget, a)] for a in actions], dtype=float)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        return actions, probs

    def select_action(self, node, budget):
        actions, probs = self._softmax(node, budget)
        if not actions:
            return None, None, None, None
        idx = int(self.rng.choice(len(actions), p=probs))
        return int(actions[idx]), float(np.log(probs[idx] + 1e-15)), list(actions), probs.copy()

    def update_from_trajectory(self, trajectory, reward):
        if reward == 0.0:
            return
        for node, budget, action, _log_prob, actions, probs in trajectory:
            for idx, a in enumerate(actions):
                grad_log_pi = (1.0 if a == action else 0.0) - probs[idx]
                self.theta[(node, budget, a)] += self.lr * reward * grad_log_pi

    def run_episode(self, train=True):
        node = self.env.origin
        budget = self.env.budget
        max_steps = len(self.env.nodes) * 3
        trajectory = []
        steps = 0

        for _ in range(max_steps):
            if node == self.env.dest:
                reward = 1.0 if budget >= 0 else 0.0
                if train:
                    self.update_from_trajectory(trajectory, reward)
                return reward, steps
            if budget < 0:
                if train:
                    self.update_from_trajectory(trajectory, 0.0)
                return 0.0, steps

            intended, log_prob, actions, probs = self.select_action(node, budget)
            if intended is None:
                if train:
                    self.update_from_trajectory(trajectory, 0.0)
                return 0.0, steps

            trajectory.append((node, budget, intended, log_prob, actions, probs))
            executed = self.env.sample_executed_action(node, intended)
            travel_time = self.env.sample_travel_time(node, executed)
            node, budget = executed, budget - travel_time
            steps += 1

        if train:
            self.update_from_trajectory(trajectory, 0.0)
        return 0.0, steps

    def train(self, episodes=5000, report_every=1000, eval_episodes=1000):
        rewards = []
        lengths = []
        curve = []
        for ep in range(1, episodes + 1):
            reward, steps = self.run_episode(train=True)
            rewards.append(reward)
            lengths.append(steps)
            if report_every and (ep % report_every == 0 or ep == episodes):
                mc = evaluate_policy(self.env, self.policy, episodes=eval_episodes)
                curve.append({"episode": ep, "MC": float(mc)})
        return curve, rewards, lengths

    def policy(self, node, budget, visited=None):
        actions = self.env.get_actions(node)
        if visited is not None:
            filtered = [a for a in actions if a not in visited]
            if filtered:
                actions = filtered
        if not actions:
            return None
        actions, probs = self._softmax(node, budget, actions)
        return int(actions[int(np.argmax(probs))])

    def has_nan(self):
        values = list(self.theta.values())
        return bool(values and np.isnan(np.asarray(values, dtype=float)).any())


def run_eu_pg_routing(
    env,
    budget=None,
    episodes=5000,
    lr=0.05,
    seed=42,
    report_every=1000,
    eval_episodes=1000,
):
    if budget is not None:
        env.budget = budget
    agent = EUPG(env, lr=lr, seed=seed)
    curve, rewards, lengths = agent.train(
        episodes=episodes,
        report_every=report_every,
        eval_episodes=eval_episodes,
    )
    mc_prob = evaluate_policy(env, agent.policy, episodes=eval_episodes)
    return {
        "policy": agent.policy,
        "learning_curve": curve,
        "training_success_rate": float(np.mean(rewards)) if rewards else 0.0,
        "average_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "MC": float(mc_prob),
        "has_nan": agent.has_nan(),
        "agent": agent,
    }


def _mean_shortest_time(env, origin, dest):
    dist = {origin: 0.0}
    pq = [(0.0, origin)]
    while pq:
        cost, node = heapq.heappop(pq)
        if node == dest:
            return cost
        if cost > dist.get(node, float("inf")):
            continue
        for nxt in env.get_actions(node):
            mean_t, _ = env.edges[(node, nxt)]
            new_cost = cost + mean_t
            if new_cost < dist.get(nxt, float("inf")):
                dist[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt))
    raise ValueError(f"No mean-time path from {origin} to {dest}")


def _write_curve(path, curve):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "MC"])
        writer.writeheader()
        writer.writerows(curve)


def parse_args():
    origin, dest = SiouxEnv.default_od()
    parser = argparse.ArgumentParser(description="Train EU-PG on Sioux Falls.")
    parser.add_argument("--origin", type=int, default=origin)
    parser.add_argument("--dest", type=int, default=dest)
    parser.add_argument("--budget", type=int, default=None, help="Default: round(t_LET)")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--exec-prob", type=float, default=0.8)
    parser.add_argument("--uncertain-ratio", type=float, default=0.2)
    parser.add_argument("--uncertain-seed", type=int, default=42)
    parser.add_argument("--uncertain-mode", choices=["random", "important"], default="random")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=str, default=str(Path(__file__).with_name("eu_pg_learning_curve.csv")))
    return parser.parse_args()


def main():
    args = parse_args()
    probe_env = SiouxEnv(origin=args.origin, dest=args.dest, budget=1)
    budget = args.budget if args.budget is not None else round(_mean_shortest_time(probe_env, args.origin, args.dest))
    env = SiouxEnv(
        origin=args.origin,
        dest=args.dest,
        budget=budget,
        exec_prob=args.exec_prob,
        uncertain_ratio=args.uncertain_ratio,
        deterministic=args.deterministic,
        seed=args.seed,
        uncertain_seed=args.uncertain_seed,
        uncertain_mode=args.uncertain_mode,
        top_k=args.top_k,
    )
    result = run_eu_pg_routing(
        env,
        episodes=args.episodes,
        lr=args.lr,
        seed=args.seed,
        report_every=args.report_every,
        eval_episodes=args.eval_episodes,
    )
    _write_curve(args.save_csv, result["learning_curve"])
    print(f"EU-PG training success rate: {result['training_success_rate']:.4f}")
    print(f"EU-PG average episode length: {result['average_episode_length']:.2f}")
    print(f"EU-PG MC probability: {result['MC']:.4f}")
    print(f"EU-PG NaN check: {result['has_nan']}")
    print(f"Learning curve saved to {args.save_csv}")


if __name__ == "__main__":
    main()