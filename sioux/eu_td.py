"""
Execution-Uncertain Temporal-Difference (EU-TD) ablation for Sioux Falls.

This module keeps the execution kernel, travel-time sampling, budget logic, and
Monte Carlo evaluator from the existing Sioux implementation. It intentionally
contains no actor gradient, warm start, shortest-path repair, or expert data.
"""

import argparse
import csv
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluator import evaluate_policy
from sioux_env import SiouxEnv


class EUTD:
    def __init__(
        self,
        env,
        alpha_e=0.2,
        alpha_d=0.2,
        epsilon=0.05,
        seed=42,
    ):
        self.env = env
        self.alpha_e = alpha_e
        self.alpha_d = alpha_d
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q_d = defaultdict(float)
        self.Q_e = defaultdict(float)

    def _q_d(self, node, budget, action):
        return self.Q_d[(node, budget, action)]

    def _q_e(self, node, budget, executed_action):
        return self.Q_e[(node, budget, executed_action)]

    def _greedy_action(self, node, budget, actions):
        if not actions:
            return None
        q_values = np.asarray([self._q_d(node, budget, a) for a in actions], dtype=float)
        max_q = float(q_values.max())
        best_actions = [a for a, q in zip(actions, q_values) if np.isclose(q, max_q)]
        return int(self.rng.choice(best_actions))

    def policy_probs(self, node, budget):
        actions = self.env.get_actions(node)
        if not actions:
            return {}
        n_actions = len(actions)
        probs = {a: self.epsilon / n_actions for a in actions}
        q_values = np.asarray([self._q_d(node, budget, a) for a in actions], dtype=float)
        max_q = float(q_values.max())
        best_actions = [a for a, q in zip(actions, q_values) if np.isclose(q, max_q)]
        greedy_mass = (1.0 - self.epsilon) / len(best_actions)
        for action in best_actions:
            probs[action] += greedy_mass
        return probs

    def select_action(self, node, budget):
        actions = self.env.get_actions(node)
        if not actions:
            return None
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        return int(self._greedy_action(node, budget, actions))

    def compute_v(self, node, budget):
        if node == self.env.dest:
            return 1.0 if budget >= 0 else 0.0
        if budget < 0:
            return 0.0
        probs = self.policy_probs(node, budget)
        return float(sum(p * self._q_d(node, budget, a) for a, p in probs.items()))

    def env_step(self, node, budget, intended_action):
        executed = self.env.sample_executed_action(node, intended_action)
        travel_time = self.env.sample_travel_time(node, executed)
        next_budget = budget - travel_time
        return executed, executed, next_budget

    def update(self, node, budget, intended_action, executed_action, next_node, next_budget):
        if next_node == self.env.dest and next_budget >= 0:
            y_e = 1.0
        elif next_budget < 0:
            y_e = 0.0
        else:
            y_e = self.compute_v(next_node, next_budget)

        key_e = (node, budget, executed_action)
        self.Q_e[key_e] += self.alpha_e * (y_e - self.Q_e[key_e])

        y_d = self.Q_e[key_e]
        key_d = (node, budget, intended_action)
        self.Q_d[key_d] += self.alpha_d * (y_d - self.Q_d[key_d])

    def run_episode(self, train=True):
        node = self.env.origin
        budget = self.env.budget
        max_steps = len(self.env.nodes) * 3
        steps = 0

        for _ in range(max_steps):
            if node == self.env.dest:
                return 1.0 if budget >= 0 else 0.0, steps
            if budget < 0:
                return 0.0, steps

            intended = self.select_action(node, budget)
            if intended is None:
                return 0.0, steps

            executed, next_node, next_budget = self.env_step(node, budget, intended)
            if train:
                self.update(node, budget, intended, executed, next_node, next_budget)

            node, budget = next_node, next_budget
            steps += 1

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
        return int(self._greedy_action(node, budget, actions))

    def has_nan(self):
        values = list(self.Q_d.values()) + list(self.Q_e.values())
        return bool(values and np.isnan(np.asarray(values, dtype=float)).any())


def run_eu_td_routing(
    env,
    budget=None,
    episodes=5000,
    alpha_e=0.2,
    alpha_d=0.2,
    epsilon=0.05,
    seed=42,
    report_every=1000,
    eval_episodes=1000,
):
    if budget is not None:
        env.budget = budget
    agent = EUTD(env, alpha_e=alpha_e, alpha_d=alpha_d, epsilon=epsilon, seed=seed)
    curve, rewards, lengths = agent.train(
        episodes=episodes,
        report_every=report_every,
        eval_episodes=eval_episodes,
    )
    mc_prob = evaluate_policy(env, agent.policy, episodes=eval_episodes)
    return {
        "policy": agent.policy,
        "Q_d": dict(agent.Q_d),
        "Q_e": dict(agent.Q_e),
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
    parser = argparse.ArgumentParser(description="Train EU-TD on Sioux Falls.")
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
    parser.add_argument("--alpha-e", type=float, default=0.2)
    parser.add_argument("--alpha-d", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=str, default=str(Path(__file__).with_name("eu_td_learning_curve.csv")))
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
    result = run_eu_td_routing(
        env,
        episodes=args.episodes,
        alpha_e=args.alpha_e,
        alpha_d=args.alpha_d,
        epsilon=args.epsilon,
        seed=args.seed,
        report_every=args.report_every,
        eval_episodes=args.eval_episodes,
    )
    _write_curve(args.save_csv, result["learning_curve"])
    print(f"EU-TD training success rate: {result['training_success_rate']:.4f}")
    print(f"EU-TD average episode length: {result['average_episode_length']:.2f}")
    print(f"EU-TD MC probability: {result['MC']:.4f}")
    print(f"EU-TD NaN check: {result['has_nan']}")
    print(f"Learning curve saved to {args.save_csv}")


if __name__ == "__main__":
    main()