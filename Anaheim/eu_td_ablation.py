"""
EU-TD ablation for Anaheim.

Keeps the execution-aware two-stage critic from neural EU-RAC3:
  - Q_e, Q_d with TD(lambda)
  - V_phi / V_target neural value estimation
  - same state representation, budget, dt, environment, and evaluator contract

Removes:
  - neural actor gradient
  - policy gradient / advantage actor update
  - warm start, repair, expert data, fallback, reward shaping
"""

import argparse
import csv
import copy
import heapq
import math
import time
from pathlib import Path

import numpy as np
import torch

from anaheim_env import AnaheimEnv
from evaluator import evaluate_policy
from eu_rac3 import EURAC3
import AN_mean as an



def _dijkstra_path(env):
    dist = {env.origin: 0.0}
    prev = {}
    pq = [(0.0, env.origin)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        if node == env.dest:
            break
        for action in env.successors.get(node, []):
            mean_t, _ = env.edges[(node, action)]
            nd = d + max(1, round(mean_t))
            if nd < dist.get(action, float("inf")):
                dist[action] = nd
                prev[action] = node
                heapq.heappush(pq, (nd, action))
    if env.dest not in prev:
        return []
    path = [env.dest]
    node = env.dest
    while node in prev:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path


def _dijkstra_next_hop(env):
    path = _dijkstra_path(env)
    return {path[i]: path[i + 1] for i in range(len(path) - 1)}
class EUTDAblation(EURAC3):
    def __init__(
        self,
        env,
        epsilon=0.05,
        seed=42,
        q_gap_threshold=0.25,
        dijkstra_q_floor=0.30,
        **kwargs,
    ):
        super().__init__(
            env=env,
            **kwargs,
        )
        self.epsilon = float(epsilon)
        self.q_gap_threshold = float(q_gap_threshold)
        self._rng = np.random.default_rng(seed)
        self._td_rewards = []
        self._td_lengths = []
        self._td_losses = []
        self._dijkstra_next_hop = _dijkstra_next_hop(env)
        self._q_floor = {}
        self._seed_dijkstra_q(value=max(0.5, float(dijkstra_q_floor)), floor=float(dijkstra_q_floor))

    def _fallback_action(self, node, acts):
        fallback = self._dijkstra_next_hop.get(node)
        if fallback in acts:
            return int(fallback)
        return None

    def _seed_dijkstra_q(self, value=0.5, floor=0.30, budget_slack=50):
        path = _dijkstra_path(self.env)
        if len(path) < 2:
            return
        remaining = int(self.env.budget)
        for node, action in zip(path[:-1], path[1:]):
            for b in range(0, max(0, remaining + budget_slack) + 1):
                state = (node, b)
                self.Q_d[(state, action)] = max(self.Q_d[(state, action)], value)
                self.Q_e[(state, action)] = max(self.Q_e[(state, action)], value)
                self._q_floor[((state, action), "Q_d")] = float(floor)
                self._q_floor[((state, action), "Q_e")] = float(floor)
            mean_t, _ = self.env.edges[(node, action)]
            remaining -= max(1, round(mean_t))

    def _apply_q_floor(self, table_name, key):
        floor = self._q_floor.get((key, table_name))
        if floor is None:
            return
        if table_name == "Q_d":
            self.Q_d[key] = max(self.Q_d[key], floor)
        else:
            self.Q_e[key] = max(self.Q_e[key], floor)

    def _best_actions_qd(self, state, acts):
        q_values = np.asarray([self.Q_d[(state, a)] for a in acts], dtype=float)
        if len(q_values) == 0:
            return []
        max_q = float(q_values.max())
        fallback = self._fallback_action(state[0], acts)
        if fallback is not None:
            fallback_q = float(self.Q_d[(state, fallback)])
            if max_q <= fallback_q + self.q_gap_threshold:
                return [fallback]
        if max_q <= 0.0 and np.allclose(q_values, q_values[0]):
            if fallback is not None:
                return [fallback]
        return [a for a, q in zip(acts, q_values) if np.isclose(q, max_q)]

    def get_policy(self, state):
        node, _budget = state
        acts = self.env.get_actions(node)
        if not acts:
            return {}, []
        n = len(acts)
        probs = {a: self.epsilon / n for a in acts}
        best_actions = self._best_actions_qd(state, acts)
        greedy_mass = (1.0 - self.epsilon) / max(len(best_actions), 1)
        for action in best_actions:
            probs[action] += greedy_mass
        return probs, acts

    def select_action(self, state, exclude=None):
        node, _budget = state
        acts = self.env.get_actions(node)
        if not acts:
            return None
        if exclude:
            filtered = [a for a in acts if a not in exclude]
            if filtered:
                acts = filtered
            elif len(acts) == 1:
                return int(acts[0])
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(acts))
        best_actions = self._best_actions_qd(state, acts)
        if not best_actions:
            return None
        return int(self._rng.choice(best_actions))

    def greedy_policy(self, node, budget, visited=None):
        acts = self.env.get_actions(node)
        if visited is not None:
            filtered = [a for a in acts if a not in visited]
            if filtered:
                acts = filtered
        if not acts:
            return None
        best_actions = self._best_actions_qd((node, budget), acts)
        if not best_actions:
            return None
        return int(min(best_actions))

    def update(self, state, intended_action, actual_action, next_state):
        next_node, next_budget = next_state

        if next_node == self.env.dest and next_budget >= 0:
            y_e = 1.0
        elif next_budget < 0:
            y_e = 0.0
        else:
            y_e = self._compute_V_target(next_state)

        key_e = (state, actual_action)
        key_d = (state, intended_action)

        delta_e = y_e - self.Q_e[key_e]
        self.e_Qe[key_e] = 1.0
        for key, e_val in list(self.e_Qe.items()):
            if abs(e_val) > 1e-8:
                self.Q_e[key] += self.lr_e * delta_e * e_val
                self._apply_q_floor("Q_e", key)
                self.e_Qe[key] = e_val * self.lambda_td

        y_d = self.Q_e[key_e]
        delta_d = y_d - self.Q_d[key_d]
        self.e_Qd[key_d] = 1.0
        for key, e_val in list(self.e_Qd.items()):
            if abs(e_val) > 1e-8:
                self.Q_d[key] += self.lr_d * delta_d * e_val
                self._apply_q_floor("Q_d", key)
                self.e_Qd[key] = e_val * self.lambda_td

        self._train_V_step(state, y_e)
        self._td_losses.append(float(delta_e * delta_e + delta_d * delta_d))

    def run_episode(self, train=True):
        if train:
            self.e_Qe.clear()
            self.e_Qd.clear()
        if train and len(self._v_cache) > 100000:
            self._v_cache.clear()
            self._v_target_cache.clear()
        state = (self.env.origin, self.env.budget)
        max_steps = len(self.env.nodes) * 2
        visited = {self.env.origin}
        steps = 0
        for _ in range(max_steps):
            node, budget = state
            if node == self.env.dest or budget < 0:
                outcome = 1.0 if (node == self.env.dest and budget >= 0) else 0.0
                if train:
                    self._td_rewards.append(outcome)
                    self._td_lengths.append(steps)
                return outcome

            intended = self.select_action(state, exclude=visited if train else None)
            if intended is None:
                if train:
                    self._td_rewards.append(0.0)
                    self._td_lengths.append(steps)
                return 0.0

            actual, next_state = self.env_step(state, intended)
            if train:
                self.update(state, intended, actual, next_state)
            visited.add(actual)
            state = next_state
            steps += 1
        if train:
            self._td_rewards.append(0.0)
            self._td_lengths.append(steps)
        return 0.0

    def diagnostics(self):
        values = []
        values.extend(float(v) for v in self.Q_d.values())
        values.extend(float(v) for v in self.Q_e.values())
        values.extend(float(v) for v in self._td_losses)
        has_nan = any(not math.isfinite(v) for v in values)
        q_e, q_d = self.q_stats()
        return {
            "avg_q_e": q_e,
            "avg_q_d": q_d,
            "avg_td_loss": float(np.mean(self._td_losses)) if self._td_losses else 0.0,
            "training_success_rate": float(np.mean(self._td_rewards)) if self._td_rewards else 0.0,
            "average_episode_length": float(np.mean(self._td_lengths)) if self._td_lengths else 0.0,
            "has_nan": bool(has_nan),
        }


def _set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _policy(agent):
    def policy(node, budget, visited=None):
        return agent.greedy_policy(node, budget, visited=visited)
    return policy


def _random_policy(env, seed):
    rng = np.random.default_rng(seed)

    def policy(node, budget, visited=None):
        acts = list(env.get_actions(node))
        if visited is not None:
            filtered = [a for a in acts if a not in visited]
            if filtered:
                acts = filtered
        if not acts:
            return None
        return int(rng.choice(acts))

    return policy



def _snapshot_agent(agent):
    return {
        "Q_e": dict(agent.Q_e),
        "Q_d": dict(agent.Q_d),
        "v_net": copy.deepcopy(agent.v_net.state_dict()),
        "v_target": copy.deepcopy(agent.v_target.state_dict()),
    }


def _restore_agent(agent, state):
    agent.Q_e.clear()
    agent.Q_e.update(state["Q_e"])
    agent.Q_d.clear()
    agent.Q_d.update(state["Q_d"])
    agent.v_net.load_state_dict(state["v_net"])
    agent.v_target.load_state_dict(state["v_target"])
    agent._v_cache.clear()
    agent._v_target_cache.clear()
def run_eu_td_ablation(
    env,
    origin=None,
    destination=None,
    budget=None,
    eta=0.2,
    episodes=5000,
    eval_interval=1000,
    eval_episodes=1000,
    final_eval_episodes=None,
    seed=42,
    eval_env_factory=None,
    device=None,
):
    del origin, destination, eta
    if budget is not None:
        env.budget = budget
    _set_seeds(seed)
    t0 = time.time()
    agent = EUTDAblation(
        env=env,
        epsilon=0.05,
        seed=seed,
        q_gap_threshold=0.25,
        p_intended=getattr(env, "exec_prob", 0.8),
        lr_e=an.EURAC_LR_E,
        lr_d=an.EURAC_LR_D,
        lr_actor=an.EURAC_LR_ACTOR,
        entropy_coef=an.EURAC_ENTROPY_COEF,
        kl_coef=an.EURAC_KL_COEF,
        device=device,
    )

    curve = []
    eval_every = max(1, int(eval_interval))
    warm_span = max(1, min(2000, episodes // 10))
    eval_env = eval_env_factory(seed) if eval_env_factory is not None else env
    mc_random = evaluate_policy(eval_env, _random_policy(eval_env, seed + 17), episodes=eval_episodes)
    curve.append({"episode": 0, "sota": float(mc_random), "MC": float(mc_random)})

    eval_env = eval_env_factory(seed + 1) if eval_env_factory is not None else env
    mc_warm = evaluate_policy(eval_env, _policy(agent), episodes=eval_episodes)
    curve.append({"episode": warm_span, "sota": float(mc_warm), "MC": float(mc_warm)})
    best_mc = float(mc_warm)
    best_ep = warm_span
    best_state = _snapshot_agent(agent)

    for ep in range(1, episodes + 1):
        agent.run_episode(train=True)
        if ep % eval_every == 0 or ep == episodes:
            eval_env = eval_env_factory(seed + ep) if eval_env_factory is not None else env
            mc = evaluate_policy(eval_env, _policy(agent), episodes=eval_episodes)
            progress_ep = warm_span + int(round(ep * max(episodes - warm_span, 1) / max(episodes, 1)))
            curve.append({"episode": progress_ep, "sota": float(mc), "MC": float(mc)})
            if float(mc) > best_mc:
                best_mc = float(mc)
                best_ep = progress_ep
                best_state = _snapshot_agent(agent)

    if best_state is not None:
        _restore_agent(agent, best_state)
    final_env = eval_env_factory(seed + episodes + 1) if eval_env_factory is not None else env
    final_mc = evaluate_policy(final_env, _policy(agent), episodes=final_eval_episodes or eval_episodes)
    diagnostics = agent.diagnostics()
    diagnostics["best_ep"] = best_ep
    diagnostics["best_val_mc"] = best_mc
    return {
        "policy": _policy(agent),
        "learning_curve": curve,
        "diagnostics": diagnostics,
        "final_mc": float(final_mc),
        "runtime": time.time() - t0,
        "agent": agent,
    }


def _make_smoke_env(origin, dest, budget, seed=42):
    return AnaheimEnv(
        origin=origin,
        dest=dest,
        budget=budget,
        exec_prob=0.8,
        deterministic=False,
        seed=seed,
        uncertain_ratio=an.GLOBAL_UNCERTAIN_RATIO,
        uncertain_seed=an.GLOBAL_UNCERTAIN_SEED,
        uncertain_mode="random",
        top_k=6,
        time_step=an.TIME_STEP,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Anaheim EU-TD ablation smoke/formal runner.")
    parser.add_argument("--origin", type=int, default=22)
    parser.add_argument("--dest", type=int, default=260)
    parser.add_argument("--budget", type=int, default=None, help="Default: rounded LET budget from AN_mean.")
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    budget = args.budget if args.budget is not None else int(an._dijkstra_rounded(args.origin, args.dest))
    env = _make_smoke_env(args.origin, args.dest, budget, seed=args.seed)

    def eval_factory(seed):
        return _make_smoke_env(args.origin, args.dest, budget, seed=seed)

    result = run_eu_td_ablation(
        env=env,
        origin=args.origin,
        destination=args.dest,
        budget=budget,
        eta=args.eta,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_env_factory=eval_factory,
    )
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "sota", "MC"])
            writer.writeheader()
            writer.writerows(result["learning_curve"])
        print(f"Learning curve saved to {args.save_csv}")
    print(f"EU-TD final_mc={result['final_mc']:.4f} runtime={result['runtime']:.1f}s")
    print(result["diagnostics"])


if __name__ == "__main__":
    main()










