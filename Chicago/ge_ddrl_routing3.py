"""
ge_ddrl_routing3.py - Chicago-adapted GE-DDRL.

This is a Chicago-adapted GE-DDRL implementation.
It retains the tabular C51 training mechanism but adds
destination-feasibility filtering and limited next-hop fallback
for large-scale bidirectional networks.

This is the frozen Chicago-adapted GE-DDRL baseline, not a strict faithful DRL_C51 reference. The original reference migration remains in ge_ddrl_reference.py for audit only. This implementation preserves the core tabular distributional Bellman updates while adapting the routing interface, feasibility filtering, and path extraction to the Chicago benchmark.
"""

import heapq
import math
from collections import defaultdict

import numpy as np


def _project_point(r, N, delta_w):
    z_max = (N - 1) * delta_w
    tz = min(z_max, max(0.0, float(r)))
    bj = tz / delta_w
    lo = min(int(math.floor(bj)), N - 1)
    hi = min(int(math.ceil(bj)), N - 1)
    out = np.zeros(N, dtype=float)
    if lo == hi:
        out[lo] = 1.0
    else:
        out[lo] = hi - bj
        out[hi] = bj - lo
    return out


def _project_shift(r, dist, N, delta_w):
    z_max = (N - 1) * delta_w
    atoms = np.arange(N, dtype=float) * delta_w
    shifted = np.clip(float(r) + atoms, 0.0, z_max)
    bj = shifted / delta_w
    lo = np.clip(np.floor(bj).astype(int), 0, N - 1)
    hi = np.clip(np.ceil(bj).astype(int), 0, N - 1)
    out = np.zeros(N, dtype=float)
    same = lo == hi
    np.add.at(out, lo[same], dist[same])
    diff = ~same
    np.add.at(out, lo[diff], dist[diff] * (hi[diff] - bj[diff]))
    np.add.at(out, hi[diff], dist[diff] * (bj[diff] - lo[diff]))
    return out


def _project_point_samples(rewards, N, delta_w):
    out = np.zeros(N, dtype=float)
    for r in rewards:
        out += _project_point(r, N, delta_w)
    return out / len(rewards)


def _project_shift_samples(rewards, dist, N, delta_w):
    out = np.zeros(N, dtype=float)
    for r in rewards:
        out += _project_shift(r, dist, N, delta_w)
    return out / len(rewards)


def _dijkstra_to_dest(edges, successors, dest):
    reverse = defaultdict(list)
    for (u, v), (mean_t, _) in edges.items():
        reverse[v].append((u, max(1, round(mean_t))))
    dist = {dest: 0.0}
    next_action = {}
    pq = [(0.0, dest)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for pred, cost in reverse.get(u, []):
            nd = d + cost
            if nd < dist.get(pred, float("inf")):
                dist[pred] = nd
                next_action[pred] = u
                heapq.heappush(pq, (nd, pred))
    return dist, next_action


def _dijkstra_from_origin(edges, successors, origin, dest):
    dist = {origin: 0.0}
    prev = {}
    pq = [(0.0, origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == dest:
            break
        for v in successors.get(u, []):
            mean_t, _ = edges[(u, v)]
            nd = d + max(1, round(mean_t))
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dest not in dist:
        return [], dist, prev
    path = [dest]
    n = dest
    while n in prev:
        n = prev[n]
        path.append(n)
    path.reverse()
    return path, dist, prev


def run_ge_ddrl_routing(env, budget: int,
                        N: int = 200,
                        delta_w: float = 1.0,
                        alpha_t: float = 0.05,
                        epsilon: float = 0.1,
                        episodes: int = 5000,
                        seed: int = 42,
                        K: int = 5,
                        pretrain_dijkstra_episodes: int = 1000,
                        final_epsilon: float = 0.1,
                        explorer_ratio: float = 0.9) -> dict:
    rng = np.random.default_rng(seed)
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest
    nodes = list(env.nodes)
    non_terminal = [n for n in nodes if n != dest and successors.get(n)]
    if not non_terminal:
        return {"prob": 0.0, "path": [origin], "q_dist": {}, "fallback_count": 0, "reaches_dest": origin == dest, "failed_extraction": origin != dest, "n_params": 0, "z_mean": 0.0, "z_std": 0.0, "z_min": 0.0, "z_max": 0.0, "first_action": None, "path_len": 1}


    Z = {(u, v): np.ones(N, dtype=float) / N for (u, v) in edges}
    atoms = np.arange(N, dtype=float) * delta_w
    max_steps = len(nodes) * 2
    dijkstra_to_go, dijkstra_next = _dijkstra_to_dest(edges, successors, dest)
    dijkstra_path, dijkstra_from_origin, _ = _dijkstra_from_origin(edges, successors, origin, dest)

    if dijkstra_path:
        for i in range(len(dijkstra_path) - 1):
            u, v = dijkstra_path[i], dijkstra_path[i + 1]
            let_to_go = max(1, round(dijkstra_from_origin[dest] - dijkstra_from_origin[u]))
            point = min(let_to_go, max(0, budget - 1), N - 1)
            Z[(u, v)] = _project_point(point, N, delta_w)

    def _sample_tt(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0:
            return float(mean_t)
        cv2 = (sigma / mean_t) ** 2
        mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        return float(np.maximum(rng.lognormal(mu_ln, sigma_ln), 0.01))

    def _sample_rewards(u, v):
        return np.array([_sample_tt(u, v) for _ in range(K)], dtype=float)

    def _epsilon_for(ep, total):
        if total <= 0:
            return final_epsilon
        span = max(1.0, total * explorer_ratio)
        return max(final_epsilon, epsilon - (epsilon - final_epsilon) * min(ep / span, 1.0))

    def _budget_index(remaining_budget):
        return min(max(int(float(remaining_budget) / delta_w), 0), N - 1)

    def _sota_obj(u, v, remaining_budget):
        dist = Z.get((u, v))
        if dist is None:
            return 0.0
        k = _budget_index(remaining_budget)
        return float(np.sum(dist[:k + 1]))

    def _expected_value(u, v):
        dist = Z.get((u, v))
        if dist is None:
            return float("inf")
        return float(np.sum(dist * atoms))

    def _mean_action(node, exclude):
        acts = [a for a in successors.get(node, []) if a not in exclude]
        if not acts:
            return None
        return min(acts, key=lambda a: _expected_value(node, a))

    def _sota_action(node, remaining_budget, candidates=None):
        acts = list(successors.get(node, [])) if candidates is None else list(candidates)
        if not acts:
            return None
        return max(acts, key=lambda a: _sota_obj(node, a, remaining_budget))

    def _apply_update(u, v, target, lr):
        Z[(u, v)] = (1.0 - lr) * Z[(u, v)] + lr * target
        s = Z[(u, v)].sum()
        if s > 0:
            Z[(u, v)] /= s

    def _advance_budget(remaining_budget, u, v):
        mean_t, _ = edges[(u, v)]
        return remaining_budget - max(1.0, float(round(mean_t)))

    def _can_reach_destination(node):
        return node == dest or node in dijkstra_to_go

    def _forms_two_node_cycle(node, action, remaining_budget, seen):
        if action in seen:
            return True
        if action == dest:
            return False
        acts = list(successors.get(action, []))
        if not acts:
            return False
        next_budget = _advance_budget(remaining_budget, node, action)
        next_action = _sota_action(action, next_budget, acts)
        return next_action == node or next_action in seen

    def get_valid_actions(state, remaining_budget, seen):
        candidates = []
        node_to_go = dijkstra_to_go.get(state, float("inf"))
        for action in successors.get(state, []):
            if action in seen:
                continue
            if not _can_reach_destination(action):
                continue
            if dijkstra_to_go.get(action, float("inf")) >= node_to_go:
                continue
            if _forms_two_node_cycle(state, action, remaining_budget, seen):
                continue
            candidates.append(action)
        return candidates

    dijkstra_update_count = 0
    on_policy_update_count = 0
    dijkstra_episode_count = int(pretrain_dijkstra_episodes)

    for ep in range(dijkstra_episode_count):
        state = non_terminal[int(rng.integers(len(non_terminal)))]
        visited = {state}
        lr = 1.0 / np.sqrt(ep + 1)
        eps = _epsilon_for(ep, dijkstra_episode_count)
        for _ in range(max_steps):
            if state == dest:
                break
            acts = get_valid_actions(state, float(budget), visited)
            if not acts:
                break
            if rng.random() < eps:
                action = acts[int(rng.integers(len(acts)))]
            else:
                action = min(acts, key=lambda a: _expected_value(state, a))
            if action is None:
                break
            rewards = _sample_rewards(state, action)
            if action == dest:
                target = _project_point_samples(rewards, N, delta_w)
            else:
                action_star = dijkstra_next.get(action)
                if action_star is None or (action, action_star) not in Z:
                    break
                target = _project_shift_samples(rewards, Z[(action, action_star)], N, delta_w)
            _apply_update(state, action, target, lr)
            dijkstra_update_count += 1
            visited.add(action)
            state = action

    for ep in range(episodes):
        state = non_terminal[int(rng.integers(len(non_terminal)))]
        visited = {state}
        remaining_budget = float(budget)
        eps = _epsilon_for(ep, episodes)
        for _ in range(max_steps):
            if state == dest:
                break
            acts = get_valid_actions(state, remaining_budget, visited)
            if not acts:
                break
            if rng.random() < eps:
                action = acts[int(rng.integers(len(acts)))]
            else:
                action = _sota_action(state, remaining_budget, acts)
            if action is None:
                break
            rewards = _sample_rewards(state, action)
            next_budget = remaining_budget - float(np.mean(rewards))
            if action == dest:
                target = _project_point_samples(rewards, N, delta_w)
            else:
                next_visited = set(visited) | {action}
                next_acts = get_valid_actions(action, next_budget, next_visited)
                if not next_acts:
                    break
                action_star = _sota_action(action, next_budget, next_acts)
                if action_star is None:
                    break
                target = _project_shift_samples(rewards, Z[(action, action_star)], N, delta_w)
            _apply_update(state, action, target, alpha_t)
            on_policy_update_count += 1
            visited.add(action)
            state = action
            remaining_budget = next_budget

    path = [origin]
    node = origin
    remaining_budget = float(budget)
    seen = {origin}
    fallback_count = 0

    for _ in range(len(nodes)):
        if node == dest:
            break
        candidates = get_valid_actions(node, remaining_budget, seen)
        action = _sota_action(node, remaining_budget, candidates)
        if action is None:
            fb = dijkstra_next.get(node)
            if fb is None or fb in seen or (node, fb) not in Z:
                break
            action = fb
            fallback_count += 1

        path.append(action)
        remaining_budget = _advance_budget(remaining_budget, node, action)
        node = action
        if node == dest:
            break
        if node in seen or not successors.get(node):
            break
        seen.add(node)
    origin_acts = get_valid_actions(origin, float(budget), {origin})
    prob = max((_sota_obj(origin, a, budget) for a in origin_acts), default=0.0)
    z_vals = np.concatenate([d for d in Z.values()]) if Z else np.array([])
    uniform = np.ones(N, dtype=float) / N
    updated_edge_distribution_count = sum(1 for d in Z.values() if not np.allclose(d, uniform))
    reaches = bool(path and path[-1] == dest)

    return {
        "prob": prob,
        "path": path,
        "reaches_dest": reaches,
        "failed_extraction": not reaches,
        "fallback_count": fallback_count,
        "q_dist": Z,
        "n_params": len(Z) * N,
        "z_mean": float(np.mean(z_vals)) if z_vals.size else 0.0,
        "z_std": float(np.std(z_vals)) if z_vals.size else 0.0,
        "z_min": float(np.min(z_vals)) if z_vals.size else 0.0,
        "z_max": float(np.max(z_vals)) if z_vals.size else 0.0,
        "first_action": path[1] if len(path) > 1 else None,
        "path_len": len(path),
        "training_protocol": "Chicago-adapted v3: 1000 Dijkstra-guided + 5000 on-policy",
        "initial_epsilon": epsilon,
        "final_epsilon": final_epsilon,
        "explorer_ratio": explorer_ratio,
        "dijkstra_training_episodes": dijkstra_episode_count,
        "on_policy_training_episodes": episodes,
        "dijkstra_update_count": dijkstra_update_count,
        "on_policy_update_count": on_policy_update_count,
        "dijkstra_guidance_enabled": True,
        "final_epsilon_after_dijkstra": _epsilon_for(max(dijkstra_episode_count - 1, 0), dijkstra_episode_count),
        "final_epsilon_after_on_policy": _epsilon_for(max(episodes - 1, 0), episodes),
        "updated_edge_distribution_count": int(updated_edge_distribution_count),
        "atom_sum_valid": bool(all(np.isfinite(d).all() and abs(float(np.sum(d)) - 1.0) < 1e-6 for d in Z.values())),
        "has_nan_inf": bool(not np.isfinite(z_vals).all()) if z_vals.size else False,
        "target_update_period": None,
    }


