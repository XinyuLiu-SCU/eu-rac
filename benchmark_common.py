"""Shared benchmark routing implementations for non-Sioux networks."""

from __future__ import annotations

import heapq

import numpy as np
from scipy.stats import lognorm, norm


def _lognormal_pmf(mean: float, sigma: float, budget: int) -> np.ndarray:
    """Discrete PMF over {1, ..., budget} for a lognormal travel-time model."""
    if sigma <= 0:
        h = max(1, round(mean))
        pmf = np.zeros(budget + 1)
        pmf[min(h, budget)] = 1.0
        return pmf

    cv2 = (sigma / mean) ** 2
    mu_ln = np.log(mean) - 0.5 * np.log(1.0 + cv2)
    sigma_ln = np.sqrt(np.log(1.0 + cv2))
    dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))

    pmf = np.zeros(budget + 1)
    for h in range(1, budget):
        lower = 0.0 if h == 1 else h - 0.5
        pmf[h] = dist.cdf(h + 0.5) - dist.cdf(lower)
    pmf[budget] = 1.0 - dist.cdf(budget - 0.5)

    total = pmf.sum()
    if total > 0:
        pmf /= total
    return pmf


def run_dot_routing(env, budget: int) -> dict:
    """DOT routing DP for networks with independent lognormal edges."""
    nodes = env.nodes
    dest = env.dest
    edges = env.edges
    successors = env.successors

    pmf_cache = {}
    for (u, v), (mean_t, sigma) in edges.items():
        pmf_cache[(u, v)] = _lognormal_pmf(mean_t, sigma, budget)

    f = {node: np.zeros(budget + 1) for node in nodes}
    f[dest][:] = 1.0

    for b in range(1, budget + 1):
        for u in nodes:
            if u == dest:
                continue
            succs = successors.get(u, [])
            if not succs:
                continue
            best_val = 0.0
            for v in succs:
                pmf = pmf_cache[(u, v)]
                val = sum(pmf[c] * f[v][b - c] for c in range(1, b + 1))
                if val > best_val:
                    best_val = val
            f[u][b] = best_val

    def policy(node, remaining_budget):
        if node == dest:
            return None
        b = max(0, min(remaining_budget, budget))
        succs = successors.get(node, [])
        if not succs:
            return None
        best_v, best_val = None, -1.0
        for v in succs:
            pmf = pmf_cache[(node, v)]
            val = sum(pmf[c] * f[v][b - c] for c in range(1, b + 1))
            if val > best_val:
                best_val, best_v = val, v
        return best_v

    path = _extract_path(f, pmf_cache, env.origin, dest, successors, edges, budget)
    return {'f_origin': f[env.origin][budget], 'policy': policy, 'path': path}


def _extract_path(f, pmf_cache, origin, dest, successors, edges, budget):
    path = [origin]
    node = origin
    remaining = budget
    visited = set()

    while node != dest and remaining > 0 and node not in visited:
        visited.add(node)
        succs = [v for v in successors.get(node, []) if v not in visited]
        if not succs:
            break
        best_v, best_val = None, -1.0
        for v in succs:
            pmf = pmf_cache[(node, v)]
            val = sum(pmf[c] * f[v][remaining - c] for c in range(1, min(remaining, len(pmf) - 1) + 1))
            if val > best_val:
                best_val, best_v = val, v
        if best_v is None:
            break
        mean_t, _ = edges.get((node, best_v), (1, 0))
        remaining -= round(mean_t)
        node = best_v
        path.append(node)
    return path


def run_robust_routing(env, budget: int, psi: float, m: int = 2) -> dict:
    """Robust routing DP with weighted top-m successor aggregation."""
    nodes = env.nodes
    dest = env.dest
    edges = env.edges
    successors = env.successors

    if m == 1:
        weights = np.array([1.0])
    elif m == 2:
        weights = np.array([psi, 1.0 - psi])
    else:
        w = np.array([psi ** k for k in range(m)], dtype=float)
        weights = w / w.sum()

    pmf_cache = {}
    for (u, v), (mean_t, sigma) in edges.items():
        pmf_cache[(u, v)] = _lognormal_pmf(mean_t, sigma, budget)

    u = {node: np.zeros(budget + 1) for node in nodes}
    u[dest][:] = 1.0

    for x in range(1, budget + 1):
        for i in nodes:
            if i == dest:
                continue
            succs = successors.get(i, [])
            if not succs:
                continue
            A_vals = []
            for j in succs:
                pmf = pmf_cache[(i, j)]
                a = 0.0
                for h in range(1, x + 1):
                    a += pmf[h] * u[j][x - h]
                A_vals.append((a, j))
            A_vals.sort(key=lambda t: t[0], reverse=True)
            top = A_vals[:m]
            u[i][x] = sum(weights[k] * top[k][0] for k in range(len(top)))

    def policy(node, remaining_budget):
        if node == dest:
            return None
        b = max(0, min(remaining_budget, budget))
        succs = successors.get(node, [])
        if not succs:
            return None
        best_j, best_a = None, -1.0
        for j in succs:
            pmf = pmf_cache[(node, j)]
            a = sum(pmf[h] * u[j][b - h] for h in range(1, b + 1))
            if a > best_a:
                best_a, best_j = a, j
        return best_j

    u_table = {(node, x): u[node][x] for node in nodes for x in range(budget + 1)}
    return {'u_origin': u[env.origin][budget], 'policy': policy, 'u_table': u_table}


def extract_path(policy: dict, origin: int, dest: int, env, budget: int, u_table: dict = None) -> list:
    """Budget-dependent greedy path extraction for the robust baseline."""
    path = [origin]
    node = origin
    remaining = budget
    visited = set()
    edges = env.edges
    successors = env.successors
    pmf_cache = {}
    max_steps = len(successors) * 2

    while node != dest and remaining > 0 and len(path) < max_steps:
        visited.add(node)

        if u_table is not None:
            all_succs = successors.get(node, [])
            unvisited = [v for v in all_succs if v not in visited]
            cands = unvisited if unvisited else all_succs
            best_j, best_a = None, -1.0
            for j in cands:
                if (node, j) not in pmf_cache:
                    mean_t, sigma = edges[(node, j)]
                    pmf_cache[(node, j)] = _lognormal_pmf(mean_t, sigma, budget)
                pmf = pmf_cache[(node, j)]
                a = sum(pmf[h] * u_table.get((j, remaining - h), 0.0)
                        for h in range(1, min(remaining, len(pmf) - 1) + 1))
                if a > best_a:
                    best_a, best_j = a, j
            if best_j is None:
                break
            nxt = best_j
        elif callable(policy):
            nxt = policy(node, remaining)
            if nxt is None or nxt in visited:
                break
        else:
            nxt = policy.get(node)
            if nxt is None or nxt in visited:
                break

        mean_t, _ = edges.get((node, nxt), (1, 0))
        if (node, nxt) in pmf_cache:
            pmf = pmf_cache[(node, nxt)]
            expected_cost = sum(c * pmf[c] for c in range(1, min(remaining, len(pmf) - 1) + 1))
            step_cost = max(1, round(expected_cost)) if expected_cost > 0 else max(1, round(mean_t))
        else:
            step_cost = max(1, round(mean_t))
        remaining -= step_cost
        node = nxt
        path.append(node)

    return path


def _path_stats(path, edges):
    mu = sum(edges[(path[i], path[i + 1])][0] for i in range(len(path) - 1))
    var = sum(edges[(path[i], path[i + 1])][1] ** 2 for i in range(len(path) - 1))
    return mu, np.sqrt(var)


def _on_time_prob(mu, std, budget):
    if std <= 0:
        return 1.0 if mu <= budget else 0.0
    return float(norm.cdf(budget, loc=mu, scale=std))


def _dijkstra_mean_std(successors, edges, origin, dest, zeta: float):
    dist = {origin: 0.0}
    prev = {origin: None}
    pq = [(0.0, origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dest:
            break
        if d > dist.get(u, float('inf')):
            continue
        for v in successors.get(u, []):
            mean_t, sigma = edges[(u, v)]
            w = mean_t + zeta * sigma
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dest not in prev and dest != origin:
        return None
    path, node = [], dest
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path[0] == origin else None


def _dijkstra_weighted(successors, weights, origin, dest):
    dist = {origin: 0.0}
    prev = {origin: None}
    pq = [(0.0, origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dest:
            break
        if d > dist.get(u, float('inf')):
            continue
        for v in successors.get(u, []):
            w = weights.get((u, v), float('inf'))
            if w <= 0:
                continue
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dest not in prev and dest != origin:
        return None
    path, node = [], dest
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path[0] == origin else None


def run_gp3_routing(env, budget: int, zeta_min: float = 0.0, zeta_max: float = 50.0,
                    n_steps: int = 200, n_random: int = 200, seed: int | None = None) -> dict:
    """GP3 routing with deterministic zeta sweep and randomized perturbations."""
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    if seed is None:
        seed = 42
    rng = np.random.default_rng(seed)

    best_prob = -1.0
    best_path = []
    best_zeta = zeta_min
    best_method = 'GP3-zeta'

    seen = set()
    zetas = np.linspace(zeta_min, zeta_max, n_steps)

    for zeta in zetas:
        path = _dijkstra_mean_std(successors, edges, origin, dest, zeta)
        if path is None:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)

        mu, std = _path_stats(path, edges)
        prob = _on_time_prob(mu, std, budget)
        if prob > best_prob:
            best_prob = prob
            best_path = path
            best_zeta = float(zeta)
            best_method = 'GP3-zeta'

    edge_list = list(edges.keys())
    base_weights = {e: edges[e][0] for e in edge_list}
    n_edges = len(edge_list)

    for _ in range(n_random):
        noise_factors = rng.lognormal(mean=0.0, sigma=0.03, size=n_edges)
        noisy_weights = {}
        for i, e in enumerate(edge_list):
            noisy_weights[e] = max(0.001, base_weights[e] * noise_factors[i])

        path = _dijkstra_weighted(successors, noisy_weights, origin, dest)
        if path is None:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)

        mu, std = _path_stats(path, edges)
        prob = _on_time_prob(mu, std, budget)
        if prob > best_prob:
            best_prob = prob
            best_path = path
            best_method = 'GP3-random'

    return {'prob': best_prob, 'path': best_path, 'best_zeta': best_zeta, 'method': best_method}
