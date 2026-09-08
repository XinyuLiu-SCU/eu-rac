"""
dot_routing.py  DOT Algorithm (Prakash 2020) adapted for SiouxEnv.

Reference:
  A. Arun Prakash, "Algorithms for most reliable routes on stochastic and
  time-dependent networks", Transportation Research Part B, 2020.

For a static (non-time-varying) network the algorithm reduces to:
  f(u, b) = max_{v in successors(u)} sum_{c=1}^{b} pmf_{uv}(c) * f(v, b-c)
  f(dest, b) = 1  for all b >= 0
  f(u, 0)    = 0  for u != dest

This is equivalent to Robust Routing with psi=1 (standard SOTA DP).
"""

import numpy as np
from robust_routing import _lognormal_pmf


def run_dot_routing(env, budget: int) -> dict:
    """
    DOT routing DP on SiouxEnv (static network variant).

    Parameters
    ----------
    env    : SiouxEnv instance (.nodes, .edges, .successors, .origin, .dest)
    budget : integer time budget (minutes)

    Returns
    -------
    dict with keys:
      'f_origin' : float   SOTA probability from origin
      'policy'   : callable(node, remaining_budget) -> next_node | None
                   Budget-adaptive policy that queries the DP table with
                   the current remaining budget.
      'path'     : list    greedy path from origin to dest
    """
    nodes = env.nodes
    dest = env.dest
    edges = env.edges          # (u,v) -> (mean, sigma)
    successors = env.successors

    # Pre-compute PMFs for every edge
    pmf_cache = {}
    for (u, v), (mean_t, sigma) in edges.items():
        pmf_cache[(u, v)] = _lognormal_pmf(mean_t, sigma, budget)

    # DP table: f[node][b] = SOTA probability from node with b budget remaining
    f = {node: np.zeros(budget + 1) for node in nodes}
    f[dest][:] = 1.0

    # Fill DP for b = 1 .. budget
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

    # Build budget-adaptive policy: queries DP table with *current* remaining budget
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

    return {
        'f_origin': f[env.origin][budget],
        'policy': policy,   # budget-adaptive callable: (node, remaining_budget) -> next_node
        'path': path,
    }


def _extract_path(f, pmf_cache, origin, dest, successors, edges, budget):
    """Budget-dependent greedy path: at each step pick the successor that
    maximises the DP value given the *current* remaining budget."""
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
            val = sum(pmf[c] * f[v][remaining - c]
                      for c in range(1, min(remaining, len(pmf) - 1) + 1))
            if val > best_val:
                best_val, best_v = val, v
        if best_v is None:
            break
        mean_t, _ = edges.get((node, best_v), (1, 0))
        remaining -= round(mean_t)
        node = best_v
        path.append(node)

    return path
