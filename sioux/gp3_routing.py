"""
gp3_routing.py  GP3 (Gaussian Process Path Planning, Guo et al. 2022).

Reference:
  Guo, Hou, Cao, Zhang, "GP3: Gaussian Process Path Planning for Reliable
  Shortest Path in Transportation Networks", IEEE TITS, Vol. 23, No. 8, 2022.

Problem:
  minimize  mu^T x + zeta * sqrt(x^T Sigma x)
  s.t.      Ax = b,  x in {0,1}^m

With diagonal Sigma (independent edges), x^T Sigma x = sum_e sigma_e^2 * x_e,
so the objective decomposes per-edge:
  cost(e) = mu_e + zeta * sigma_e

This means the mean-std sub-problem reduces to a weighted Dijkstra with
edge weight  w_zeta(e) = mu_e + zeta * sigma_e.

SOTA equivalence (Theorem 4):
  For a given budget T, the optimal zeta satisfies
    mu^T x*(zeta) + zeta * sigma(x*(zeta)) = T
  We sweep zeta over [zeta_min, zeta_max] and keep the path with the
  highest on-time probability Phi((T - mu_path) / sigma_path).
"""

import heapq
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Dijkstra with parametric edge weight  w = mu + zeta * sigma
# ---------------------------------------------------------------------------

def _dijkstra_mean_std(successors, edges, origin, dest, zeta: float):
    """Shortest path under weight mu + zeta * sigma."""
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


# ---------------------------------------------------------------------------
# Path statistics
# ---------------------------------------------------------------------------

def _path_stats(path, edges):
    """Return (mean, std) of total travel time assuming independent edges."""
    mu = sum(edges[(path[i], path[i + 1])][0] for i in range(len(path) - 1))
    var = sum(edges[(path[i], path[i + 1])][1] ** 2 for i in range(len(path) - 1))
    return mu, np.sqrt(var)


def _on_time_prob(mu, std, budget):
    if std <= 0:
        return 1.0 if mu <= budget else 0.0
    return float(norm.cdf(budget, loc=mu, scale=std))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gp3_routing(env, budget: int,
                    zeta_min: float = 0.0,
                    zeta_max: float = 50.0,
                    n_steps: int = 200) -> dict:
    """
    GP3 routing on SiouxEnv.

    Sweeps zeta in [zeta_min, zeta_max] with n_steps values, solves the
    mean-std Dijkstra for each, and returns the path with the highest
    on-time probability under the normal approximation.

    Parameters
    ----------
    env       : SiouxEnv instance
    budget    : integer time budget (minutes)
    zeta_min  : lower bound of zeta search range (default 0.0)
    zeta_max  : upper bound of zeta search range (default 50.0)
    n_steps   : number of zeta values to evaluate (default 200)

    Returns
    -------
    dict with keys:
      'prob'      : float       best on-time probability (normal approx)
      'path'      : list[int]   optimal path node sequence
      'best_zeta' : float       zeta value that achieved best_prob
    """
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    best_prob = -1.0
    best_path = []
    best_zeta = zeta_min

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

    return {
        'prob':      best_prob,
        'path':      best_path,
        'best_zeta': best_zeta,
    }
