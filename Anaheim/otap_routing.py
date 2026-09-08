"""
otap_routing.py  Sample-based OTAP routing (Yang & Zhou, 2017).

Reference:
  Yang & Zhou, "Optimizing on-time arrival probability and percentile travel
  time for elementary path finding in time-dependent transportation networks",
  Transportation Research Part B, 2017.

Simplified PSTOTAP approach:
  1. Draw K independent samples of edge travel times from lognormal distributions.
  2. For each candidate path, estimate on-time prob = #{samples with total time <= T} / K.
  3. Search for the path maximising this estimate.

Search strategy (three-stage):
  Stage 1  mean-Dijkstra: shortest path by mean travel time (fast baseline).
  Stage 2  sample-Dijkstra: for each of the K samples run Dijkstra on that
             deterministic graph, collect unique paths, keep the best.
  Stage 3  k-shortest paths (Yen's algorithm via networkx): enumerate top-k
             paths by mean cost, evaluate each on all K samples, keep the best.
  The final answer is the best path found across all three stages.
"""

import heapq
import numpy as np
from collections import defaultdict

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _draw_samples(edges, K: int, rng) -> dict:
    """
    Returns samples[(u,v)] = np.array of shape (K,) with K travel-time draws.
    """
    samples = {}
    for (u, v), (mean_t, sigma) in edges.items():
        if sigma <= 0:
            samples[(u, v)] = np.full(K, mean_t)
        else:
            cv2 = (sigma / mean_t) ** 2
            mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
            sigma_ln = np.sqrt(np.log(1.0 + cv2))
            draws = rng.lognormal(mu_ln, sigma_ln, size=K)
            samples[(u, v)] = np.maximum(draws, 0.01)
    return samples


# ---------------------------------------------------------------------------
# Path evaluation
# ---------------------------------------------------------------------------

def _eval_path(path: list, samples: dict, budget: int, K: int) -> float:
    """Fraction of K samples where path total time <= budget."""
    if len(path) < 2:
        return 0.0
    totals = np.zeros(K)
    for i in range(len(path) - 1):
        totals += samples[(path[i], path[i + 1])]
    return float(np.sum(totals <= budget) / K)


# ---------------------------------------------------------------------------
# Dijkstra on a single deterministic weight dict
# ---------------------------------------------------------------------------

def _dijkstra_path(successors, weight_dict, origin, dest):
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
            nd = d + weight_dict.get((u, v), float('inf'))
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dest not in prev and dest != origin:
        return None
    path = []
    node = dest
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path[0] == origin else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_otap_routing(env, budget: int, K: int = 30) -> dict:
    """
    Sample-based OTAP routing on SiouxEnv.

    Parameters
    ----------
    env    : SiouxEnv instance
    budget : integer time budget (minutes)
    K      : number of Monte Carlo samples (default 30)

    Returns
    -------
    dict with keys:
      'prob' : float  estimated on-time arrival probability
      'path' : list   best path found (node sequence)
    """
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    rng = np.random.default_rng(0)
    samples = _draw_samples(edges, K, rng)

    best_prob = -1.0
    best_path = []

    def _update(path):
        nonlocal best_prob, best_path
        if path and path[0] == origin and path[-1] == dest:
            p = _eval_path(path, samples, budget, K)
            if p > best_prob:
                best_prob = p
                best_path = path[:]

    # --- Stage 1: mean-Dijkstra ---
    mean_w = {e: m for e, (m, _) in edges.items()}
    path_mean = _dijkstra_path(successors, mean_w, origin, dest)
    if path_mean:
        _update(path_mean)

    # --- Stage 2: per-sample Dijkstra (collect unique paths) ---
    seen = set()
    for k in range(K):
        w_k = {e: float(samples[e][k]) for e in edges}
        p = _dijkstra_path(successors, w_k, origin, dest)
        if p:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                _update(p)

    # --- Stage 3: k-shortest paths via networkx (if available) ---
    if _HAS_NX:
        G = nx.DiGraph()
        for (u, v), (m, _) in edges.items():
            G.add_edge(u, v, weight=m)
        try:
            gen = nx.shortest_simple_paths(G, origin, dest, weight='weight')
            for i, p in enumerate(gen):
                if i >= 50:
                    break
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    _update(p)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return {'prob': best_prob, 'path': best_path}
