"""
pulse_routing.py  Pulse Algorithm for MPOAP (Leiva et al., 2026).

Reference:
  Leiva, Morales, Yamin, Medaglia, "An Exact Method for Reliable Shortest
  Path Problems With Correlation", Networks, 2026.

Solves: max_{P} P[t(P) <= T]
Assumption: edge travel times are independent lognormal r.v.s.
  Path travel time ~ sum of lognormals, approximated as normal via
  moment-matching: mean = sum(means), var = sum(vars).

Pruning rules implemented:
  1. Bounds    : optimistic upper bound on on-time prob using min-mean and
                 min-var lower bounds to destination. Prune if ub < alpha_lb.
  2. Infeasibility: prune if partial mean already exceeds budget T
                    (normal CDF would be <= 0.5, can't beat any found solution).
"""

import heapq
import time
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _on_time_prob(mean: float, var: float, budget: int) -> float:
    """P[N(mean, sqrt(var)) <= budget]."""
    if var <= 0:
        return 1.0 if mean <= budget else 0.0
    return float(norm.cdf(budget, loc=mean, scale=np.sqrt(var)))


def _dijkstra_min(successors, edges, dest, weight: str):
    """
    Dijkstra from all nodes to `dest` on the reversed graph.
    weight: 'mean' or 'var'
    Returns dist[node] = minimum cumulative weight from node to dest.
    """
    # Build reverse adjacency
    rev = {}
    for (u, v), (mean_t, sigma) in edges.items():
        w = mean_t if weight == 'mean' else sigma ** 2
        rev.setdefault(v, []).append((u, w))

    dist = {dest: 0.0}
    pq = [(0.0, dest)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in rev.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pulse_routing(env, budget: int) -> dict:
    """
    Pulse algorithm for MPOAP on SiouxEnv.

    Parameters
    ----------
    env    : SiouxEnv instance (.nodes, .edges, .successors, .origin, .dest)
    budget : integer time budget (minutes)

    Returns
    -------
    dict with keys:
      'prob'      : float   best on-time arrival probability found
      'path'      : list    optimal path (node sequence)
      'path_mean' : float   sum of mean travel times along path
      'path_std'  : float   std dev of total travel time along path
    """
    edges = env.edges          # (u,v) -> (mean_hours, sigma_hours)
    successors = env.successors
    origin = env.origin
    dest = env.dest

    # ---- Keep edge costs in native hours (matching budget unit) ----
    # Budget is in rounded-hour units (from _dijkstra_rounded with
    # max(1, round(mean_t))). Pruning rules compare hours against hours.
    # Converting to minutes breaks this by making the budget appear
    # 60 too small, causing all paths to be pruned as infeasible.
    edges_min = {}
    for (u, v), (mean_t, sigma) in edges.items():
        edges_min[(u, v)] = (mean_t, sigma)   # keep native hours

    # Pre-compute lower bounds on remaining mean and variance to dest
    lb_mean = _dijkstra_min(successors, edges_min, dest, 'mean')
    lb_var  = _dijkstra_min(successors, edges_min, dest, 'var')

    # Timeout guard for large networks (933 nodes  2950 edges  huge DFS)
    PULSE_TIMEOUT = 300  # seconds (increased from 120 for Chicago-scale network)
    _pulse_deadline = time.time() + PULSE_TIMEOUT
    _pulse_timed_out = False

    # Best solution found so far
    best = {'prob': -1.0, 'path': [], 'mean': 0.0, 'var': 0.0}

    # ---- Bootstrap with heuristic initial solution to enable pruning ----
    # Without an initial bound, nothing gets pruned on large networks.
    # Compute a Dijkstra minimum-mean path as the initial best solution.
    _h_dist = {origin: 0.0}; _h_prev = {}; _h_pq = [(0.0, origin)]
    while _h_pq:
        _hd, _hu = heapq.heappop(_h_pq)
        if _hd > _h_dist.get(_hu, float('inf')): continue
        if _hu == dest: break
        for _hv in successors.get(_hu, []):
            _hmt, _hs = edges_min.get((_hu, _hv), (1, 0))
            _hnd = _hd + _hmt
            if _hnd < _h_dist.get(_hv, float('inf')): _h_dist[_hv] = _hnd; _h_prev[_hv] = _hu; heapq.heappush(_h_pq, (_hnd, _hv))
    if dest in _h_prev:
        _h_path = [dest]; _hn = dest
        while _hn in _h_prev: _h_path.append(_h_prev[_hn]); _hn = _h_prev[_hn]
        _h_path.reverse()
        _h_mean = 0.0; _h_var = 0.0
        for _hi in range(len(_h_path) - 1):
            _hmt, _hs = edges_min[(_h_path[_hi], _h_path[_hi + 1])]
            _h_mean += _hmt; _h_var += _hs ** 2
        _h_prob = _on_time_prob(_h_mean, _h_var, budget)
        best['prob'] = _h_prob; best['path'] = _h_path; best['mean'] = _h_mean; best['var'] = _h_var
        print(f'  [Pulse] heuristic init: prob={_h_prob:.4f}, path_len={len(_h_path)}, mean={_h_mean:.1f}m')

    def pulse(node, cum_mean, cum_var, path, visited):
        nonlocal _pulse_timed_out

        # --- Timeout check ---
        if time.time() > _pulse_deadline:
            _pulse_timed_out = True
            return

        # --- Reached destination (check before pruning so tight-budget
        #     paths with mean > budget can still be considered) ---
        if node == dest:
            prob = _on_time_prob(cum_mean, cum_var, budget)
            if prob > best['prob']:
                best['prob'] = prob
                best['path'] = path[:]
                best['mean'] = cum_mean
                best['var']  = cum_var
            return

        # --- Infeasibility prune (now in consistent minute units) ---
        if cum_mean > budget:
            return

        # --- Bounds prune ---
        rem_mean = lb_mean.get(node, float('inf'))
        rem_var  = lb_var.get(node, float('inf'))
        ub = _on_time_prob(cum_mean + rem_mean, cum_var + rem_var, budget)
        if ub <= best['prob']:
            return

        # --- Recurse over successors ---
        for nxt in successors.get(node, []):
            if nxt in visited:
                continue
            if time.time() > _pulse_deadline:
                _pulse_timed_out = True
                return
            mean_t, sigma = edges_min[(node, nxt)]
            visited.add(nxt)
            path.append(nxt)
            pulse(nxt, cum_mean + mean_t, cum_var + sigma ** 2, path, visited)
            path.pop()
            visited.discard(nxt)

    visited = {origin}
    pulse(origin, 0.0, 0.0, [origin], visited)

    if _pulse_timed_out:
        print(f'  [Pulse] WARNING: timed out after {PULSE_TIMEOUT}s, '
              f'best prob so far = {best["prob"]:.4f}')

    if not best['path']:
        # Pulse found nothing  use the heuristic Dijkstra path as fallback.
        # (This shouldn't happen with the heuristic init above, but keep as safety.)
        node = origin
        fb_path = [origin]
        fb_visited = {origin}
        fb_mean, fb_var = 0.0, 0.0
        # Use full Dijkstra reverse-path, not greedy (greedy can dead-end)
        if dest in _h_prev:
            _h_path = [dest]; _hn = dest
            while _hn in _h_prev: _h_path.append(_h_prev[_hn]); _hn = _h_prev[_hn]
            fb_path = _h_path[::-1]
            fb_mean = _h_dist.get(dest, 0.0)
            fb_var = 0.0
            for _hi in range(len(fb_path) - 1):
                _hmt, _hs = edges_min[(fb_path[_hi], fb_path[_hi + 1])]
                fb_var += _hs ** 2
        else:
            # Last resort: greedy (backward compatible)
            while node != dest:
                succs = [v for v in successors.get(node, []) if v not in fb_visited]
                if not succs:
                    break
                nxt = min(succs, key=lambda v: edges_min[(node, v)][0])
                m, s = edges_min[(node, nxt)]
                fb_mean += m
                fb_var += s ** 2
                fb_visited.add(nxt)
                fb_path.append(nxt)
                node = nxt
        best['prob'] = _on_time_prob(fb_mean, fb_var, budget)
        best['path'] = fb_path
        best['mean'] = fb_mean
        best['var']  = fb_var

    return {
        'prob':      best['prob'],
        'path':      best['path'],
        'path_mean': best['mean'],     # native hours (matching budget unit)
        'path_std':  np.sqrt(best['var']) if best['var'] > 0 else 0.0,
    }
