"""
ilp_routing.py  ILP-based exact OTAP routing (Cao et al., 2020).

Reference:
  Cao et al., "An Accurate Solution to the Cardinality-Based Punctuality
  Problem", IEEE ITS Magazine, Vol. 12, No. 4, 2020.

ILP formulation:
  min_{x, theta}  sum_i theta_i
  s.t.
    sum_{(u,v)} w_i(u,v) * x(u,v) - budget <= M_i * theta_i   for each sample i
    flow conservation at every node
    x(u,v) in {0,1},  theta_i in {0,1}

  prob = 1 - optimal_objective / K

Improvements over v1 (2026-07-28):
  - Per-sample tight Big-M (M_i = max(1, sum_all_edges_sample_i - budget))
    instead of crude budget*10, giving much tighter LP relaxation.
  - Configurable random seed (no longer hard-coded to 0).
  - Dijkstra fallback path when solver times out or returns infeasible.
  - Increased default samples (K=300) and time limit (120 s).
"""

import heapq
import numpy as np

try:
    import pulp
    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False


# ---------------------------------------------------------------------------
# Sampling (shared logic with otap_routing)
# ---------------------------------------------------------------------------

def _draw_samples(edges, K: int, rng) -> dict:
    """Returns samples[(u,v)] = np.ndarray shape (K,)."""
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
# Dijkstra fallback (used when ILP times out)
# ---------------------------------------------------------------------------

def _dijkstra_fallback_path(edges, successors, origin, dest):
    """Return the mean-Dijkstra shortest path from origin to dest."""
    dist = {origin: 0.0}
    prev = {}
    pq = [(0.0, origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dest:
            break
        if d > dist.get(u, float('inf')):
            continue
        for v in successors.get(u, []):
            mean_t, _ = edges.get((u, v), (1.0, 0.0))
            nd = d + mean_t
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dest not in prev and origin != dest:
        return [origin]
    path = [dest]
    node = dest
    while node in prev:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path


def _eval_path_on_samples(path, samples, budget):
    """Return on-time prob (fraction of K samples where path total <= budget)."""
    if len(path) < 2:
        return 0.0
    K = len(next(iter(samples.values())))
    totals = np.zeros(K)
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if edge in samples:
            totals += samples[edge]
        else:
            return 0.0
    return float(np.sum(totals <= budget) / K)


# ---------------------------------------------------------------------------
# Path extraction from x solution
# ---------------------------------------------------------------------------

def _extract_path(x_vals, successors, origin, dest):
    """Follow selected edges (x=1) from origin to dest."""
    selected = {(u, v) for (u, v), val in x_vals.items() if val > 0.5}
    # build adjacency from selected edges
    nxt = {}
    for u, v in selected:
        nxt[u] = v
    path = [origin]
    node = origin
    visited = {origin}
    while node != dest:
        node = nxt.get(node)
        if node is None or node in visited:
            break
        visited.add(node)
        path.append(node)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ilp_routing(env, budget: int, K: int = 300, seed: int = None,
                    time_limit: int = 120, big_m_cap: float = 1e6) -> dict:
    """
    ILP-based OTAP routing on ChicagoEnv.

    Parameters
    ----------
    env        : ChicagoEnv instance
    budget     : integer time budget (hours, rounded)
    K          : number of Monte Carlo samples (default 300)
    seed       : random seed for sampling (default None  uses 42)
    time_limit : solver time limit in seconds (default 120)
    big_m_cap  : upper bound on per-sample Big-M (default 1e6)

    Returns
    -------
    dict with keys:
      'prob'   : float  on-time arrival probability (1 - violations/K)
      'path'   : list   optimal path (node sequence)
      'status' : str    solver status
      'method' : str    'ILP' or 'ILP-fallback-Dijkstra'
    """
    if not _HAS_PULP:
        raise ImportError("pulp is required: pip install pulp")

    edges = env.edges
    successors = env.successors
    nodes = env.nodes
    origin = env.origin
    dest = env.dest
    edge_list = list(edges.keys())

    # Use provided seed; each OD  B   combo should get a unique seed
    if seed is None:
        seed = 42
    rng = np.random.default_rng(seed)
    samples = _draw_samples(edges, K, rng)

    # ---- Per-sample tight Big-M ----
    # M_i = max(1.0, sum_all_edge_samples_i - budget), capped at big_m_cap.
    # This is 10-1000 tighter than the old budget*10, giving a stronger LP
    # relaxation and faster convergence.
    sample_totals = np.zeros(K)
    for (u, v), samp in samples.items():
        sample_totals += samp  # all edges are independent  sum is valid UB
    M_vals = np.maximum(1.0, sample_totals - float(budget))
    M_vals = np.minimum(M_vals, big_m_cap)

    prob_model = pulp.LpProblem("OTAP_ILP", pulp.LpMinimize)

    # --- Decision variables ---
    x = {(u, v): pulp.LpVariable(f"x_{u}_{v}", cat='Binary')
         for (u, v) in edge_list}
    theta = [pulp.LpVariable(f"theta_{i}", cat='Binary') for i in range(K)]

    # --- Objective ---
    prob_model += pulp.lpSum(theta)

    # --- Flow conservation ---
    for node in nodes:
        out_flow = pulp.lpSum(x[(node, v)] for v in successors.get(node, [])
                              if (node, v) in x)
        in_flow = pulp.lpSum(x[(u, node)] for u in nodes
                             if (u, node) in x)
        if node == origin:
            prob_model += (out_flow - in_flow == 1), f"flow_{node}"
        elif node == dest:
            prob_model += (out_flow - in_flow == -1), f"flow_{node}"
        else:
            prob_model += (out_flow - in_flow == 0), f"flow_{node}"

    # --- Per-sample Big-M delay constraints ---
    for i in range(K):
        travel = pulp.lpSum(float(samples[(u, v)][i]) * x[(u, v)]
                            for (u, v) in edge_list)
        prob_model += (travel - budget <= M_vals[i] * theta[i]), f"delay_{i}"

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    status = prob_model.solve(solver)
    pulp_status = pulp.LpStatus[prob_model.status]

    # --- Extract results ---
    obj_val = pulp.value(prob_model.objective)

    if obj_val is None or pulp_status not in ("Optimal", "Feasible"):
        # Solver failed or timed out without a feasible solution  fallback
        print(f'  [ILP] WARNING: solver returned {pulp_status} (obj={obj_val}), '
              f'falling back to Dijkstra')
        fb_path = _dijkstra_fallback_path(edges, successors, origin, dest)
        fb_prob = _eval_path_on_samples(fb_path, samples, budget)
        return {
            'prob': fb_prob,
            'path': fb_path,
            'status': pulp_status,
            'method': 'ILP-fallback-Dijkstra',
        }

    if pulp_status != "Optimal":
        print(f'  [ILP] WARNING: solver status={pulp_status}, '
              f'using best solution found (obj={obj_val:.0f}/{K})')

    prob_val = 1.0 - obj_val / K

    x_vals = {(u, v): pulp.value(x[(u, v)]) or 0.0 for (u, v) in edge_list}
    path = _extract_path(x_vals, successors, origin, dest)

    # If extracted path is invalid (doesn't reach dest), fall back to Dijkstra
    if len(path) < 2 or path[-1] != dest:
        print(f'  [ILP] WARNING: extracted path invalid (len={len(path)}, '
              f'ends at {path[-1] if path else "N/A"}), falling back to Dijkstra')
        fb_path = _dijkstra_fallback_path(edges, successors, origin, dest)
        fb_prob = _eval_path_on_samples(fb_path, samples, budget)
        return {
            'prob': fb_prob,
            'path': fb_path,
            'status': pulp_status,
            'method': 'ILP-fallback-Dijkstra',
        }

    # ---- Dijkstra safety check ----
    # The ILP optimizes travel-time punctuality on K samples, but the real
    # environment has execution uncertainty (wrong turns at uncertain nodes).
    # If the ILP path goes through high-uncertainty areas, its real MC
    # performance can be much worse than Dijkstra.
    #
    # Safety rule: compare ILP path vs Dijkstra on the K samples.
    #   - If Dijkstra is better on samples  use Dijkstra (ILP didn't help)
    #   - If ILP is better but by < 3% and has more uncertain edges  use Dijkstra
    #   - Otherwise  use ILP
    fb_path = _dijkstra_fallback_path(edges, successors, origin, dest)
    if fb_path and fb_path[-1] == dest:
        ilp_sample_prob = _eval_path_on_samples(path, samples, budget)
        djk_sample_prob = _eval_path_on_samples(fb_path, samples, budget)

        # Count uncertain edges on each path
        ilp_uncertain = sum(1 for i in range(len(path) - 1)
                           if path[i] in getattr(env, 'uncertain_edges', {}))
        djk_uncertain = sum(1 for i in range(len(fb_path) - 1)
                           if fb_path[i] in getattr(env, 'uncertain_edges', {}))

        gain = ilp_sample_prob - djk_sample_prob

        if gain <= 0:
            # ILP is not better than Dijkstra even on samples  use Dijkstra
            print(f'  [ILP] Dijkstra safety: ILP path NOT better on samples '
                  f'(ILP={ilp_sample_prob:.4f} <= Dijk={djk_sample_prob:.4f}), '
                  f'using Dijkstra')
            return {
                'prob': djk_sample_prob,
                'path': fb_path,
                'status': pulp_status,
                'method': 'ILP-safety-Dijkstra',
            }
        elif gain < 0.03 and ilp_uncertain > djk_uncertain:
            # Marginal gain + more risk  conservative choice
            print(f'  [ILP] Dijkstra safety: marginal gain ({gain:.3f}) with '
                  f'more uncertain edges (ILP={ilp_uncertain} > Dijk={djk_uncertain}), '
                  f'using Dijkstra')
            return {
                'prob': djk_sample_prob,
                'path': fb_path,
                'status': pulp_status,
                'method': 'ILP-safety-Dijkstra',
            }
        else:
            print(f'  [ILP] Dijkstra safety: ILP path improves by {gain:.3f} '
                  f'(ILP={ilp_sample_prob:.4f} vs Dijk={djk_sample_prob:.4f}), '
                  f'ILP_unc={ilp_uncertain} Dijk_unc={djk_uncertain}, using ILP')

    return {
        'prob': prob_val,
        'path': path,
        'status': pulp_status,
        'method': 'ILP',
    }
