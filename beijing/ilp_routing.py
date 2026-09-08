"""
ilp_routing.py  ILP-based exact OTAP routing (Cao et al., 2020).

Reference:
  Cao et al., "An Accurate Solution to the Cardinality-Based Punctuality
  Problem", IEEE ITS Magazine, Vol. 12, No. 4, 2020.

ILP formulation:
  min_{x, theta}  sum_i theta_i
  s.t.
    sum_{(u,v)} w_i(u,v) * x(u,v) - budget <= M * theta_i   for each sample i
    flow conservation at every node
    x(u,v) in {0,1},  theta_i in {0,1}

  prob = 1 - optimal_objective / K
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

def run_ilp_routing(env, budget: int, K: int = 200) -> dict:
    """
    ILP-based OTAP routing on SiouxEnv.

    Parameters
    ----------
    env    : SiouxEnv instance
    budget : integer time budget (minutes)
    K      : number of samples (default 200)

    Returns
    -------
    dict with keys:
      'prob' : float  on-time arrival probability (1 - violations/K)
      'path' : list   optimal path (node sequence)
    """
    if not _HAS_PULP:
        raise ImportError("pulp is required: pip install pulp")

    edges = env.edges
    successors = env.successors
    nodes = env.nodes
    origin = env.origin
    dest = env.dest
    edge_list = list(edges.keys())

    rng = np.random.default_rng(0)
    samples = _draw_samples(edges, K, rng)

    # Big-M: upper bound on any path travel time
    M = float(budget * 10)

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
        in_flow  = pulp.lpSum(x[(u, node)] for u in nodes
                              if (u, node) in x)
        if node == origin:
            prob_model += (out_flow - in_flow == 1), f"flow_{node}"
        elif node == dest:
            prob_model += (out_flow - in_flow == -1), f"flow_{node}"
        else:
            prob_model += (out_flow - in_flow == 0), f"flow_{node}"

    # --- Big-M delay constraints ---
    for i in range(K):
        travel = pulp.lpSum(float(samples[(u, v)][i]) * x[(u, v)]
                            for (u, v) in edge_list)
        prob_model += (travel - budget <= M * theta[i]), f"delay_{i}"

    # --- Solve (suppress output, 60s timeout for large networks) ---
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    status = prob_model.solve(solver)

    # --- Extract results ---
    obj_val = pulp.value(prob_model.objective)
    if obj_val is None:
        print('  [ILP] WARNING: solver returned None (likely timeout or infeasible), '
              f'falling back to obj_val=K={K}')
        obj_val = K  # infeasible fallback
    elif status == 0:  # 0 = Not Solved (typically timeout)
        print(f'  [ILP] WARNING: solver did not converge (status={status}), '
              f'using best solution found (obj={obj_val:.0f}/{K})')

    prob_val = 1.0 - obj_val / K

    x_vals = {(u, v): pulp.value(x[(u, v)]) or 0.0 for (u, v) in edge_list}
    path = _extract_path(x_vals, successors, origin, dest)

    return {'prob': prob_val, 'path': path}
