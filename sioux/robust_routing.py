"""
robust_routing.py  Nadir Robust Routing (SOTA-Robust) baseline.

Reference:
  Manseur, Farhi et al., "Robust routing, its price, and the tradeoff between
  routing robustness and travel time reliability in road networks", EJOR 2020.

Algorithm (discrete-time DP):
  u_d(x) = 1  for all x >= 0          (destination: always on time)
  u_i(0) = 0  for i != d              (budget exhausted: failure)

  For x = 1, 2, ..., budget and i != d:
    A_ij(x) = sum_{h=0}^{x} p_ij(h) * u_j(x - h)   (convolution over PMF)
    u_i(x)  = sum_{p=1}^{m} psi_p * sorted_A_i(x)[p-1]   (weighted sum, descending)
    s_i(x)  = argmax_j A_ij(x)

  With m=2 and weights [psi, 1-psi] the formula becomes:
    u_i(x) = psi * A_i_best(x) + (1-psi) * A_i_second(x)
  When m=1 this reduces to standard SOTA (psi=1).
"""

import numpy as np
from scipy.stats import lognorm


# ---------------------------------------------------------------------------
# PMF construction
# ---------------------------------------------------------------------------

def _lognormal_pmf(mean: float, sigma: float, budget: int) -> np.ndarray:
    """
    Discrete PMF over {1, 2, ..., budget} for a lognormal(mean, sigma) r.v.
    Mass beyond `budget` is folded into the last bin so probabilities sum to 1.
    """
    if sigma <= 0:
        h = max(1, round(mean))
        pmf = np.zeros(budget + 1)
        pmf[min(h, budget)] = 1.0
        return pmf

    cv2 = (sigma / mean) ** 2
    mu_ln = np.log(mean) - 0.5 * np.log(1.0 + cv2)
    sigma_ln = np.sqrt(np.log(1.0 + cv2))
    dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))

    pmf = np.zeros(budget + 1)   # index h = 0..budget
    for h in range(1, budget):
        pmf[h] = dist.cdf(h + 0.5) - dist.cdf(h - 0.5)
    # fold tail into last bin
    pmf[budget] = 1.0 - dist.cdf(budget - 0.5)
    # normalise (handles any floating-point residual)
    total = pmf.sum()
    if total > 0:
        pmf /= total
    return pmf


# ---------------------------------------------------------------------------
# Main DP
# ---------------------------------------------------------------------------

def run_robust_routing(env, budget: int, psi: float, m: int = 2) -> dict:
    """
    Nadir Robust Routing DP on SiouxEnv.

    Parameters
    ----------
    env    : SiouxEnv instance (provides .nodes, .edges, .successors, .dest)
    budget : integer time budget (minutes)
    psi    : robustness weight in [0.5, 1].  psi=1  standard SOTA.
    m      : number of successors considered in the robust criterion (default 2)

    Returns
    -------
    dict with keys:
      'u_origin' : float   robust SOTA probability from origin
      'policy'   : dict    {node: best_successor} at budget x=budget
      'u_table'  : dict    {(node, x): u value}  (full DP table)
    """
    nodes = env.nodes
    dest = env.dest
    edges = env.edges          # (u,v) -> (mean, sigma)
    successors = env.successors

    # Robustness weights: descending, length m, sum to 1
    # psi weights best successor, (1-psi) weights second, etc.
    if m == 1:
        weights = np.array([1.0])
    elif m == 2:
        weights = np.array([psi, 1.0 - psi])
    else:
        # Linear decay, normalised
        w = np.array([psi ** k for k in range(m)], dtype=float)
        weights = w / w.sum()

    # Pre-compute PMFs for every edge
    pmf_cache = {}
    for (u, v), (mean_t, sigma) in edges.items():
        pmf_cache[(u, v)] = _lognormal_pmf(mean_t, sigma, budget)

    # DP table: u[node][x] = robust SOTA probability from node with x budget left
    u = {node: np.zeros(budget + 1) for node in nodes}

    # Boundary: destination is always success for any x >= 0
    u[dest][:] = 1.0

    # Fill DP for x = 1 .. budget
    for x in range(1, budget + 1):
        for i in nodes:
            if i == dest:
                continue
            succs = successors.get(i, [])
            if not succs:
                continue

            # A_ij(x) = sum_{h=1}^{x} pmf_ij(h) * u_j(x-h)
            A_vals = []
            for j in succs:
                pmf = pmf_cache[(i, j)]
                a = 0.0
                for h in range(1, x + 1):
                    a += pmf[h] * u[j][x - h]
                A_vals.append((a, j))

            # Sort descending by A value
            A_vals.sort(key=lambda t: t[0], reverse=True)

            # Robust value: weighted sum of top-m
            top = A_vals[:m]
            u_i_x = sum(weights[k] * top[k][0] for k in range(len(top)))
            u[i][x] = u_i_x

    # Build budget-adaptive policy: queries DP table with *current* remaining budget
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

    # Build flat u_table for inspection
    u_table = {(node, x): u[node][x] for node in nodes for x in range(budget + 1)}

    return {
        'u_origin': u[env.origin][budget],
        'policy': policy,   # budget-adaptive callable: (node, remaining_budget) -> next_node
        'u_table': u_table,
    }


# ---------------------------------------------------------------------------
# Path extraction from policy
# ---------------------------------------------------------------------------

def extract_path(policy, origin: int, dest: int, env, budget: int,
                 u_table: dict = None) -> list:
    """Budget-dependent greedy path extraction.

    If *policy* is a callable (budget-adaptive), queries it with the
    *current* remaining budget at each step.  Otherwise falls back to
    static dict lookup.

    If *u_table* (the full DP table keyed by (node, x)) is supplied the path
    is extracted by re-evaluating the best successor at each step using the
    *current* remaining budget, which avoids cycles caused by following a
    fixed policy computed at the original budget.
    """
    path = [origin]
    node = origin
    remaining = budget
    visited = set()
    edges = env.edges
    successors = env.successors
    pmf_cache = {}  # lazily populated only when u_table is provided

    while node != dest and remaining > 0 and node not in visited:
        visited.add(node)

        if u_table is not None:
            # Dynamic: pick best unvisited successor at current remaining budget
            succs = [v for v in successors.get(node, []) if v not in visited]
            if not succs:
                break
            best_j, best_a = None, -1.0
            for j in succs:
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
            # Budget-adaptive callable: query with current remaining budget
            nxt = policy(node, remaining)
            if nxt is None or nxt in visited:
                break
        else:
            nxt = policy.get(node)
            if nxt is None or nxt in visited:
                break

        mean_t, _ = edges.get((node, nxt), (1, 0))
        remaining -= round(mean_t)
        node = nxt
        path.append(node)

    return path
