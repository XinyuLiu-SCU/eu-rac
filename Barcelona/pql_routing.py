"""
pql_routing.py ,Practical Q-Learning (PQL) baseline (Cao et al., 2020).

Reference:
  Cao, Huang, Guo, "Practical Q-Learning for Reliable Shortest Path in
  Stochastic Time-Dependent Networks", IEEE TVT, Vol. 69, No. 9, 2020.

Algorithm:
  Standard Q-learning (gamma=1) on the MDP:
    state  = (node, remaining_budget)
    action = next_node (a successor of current node)
    reward = 1 if dest reached with budget >= 0, else 0
  Travel times are sampled from lognormal distributions.
  Execution uncertainty is NOT modelled ,the agent always moves to the
  chosen successor (unlike EU-RAC which accounts for action deviation).
"""

import numpy as np
from collections import defaultdict


def run_pql_routing(env, budget: int,
                    episodes: int = 50000,
                    alpha: float = 0.1,
                    epsilon: float = 0.1,
                    seed: int = 42,
                    use_warm_start: bool = False,
                    warm_start_budgets=None) -> dict:
    """
    PQL on SiouxEnv.

    Parameters
    ----------
    env      : SiouxEnv instance
    budget   : integer time budget (minutes)
    episodes : training episodes (default 50000)
    alpha    : learning rate (default 0.1)
    epsilon  : -greedy exploration rate (default 0.1)
    seed     : RNG seed (default 42)
    use_warm_start : bool
        If True, initialize positive Q values along Dijkstra LET paths.
        The subsequent Q-learning update is unchanged.
    warm_start_budgets : iterable[int] or None
        Budgets to initialize when warm-start is enabled. Defaults to [budget].

    Returns
    -------
    dict with keys:
      'prob'    : float      ,on-time arrival probability (greedy policy, MC)
      'path'    : list[int]  ,greedy path using mean travel times
      'q_table' : dict       ,Q[(node, budget)][next_node] = value
    """
    rng = np.random.default_rng(seed)
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    # Q[(node, b)][next_node] = float
    Q = defaultdict(lambda: defaultdict(float))
    _dij_next = {}

    # ---- optional warm_start: bias Q along Dijkstra LET paths ----
    if use_warm_start:
        import heapq as _hq
        _dist = {origin: 0.0}; _prev = {}; _pq = [(0.0, origin)]
        while _pq:
            _d, _u = _hq.heappop(_pq)
            if _d > _dist.get(_u, float('inf')): continue
            for _v in successors.get(_u, []):
                _mt, __ = edges[(_u, _v)]
                _nd = _d + max(1, round(_mt))
                if _nd < _dist.get(_v, float('inf')): _dist[_v] = _nd; _prev[_v] = _u; _hq.heappush(_pq, (_nd, _v))
        if dest in _prev:
            _path = []; _n = dest
            while _n in _prev: _path.append(_n); _n = _prev[_n]
            _path.append(origin); _path.reverse()
            _budgets = [budget] if warm_start_budgets is None else warm_start_budgets
            for _init_budget in sorted({int(_b) for _b in _budgets}):
                _rem = _init_budget
                for _i in range(len(_path) - 1):
                    _node, _act = _path[_i], _path[_i+1]
                    _min_b = int(_dist[dest] - _dist[_node])
                    for _bb in range(_min_b, _rem + 4):
                        Q[(_node, _bb)][_act] = 0.5
                    _mt, __ = edges[(_node, _act)]
                    _rem -= max(1, round(_mt))

        # ---- Precompute Dijkstra next-hop for training guidance ----
        _rev = {}
        for (_u, _v), (_mt_edge, _) in edges.items():
            _rev.setdefault(_v, []).append((_u, max(1, round(_mt_edge))))
        _dist_rev = {dest: 0.0}
        _pq_rev = [(0.0, dest)]
        _visited_rev = set()
        while _pq_rev:
            _d, _u = _hq.heappop(_pq_rev)
            if _u in _visited_rev: continue
            _visited_rev.add(_u)
            for _pred, _c in _rev.get(_u, []):
                _nd = _d + _c
                if _nd < _dist_rev.get(_pred, float('inf')):
                    _dist_rev[_pred] = _nd
                    _dij_next[_pred] = _u
                    _hq.heappush(_pq_rev, (_nd, _pred))

    def _greedy_action(node, b):
        acts = successors.get(node, [])
        if not acts:
            return None
        q_vals = Q[(node, b)]
        return max(acts, key=lambda a: q_vals[a])

    def _sample_travel_time(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0:
            return mean_t
        cv2 = (sigma / mean_t) ** 2
        mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        return float(np.maximum(rng.lognormal(mu_ln, sigma_ln), 0.01))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    for _ in range(episodes):
        node = origin
        b = budget
        visited = {node}

        while True:
            if node == dest:
                break

            acts = [v for v in successors.get(node, []) if v not in visited]
            if not acts:
                break

            # epsilon-greedy with Dijkstra guidance for unseen states
            if rng.random() < epsilon:
                action = acts[int(rng.integers(len(acts)))]
            else:
                q_vals = Q[(node, b)]
                best_q = max(q_vals.get(a, 0.0) for a in acts)
                if use_warm_start and best_q == 0.0:
                    fb = _dij_next.get(node)
                    action = fb if (fb is not None and fb in acts) else acts[0]
                else:
                    action = max(acts, key=lambda a: q_vals[a])

            t = _sample_travel_time(node, action)
            b_next = b - round(t)

            visited.add(action)

            if action == dest:
                r = 1.0 if b_next >= 0 else 0.0
                # terminal: no next state value
                td = r - Q[(node, b)][action]
                Q[(node, b)][action] += alpha * td
                break

            if b_next < 0:
                # over budget ,terminal failure
                td = 0.0 - Q[(node, b)][action]
                Q[(node, b)][action] += alpha * td
                break

            # non-terminal update
            next_acts = [v for v in successors.get(action, []) if v not in visited]
            if next_acts:
                max_q_next = max(Q[(action, b_next)][a] for a in next_acts)
            else:
                max_q_next = 0.0

            td = 0.0 + max_q_next - Q[(node, b)][action]
            Q[(node, b)][action] += alpha * td

            node = action
            b = b_next

    # ------------------------------------------------------------------
    # Greedy path (mean travel times, no exploration)
    # ------------------------------------------------------------------
    path = [origin]
    node = origin
    b = budget
    visited = {node}
    max_steps = len(env.nodes) * 2

    for _ in range(max_steps):
        if node == dest or b < 0:
            break
        acts = [v for v in successors.get(node, []) if v not in visited]
        if not acts:
            break
        action = max(acts, key=lambda a: Q[(node, b)][a])
        mean_t, _ = edges[(node, action)]
        b -= round(mean_t)
        visited.add(action)
        path.append(action)
        node = action

    # ------------------------------------------------------------------
    # Monte Carlo evaluation of greedy policy (no execution uncertainty)
    # ------------------------------------------------------------------
    n_eval = 10000
    success = 0
    for _ in range(n_eval):
        node = origin
        b = budget
        visited_mc = {node}
        reached = False
        for _ in range(max_steps):
            if node == dest:
                reached = True
                break
            if b < 0:
                break
            acts = [v for v in successors.get(node, []) if v not in visited_mc]
            if not acts:
                break
            action = max(acts, key=lambda a: Q[(node, b)][a])
            t = _sample_travel_time(node, action)
            b -= round(t)
            visited_mc.add(action)
            node = action
        if reached and b >= 0:
            success += 1

    prob = success / n_eval

    return {
        'prob':    prob,
        'path':    path,
        'q_table': dict(Q),
        'dijkstra_next_hop': dict(_dij_next),
        'use_warm_start': use_warm_start,
        'warm_start_budgets': None if warm_start_budgets is None else [int(b) for b in warm_start_budgets],
    }
