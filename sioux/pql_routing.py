"""
pql_routing.py  Practical Q-Learning (PQL) baseline (Cao et al., 2020).

Reference:
  Cao, Huang, Guo, "Practical Q-Learning for Reliable Shortest Path in
  Stochastic Time-Dependent Networks", IEEE TVT, Vol. 69, No. 9, 2020.

Algorithm:
  Standard Q-learning (gamma=1) on the MDP:
    state  = (node, remaining_budget)
    action = next_node (a successor of current node)
    reward = 1 if dest reached with budget >= 0, else 0
  Travel times are sampled from lognormal distributions.
  Execution uncertainty is NOT modelled  the agent always moves to the
  chosen successor (unlike EU-RAC which accounts for action deviation).
"""

import numpy as np
from collections import defaultdict


def run_pql_routing(env, budget: int,
                    episodes: int = 50000,
                    alpha: float = 0.1,
                    epsilon: float = 0.1,
                    seed: int = 42) -> dict:
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

    Returns
    -------
    dict with keys:
      'prob'    : float       on-time arrival probability (greedy policy, MC)
      'path'    : list[int]   greedy path using mean travel times
      'q_table' : dict        Q[(node, budget)][next_node] = value
    """
    rng = np.random.default_rng(seed)
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    # Q[(node, b)][next_node] = float
    Q = defaultdict(lambda: defaultdict(float))

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

            # -greedy
            if rng.random() < epsilon:
                action = acts[int(rng.integers(len(acts)))]
            else:
                action = max(acts, key=lambda a: Q[(node, b)][a])

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
                # over budget  terminal failure
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
    }
