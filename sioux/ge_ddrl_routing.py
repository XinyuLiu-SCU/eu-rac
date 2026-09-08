"""
ge_ddrl_routing.py  Tabular GE-DDRL (Guo et al., 2023), SOTA mode.

Reference:
  Guo, Sheng, Zhou, Chen, "GE-DDRL: Graph Embedding and Deep Distributional
  Reinforcement Learning for Reliable Shortest Path: A Universal and Scale
  Free Solution", IEEE TITS, Vol. 24, No. 11, 2023.

Algorithm (Tabular DRL, Algorithm 2 of the paper):
  State  = current node  (no remaining-budget tracking)
  Action = next node (successor)
  Reward = sampled travel time on the chosen edge  (positive, unlike the
           reference code which uses negative travel times)
  Z(s,a) = categorical distribution over N atoms representing the CDF of
           total travel time from s to dest when taking action a then
           following the greedy policy.
  Atoms  : z_i = i * delta_w  for i = 0, ..., N-1

  SOTA objective:
    obj(s,a) = P[Z(s,a) <= budget] = sum_{i : z_i <= budget} Z_i(s,a)

  Update (C51-style projection):
    Terminal (next == dest):
      target = point mass at r
    Non-terminal:
      a* = argmax_{a'} obj(next, a')
      target = T(r + Z(next, a*))   [shift distribution right by r, project]
    Z(s,a) <- (1 - alpha) * Z(s,a) + alpha * target

  No execution uncertainty  only travel-time sampling.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_shift(r: float, z_dist: np.ndarray, N: int, delta_w: float) -> np.ndarray:
    """Vectorised C51 projection of (r + Z) onto atoms z_i = i*delta_w."""
    z_max = (N - 1) * delta_w
    shifted = np.clip(r + np.arange(N, dtype=float) * delta_w, 0.0, z_max)
    bj = shifted / delta_w
    m_l = np.clip(np.floor(bj).astype(int), 0, N - 1)
    m_u = np.clip(np.ceil(bj).astype(int), 0, N - 1)

    m_prob = np.zeros(N)
    same = m_l == m_u
    np.add.at(m_prob, m_l[same], z_dist[same])
    diff = ~same
    np.add.at(m_prob, m_l[diff], z_dist[diff] * (m_u[diff] - bj[diff]))
    np.add.at(m_prob, m_u[diff], z_dist[diff] * (bj[diff] - m_l[diff]))
    return m_prob


def _project_point(r: float, N: int, delta_w: float) -> np.ndarray:
    """Project a point mass at r onto atoms."""
    z_max = (N - 1) * delta_w
    Tz = min(z_max, max(0.0, r))
    bj = Tz / delta_w
    m_l = min(int(math.floor(bj)), N - 1)
    m_u = min(int(math.ceil(bj)), N - 1)
    m_prob = np.zeros(N)
    if m_l == m_u:
        m_prob[m_l] = 1.0
    else:
        m_prob[m_l] = m_u - bj
        m_prob[m_u] = bj - m_l
    return m_prob


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ge_ddrl_routing(env, budget: int,
                        N: int = 200,
                        delta_w: float = 1.0,
                        alpha_t: float = 0.05,
                        epsilon: float = 0.1,
                        episodes: int = 50000,
                        seed: int = 42) -> dict:
    """
    Tabular GE-DDRL (SOTA mode) on SiouxEnv.

    Parameters
    ----------
    env      : SiouxEnv instance
    budget   : integer time budget (minutes)
    N        : number of distribution atoms (default 200)
    delta_w  : atom width in minutes (default 1.0)
    alpha_t  : learning rate (default 0.05)
    epsilon  : -greedy exploration rate (default 0.1)
    episodes : training episodes (default 50000)
    seed     : RNG seed (default 42)

    Returns
    -------
    dict with keys:
      'prob'   : float       SOTA probability at origin
      'path'   : list[int]   greedy path (mean travel times for tie-breaking)
      'q_dist' : dict        Z[(u,v)] = np.ndarray of N probabilities
    """
    rng = np.random.default_rng(seed)
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest
    nodes = list(env.nodes)

    # Initialise distributions: uniform over N atoms
    Z = {(u, v): np.ones(N) / N for (u, v) in edges}

    # Precompute SOTA threshold index
    sota_k = min(int(budget / delta_w), N - 1)

    def _sota_obj(u, v):
        dist = Z.get((u, v))
        if dist is None:
            return 0.0
        return float(np.sum(dist[:sota_k + 1]))

    def _best_action(node, exclude):
        acts = [v for v in successors.get(node, []) if v not in exclude]
        if not acts:
            return None
        return max(acts, key=lambda a: _sota_obj(node, a))

    def _sample_tt(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0:
            return float(mean_t)
        cv2 = (sigma / mean_t) ** 2
        mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        return float(np.maximum(rng.lognormal(mu_ln, sigma_ln), 0.01))

    # Non-terminal nodes that have outgoing edges
    non_terminal = [n for n in nodes if n != dest and successors.get(n)]
    max_steps = len(nodes) * 2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    for _ in range(episodes):
        node = non_terminal[int(rng.integers(len(non_terminal)))]
        visited = {node}

        for _ in range(max_steps):
            if node == dest:
                break
            acts = [v for v in successors.get(node, []) if v not in visited]
            if not acts:
                break

            # -greedy
            if rng.random() < epsilon:
                action = acts[int(rng.integers(len(acts)))]
            else:
                action = max(acts, key=lambda a: _sota_obj(node, a))

            r = _sample_tt(node, action)

            # Compute target distribution
            if action == dest:
                target = _project_point(r, N, delta_w)
            else:
                next_acts = [v for v in successors.get(action, []) if v not in visited]
                if not next_acts:
                    target = _project_point(r, N, delta_w)
                else:
                    a_star = max(next_acts, key=lambda a: _sota_obj(action, a))
                    target = _project_shift(r, Z[(action, a_star)], N, delta_w)

            # Exponential moving average update
            Z[(node, action)] = (1.0 - alpha_t) * Z[(node, action)] + alpha_t * target
            # Re-normalise to guard against floating-point drift
            s = Z[(node, action)].sum()
            if s > 0:
                Z[(node, action)] /= s

            visited.add(action)
            node = action

    # ------------------------------------------------------------------
    # Greedy path (no exploration, no revisits)
    # ------------------------------------------------------------------
    path = [origin]
    node = origin
    visited = {node}
    for _ in range(max_steps):
        if node == dest:
            break
        acts = [v for v in successors.get(node, []) if v not in visited]
        if not acts:
            break
        action = max(acts, key=lambda a: _sota_obj(node, a))
        visited.add(action)
        path.append(action)
        node = action

    # SOTA probability at origin
    origin_acts = successors.get(origin, [])
    prob = max((_sota_obj(origin, a) for a in origin_acts), default=0.0)

    return {
        'prob':   prob,
        'path':   path,
        'q_dist': Z,
    }
