"""
segac_routing.py  Tabular SEGAC (Guo et al., 2024), On-Policy GAC mode.

Reference:
  Guo, He, Sheng, Cao, Zhou, Gao, "SEGAC: Sample Efficient Generalized
  Actor Critic for the Stochastic On-Time Arrival Problem", IEEE TITS,
  Vol. 25, No. 8, 2024.

Algorithm (Tabular On-Policy GAC, simplified from Algorithm 3):
  State  = (node, remaining_budget)
  Policy = Softmax over logits [(node, budget, action)]
  Critic = E-VFA baseline b[(node, budget)]  P[on-time from s]

  Per episode:
    1. Roll out trajectory under current policy (no execution uncertainty).
    2. Critic update (TD, reverse order):
         y = 1 if terminal and on-time, 0 if terminal and late,
             else b[next_state]
         b[s] += lr_critic * (y - b[s])
    3. Actor update (GPG with E-VFA baseline, Eq. 7):
         advantage = reward - b[s_0]   (reward = 1 if on-time, else 0)
         for each (s, a) in trajectory:
           logits[s][a] += lr_actor * advantage * (1 - (a|s))
           logits[s][a'] -= lr_actor * advantage * (a'|s)  for a'  a
         (equivalent to: logits[s][a] += lr_actor * advantage * _ log (a|s))

  No execution uncertainty  only travel-time sampling.
"""

import numpy as np
from collections import defaultdict


def run_segac_routing(env, budget: int,
                      episodes: int = 50000,
                      lr_actor: float = 0.01,
                      lr_critic: float = 0.1,
                      seed: int = 42) -> dict:
    """
    Tabular On-Policy GAC (SEGAC) on SiouxEnv.

    Parameters
    ----------
    env       : SiouxEnv instance
    budget    : integer time budget (minutes)
    episodes  : training episodes (default 50000)
    lr_actor  : actor learning rate (default 0.01)
    lr_critic : critic (E-VFA) learning rate (default 0.1)
    seed      : RNG seed (default 42)

    Returns
    -------
    dict with keys:
      'prob' : float       estimated on-time probability at origin
      'path' : list[int]   greedy path
    """
    rng = np.random.default_rng(seed)
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    # Softmax logits: [(node, b, action)] = float
    theta = defaultdict(float)

    # E-VFA baseline: b[(node, b)] = float in [0, 1]
    baseline = defaultdict(float)
    # Destination is always on-time (absorbing state with value 1)
    for b in range(budget + 1):
        baseline[(dest, b)] = 1.0

    def _softmax_policy(node, b):
        """Returns (actions, probs) for current state."""
        acts = successors.get(node, [])
        if not acts:
            return [], []
        logits = np.array([theta[(node, b, a)] for a in acts])
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        return acts, probs

    def _sample_tt(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0:
            return float(mean_t)
        cv2 = (sigma / mean_t) ** 2
        mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        return float(np.maximum(rng.lognormal(mu_ln, sigma_ln), 0.01))

    max_steps = len(env.nodes) * 2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    for _ in range(episodes):
        # --- Collect trajectory ---
        traj = []          # list of (node, b, action, next_node, next_b)
        node, b = origin, budget
        visited = {node}

        for _ in range(max_steps):
            if node == dest or b < 0:
                break
            acts, probs = _softmax_policy(node, b)
            acts = [a for a in acts if a not in visited]
            if not acts:
                break
            # Re-compute probs for unvisited actions only
            logits = np.array([theta[(node, b, a)] for a in acts])
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()

            idx = int(rng.choice(len(acts), p=probs))
            action = acts[idx]
            t = _sample_tt(node, action)
            b_next = b - round(t)

            traj.append((node, b, action, action, b_next))
            visited.add(action)
            node = action
            b = b_next

        if not traj:
            continue

        # Determine outcome
        on_time = (node == dest and b >= 0)
        reward = 1.0 if on_time else 0.0

        # --- Critic update (TD, reverse) ---
        for i in range(len(traj) - 1, -1, -1):
            s_node, s_b, _, next_node, next_b = traj[i]
            last = (i == len(traj) - 1)
            if last:
                if next_node == dest and next_b >= 0:
                    y = 1.0
                elif next_b < 0:
                    y = 0.0
                else:
                    y = baseline[(next_node, max(0, next_b))]
            else:
                y = baseline[(next_node, max(0, next_b))]
            baseline[(s_node, s_b)] += lr_critic * (y - baseline[(s_node, s_b)])

        # --- Actor update (GPG with E-VFA baseline) ---
        b0 = baseline[(origin, budget)]
        advantage = reward - b0
        if abs(advantage) < 1e-9:
            continue

        # Reconstruct visited set prefix for each step so the actor update
        # uses the same available-action set that was used during rollout.
        # Using acts_all (all successors) would compute gradients over actions
        # that were never available at that step, causing a policy mismatch.
        visited_prefix = {origin}
        for s_node, s_b, action, _, _ in traj:
            acts_avail = [a for a in successors.get(s_node, []) if a not in visited_prefix]
            if not acts_avail:
                visited_prefix.add(action)
                continue
            logits = np.array([theta[(s_node, s_b, a)] for a in acts_avail])
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()

            # _ log (a|s) = e_a - (|s)
            #  += lr * advantage * _ log (a|s)
            for j, a in enumerate(acts_avail):
                grad = (1.0 if a == action else 0.0) - probs[j]
                theta[(s_node, s_b, a)] += lr_actor * advantage * grad
            visited_prefix.add(action)

    # ------------------------------------------------------------------
    # Greedy path (argmax logit, no revisits)
    # ------------------------------------------------------------------
    path = [origin]
    node, b = origin, budget
    visited = {node}
    for _ in range(max_steps):
        if node == dest or b < 0:
            break
        acts = [a for a in successors.get(node, []) if a not in visited]
        if not acts:
            break
        action = max(acts, key=lambda a: theta[(node, b, a)])
        mean_t, _ = edges[(node, action)]
        b -= round(mean_t)
        visited.add(action)
        path.append(action)
        node = action

    prob = float(baseline[(origin, budget)])

    return {'prob': prob, 'path': path}
