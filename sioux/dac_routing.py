"""
dac_routing.py  DAC (Discrete Actor-Critic with decoupled entropy, Chen et al. 2024).

Reference:
  Chen et al., "A Systematic Reexamination of Discrete Off-Policy Actor-Critic
  Methods", 2024.

Algorithm (NPG-RKL(, 0)):
  State  = (node, remaining_budget)
  Policy = Softmax over logits [(node, budget, action)]
  Critic = single Q-table Q[(node, budget, action)]

  Key insight: DSAC performance issues stem from coupled entropy regularization
  between Actor and Critic ( = ). The fix:
    - Critic:  = 0 (no entropy, hard Bellman operator)
    - Actor:  adaptive (entropy-regularized NPG + reverse KL projection)

  Critic update (Policy Evaluation,  = 0):
    y = r +  * max_a Q_target(s', a)
    L = (Q(s, a) - y)^2

  Actor update (NPG + RKL):
    Step 1: _{t+1/2}  _t * exp( * Q(s))
    Step 2: _new  (_{t+1/2})^{1 / (1 +  * )}   (RKL projection)

  Entropy adaptation:
       + lr_ent * (H_current - H_target)
    H_target = -0.2 * log(|A_max|)

No execution uncertainty  only travel-time sampling. Off-Policy with replay buffer.
"""

import numpy as np
from collections import defaultdict, deque


def run_dac_routing(env, budget: int,
                    episodes: int = 50000,
                    lr_critic: float = 3e-4,
                    lr_actor: float = 0.1,
                    lr_ent: float = 3e-4,
                    gamma: float = 0.99,
                    buffer_size: int = 10000,
                    batch_size: int = 256,
                    target_update_freq: int = 100,
                    seed: int = 42) -> dict:
    """
    DAC (NPG-RKL(, 0)) on SiouxEnv.

    Parameters
    ----------
    env                : SiouxEnv instance
    budget             : integer time budget (minutes)
    episodes           : training episodes (default 50000)
    lr_critic          : Critic learning rate (default 3e-4)
    lr_actor           : Actor learning rate  (default 0.1)
    lr_ent             : Entropy coefficient learning rate (default 3e-4)
    gamma              : discount factor (default 0.99)
    buffer_size        : replay buffer capacity (default 10000)
    batch_size         : mini-batch size (default 256)
    target_update_freq : target network update frequency (default 100)
    seed               : RNG seed (default 42)

    Returns
    -------
    dict with keys:
      'prob'    : float       on-time arrival probability (max Q at origin)
      'path'    : list[int]   greedy path using Q-values
    """
    rng = np.random.default_rng(seed)
    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    # Q table: Q[(node, budget, action)] = float
    Q = defaultdict(float)
    Q_target = defaultdict(float)

    # Policy logits: theta[(node, budget, action)] = float
    theta = defaultdict(float)

    # Replay buffer: FIFO queue of (node, b, action, reward, next_node, next_b)
    replay_buffer = deque(maxlen=buffer_size)

    # Entropy coefficient 
    tau = 1.0

    # Target entropy
    max_actions = max((len(v) for v in successors.values()), default=1)
    H_target = -0.2 * np.log(max_actions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_tt(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0:
            return float(mean_t)
        cv2 = (sigma / mean_t) ** 2
        mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        return float(np.maximum(rng.lognormal(mu_ln, sigma_ln), 0.01))

    def _softmax_policy(node, b):
        """Return (actions, probs) for current state using theta logits."""
        acts = successors.get(node, [])
        if not acts:
            return [], []
        logits = np.array([theta[(node, b, a)] for a in acts])
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        return acts, probs

    def _entropy(probs):
        """Compute entropy of a probability distribution."""
        # Mask out zero probabilities for log
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log(probs)))

    max_steps = len(env.nodes) * 2

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(episodes):
        # --- Collect one episode ---
        node = origin
        b = budget
        visited = {node}

        episode_transitions = []

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

            # Sample action from softmax policy
            idx = int(rng.choice(len(acts), p=probs))
            action = acts[idx]

            t = _sample_tt(node, action)
            b_next = b - round(t)

            next_node = action
            if next_node == dest and b_next >= 0:
                r = 1.0
            else:
                r = 0.0

            episode_transitions.append((node, b, action, r, next_node, b_next))

            visited.add(action)
            node = next_node
            b = b_next

        # --- Store in replay buffer ---
        for trans in episode_transitions:
            replay_buffer.append(trans)

        # --- Sample batch and update Critic ---
        if len(replay_buffer) >= batch_size:
            batch_indices = rng.integers(0, len(replay_buffer), size=batch_size)
            for idx in batch_indices:
                s_node, s_b, s_a, s_r, s_next_node, s_next_b = replay_buffer[idx]

                # Target: y = r + gamma * max_a Q_target(s', a)
                if s_next_node == dest and s_next_b >= 0:
                    y = s_r
                elif s_next_b < 0:
                    y = s_r
                else:
                    next_acts = successors.get(s_next_node, [])
                    if next_acts:
                        max_q_next = max(Q_target[(s_next_node, s_next_b, a)] for a in next_acts)
                    else:
                        max_q_next = 0.0
                    y = s_r + gamma * max_q_next

                # TD update: Q(s, a)  Q(s, a) - lr_critic * (Q(s, a) - y)
                q_val = Q[(s_node, s_b, s_a)]
                Q[(s_node, s_b, s_a)] += lr_critic * (y - q_val)

        # --- Update Actor (NPG + RKL) ---
        for s_node, s_b, s_a, _, _, _ in episode_transitions:
            acts = successors.get(s_node, [])
            if not acts:
                continue

            # Compute  = softmax(theta[s])
            logits = np.array([theta[(s_node, s_b, a)] for a in acts])
            logits_max = logits.max()
            logits_stable = logits - logits_max
            probs = np.exp(logits_stable)
            probs /= probs.sum()

            # Step 1: NPG intermediate policy
            # logits_1/2 = log() +  * Q(s)
            q_vals = np.array([Q[(s_node, s_b, a)] for a in acts])
            log_pi = logits_stable - np.log(probs.sum())  # log() from logits
            logits_half = log_pi + lr_actor * q_vals

            # Softmax to get _{t+1/2}
            logits_half_max = logits_half.max()
            logits_half_stable = logits_half - logits_half_max
            pi_half = np.exp(logits_half_stable)
            pi_half /= pi_half.sum()

            # Step 2: Reverse KL projection
            # _new  (_{t+1/2})^{1 / (1 +  * )}
            alpha = 1.0 / (1.0 + lr_actor * tau)
            pi_new_unnorm = pi_half ** alpha
            Z = pi_new_unnorm.sum()
            if Z > 0:
                pi_new = pi_new_unnorm / Z
            else:
                pi_new = np.ones_like(pi_half) / len(pi_half)

            # Update logits: theta[s] = log(_new) + constant
            # Constant = logits_max (from earlier) to maintain scale
            new_logits = np.log(np.maximum(pi_new, 1e-15)) + logits_max
            for j, a in enumerate(acts):
                theta[(s_node, s_b, a)] = float(new_logits[j])

            # --- Update entropy coefficient  ---
            H_current = _entropy(pi_new)
            tau += lr_ent * (H_current - H_target)

        # --- Periodic target network update ---
        if (ep + 1) % target_update_freq == 0:
            # Copy Q to Q_target
            for key, val in Q.items():
                Q_target[key] = val

    # ------------------------------------------------------------------
    # Greedy path (argmax Q, mean travel times, no revisits)
    # ------------------------------------------------------------------
    path = [origin]
    node = origin
    b = budget
    visited = {node}
    for _ in range(max_steps):
        if node == dest or b < 0:
            break
        acts = [a for a in successors.get(node, []) if a not in visited]
        if not acts:
            break
        action = max(acts, key=lambda a: Q[(node, b, a)])
        mean_t, _ = edges[(node, action)]
        b -= round(mean_t)
        visited.add(action)
        path.append(action)
        node = action

    # ------------------------------------------------------------------
    # On-time probability from Q at origin
    # ------------------------------------------------------------------
    origin_acts = successors.get(origin, [])
    if origin_acts:
        prob = float(max(Q[(origin, budget, a)] for a in origin_acts))
    else:
        prob = 0.0

    return {'prob': prob, 'path': path, 'q_table': dict(Q)}
