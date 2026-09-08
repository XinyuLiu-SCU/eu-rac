import heapq, math
import os
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Network data (module-level, shared).
# Beijing runner sets CHENGDU_PERIOD before importing this module.
_PERIOD_FILES = {
    'weekday_peak': 'Weekday_Peak_network.csv',
    'weekday_offpeak': 'Weekday_Offpeak_network.csv',
    'weekend_peak': 'Weekend_Peak_network.csv',
    'weekend_offpeak': 'Weekend_Offpeak_network.csv',
}
_CHENGDU_PERIOD = os.environ.get('CHENGDU_PERIOD', 'weekday_peak').strip().lower()
if _CHENGDU_PERIOD not in _PERIOD_FILES:
    raise ValueError(
        f"Unsupported CHENGDU_PERIOD {_CHENGDU_PERIOD!r}. "
        f"Supported periods: {', '.join(_PERIOD_FILES)}"
    )
_NET_DIR = Path(__file__).parent.parent / 'network' / 'Chengdu'
_net_df = __import__('pandas').read_csv(_NET_DIR / _PERIOD_FILES[_CHENGDU_PERIOD])
_net_df = _net_df[[c for c in _net_df.columns if not str(c).startswith('Unnamed')]]
_mean_edges = {}; _sigma_edges = {}
_out_degree = defaultdict(int); _successors_g = defaultdict(list)
for _, row in _net_df.iterrows():
    u, v = int(row['From']), int(row['To'])
    _mean_edges[(u, v)] = float(row['Cost'])
    # sigma is loaded from npy, approximate if not available
    _out_degree[u] += 1
    _successors_g.setdefault(u, []).append(v)
_nodes_set = set()
for u, v in _mean_edges: _nodes_set.add(u); _nodes_set.add(v)
_NUM_NODES = max(_nodes_set) + 1
_MAX_OUT_DEGREE = max(_out_degree.values()) if _out_degree else 1

def _precompute_let_to_go(dest):
    """Dijkstra from dest backwards to all nodes. Returns dict node->rounded_cost."""
    rev = defaultdict(list)
    for (u, v), cost in _mean_edges.items():
        rev.setdefault(v, []).append((u, max(1, round(cost))))
    dist = {dest: 0}; pq = [(0, dest)]; visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        for v, c in rev.get(u, []):
            nd = d + c
            if nd < dist.get(v, float('inf')): dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist


def _precompute_dijkstra_next_hop(dest):
    """Precompute node -> next hop on rounded-cost Dijkstra path to dest."""
    rev = defaultdict(list)
    for (u, v), cost in _mean_edges.items():
        rev.setdefault(v, []).append((u, max(1, round(cost))))
    dist = {dest: 0}; pq = [(0, dest)]; visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        for pred, c in rev.get(u, []):
            nd = d + c
            if nd < dist.get(pred, float('inf')):
                dist[pred] = nd; heapq.heappush(pq, (nd, pred))
    next_hop = {}
    for u in dist:
        if u == dest: continue
        candidates = [v for v in _successors_g.get(u, []) if v in dist]
        if not candidates: continue
        next_hop[u] = min(candidates, key=lambda v: max(1, round(_mean_edges[(u, v)])) + dist[v])
    return next_hop
def _k_shortest_path_stats(node, dest, K=5, budget=200):
    """Compute mean and std of up to K shortest paths from node to dest.
    Returns (means, stds) lists of length ,K.  Cached per node."""
    if not hasattr(_k_shortest_path_stats, '_cache'):
        _k_shortest_path_stats._cache = {}
    cache = _k_shortest_path_stats._cache
    if node in cache: return cache[node]
    if node == dest:
        cache[node] = ([0.0], [0.0]); return cache[node]

    means, stds = [], []
    excluded_edges = set()

    for _ in range(K):
        dist = {node: 0.0}; prev = {}; pq = [(0.0, node)]
        found = False
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            if u == dest: found = True; break
            for v in _successors_g.get(u, []):
                if (u, v) in excluded_edges: continue
                mt = _mean_edges.get((u, v), 1.0)
                nd = d + max(1, round(mt))
                if nd < dist.get(v, float('inf')): dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if not found: break

        # Reconstruct path and compute stats
        path_nodes = [dest]; n = dest
        while n in prev: path_nodes.append(prev[n]); n = prev[n]
        path_nodes.reverse()
        if len(path_nodes) < 2: break

        total_mean = 0.0; total_var = 0.0
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            mt = _mean_edges.get((u, v), 1.0)
            total_mean += max(1, round(mt))
        means.append(total_mean)
        stds.append(np.sqrt(total_var) if total_var > 0 else 1.0)

        # Exclude first edge for next iteration
        if len(path_nodes) >= 2:
            excluded_edges.add((path_nodes[0], path_nodes[1]))

    if not means: means, stds = [float('inf')], [1.0]
    # Pad to K
    while len(means) < K: means.append(means[-1]); stds.append(stds[-1])
    cache[node] = (means[:K], stds[:K])
    return cache[node]

#  DAC Q-Network (Chen et al., 2024) 
# Critic: standard Q(s,a), NO entropy (zeta=0).  Actor: NPG+RKL with adaptive tau.
# Input: node_emb(32) + action_emb(32) + t/B + LET/B + slack/B + deg_norm = 68

class DACQNetwork(nn.Module):
    def __init__(self, num_nodes, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        self.action_embedding = nn.Embedding(num_nodes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim*2 + 4, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))  # linear scalar Q ,no Sigmoid

    def forward(self, node_ids, action_ids, budget_remaining, let_to_go, budget_max, out_degrees):
        n_emb = self.node_embedding(node_ids); a_emb = self.action_embedding(action_ids)
        b_norm = (budget_remaining.float() / budget_max).unsqueeze(1)
        l_norm = (let_to_go.float() / budget_max).unsqueeze(1)
        s_norm = ((budget_remaining.float() - let_to_go.float()) / budget_max).unsqueeze(1)
        d_norm = (out_degrees.float() / max(_MAX_OUT_DEGREE, 1)).unsqueeze(1)
        x = torch.cat([n_emb, a_emb, b_norm, l_norm, s_norm, d_norm], dim=1)
        return self.net(x).squeeze(-1)


class _DACQWrapper:
    def __init__(self, q_net, let_to_go, B, device):
        self.q_net = q_net; self.let = let_to_go; self.B = B
        self.dev = device; self._cache = {}
    def __getitem__(self, key):
        node, b, action = key; k = (node, b, action)
        if k in self._cache: return self._cache[k]
        with torch.no_grad():
            n = torch.tensor([node], device=self.dev); a = torch.tensor([action], device=self.dev)
            bb = torch.tensor([b], device=self.dev)
            let = torch.tensor([self.let.get(node, 0)], device=self.dev)
            deg = torch.tensor([_out_degree.get(node, 0)], device=self.dev)
            val = float(self.q_net(n, a, bb, let, self.B, deg).item())
        self._cache[k] = val; return val
    def get(self, key, default=None):
        try: return self[key]
        except: return default
    def __len__(self): return len(self._cache)
    def keys(self): return list(self._cache.keys())


def _actor_probs(theta, node, budget, actions):
    if not actions:
        return np.asarray([], dtype=np.float64)
    logits = np.asarray([theta[(node, budget, a)] for a in actions], dtype=np.float64)
    logits -= logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def run_dac_routing(env, budget: int,
                    episodes: int = 50000, lr_critic: float = 3e-4, lr_actor: float = 0.1,
                    lr_ent: float = 3e-4, gamma: float = 0.99,
                    buffer_size: int = 10000, batch_size: int = 256,
                    target_update_freq: int = 100, seed: int = 42,
                    use_dijkstra_warm_start: bool = False,
                    use_dijkstra_replay_warm_start: bool = False,
                    use_persistent_expert_replay: bool = False,
                    expert_replay_fraction: float = 0.10,
                    use_adaptive_tau: bool = False,
                    fixed_tau: float = 1.0,
                    actor_state_source: str = 'trajectory',
                    evaluate_dijkstra_fallback: bool = False,
                    fallback_q_gap_threshold: float = 0.01,
                    deployment_audit_detail: bool = True,
                    evaluate_q_confidence_audit: bool = False,
                    evaluate_actor_learning_audit: bool = False,
                    diagnostic_interval: int = 5000,
                    mc_episodes: int = 10000) -> dict:
    """DAC (Chen et al., 2024): Critic zeta=0 (NO entropy), Actor NPG+RKL with adaptive tau."""
    if actor_state_source not in ('trajectory', 'replay'):
        raise ValueError("actor_state_source must be 'trajectory' or 'replay'")
    rng = np.random.default_rng(seed)
    edges = env.edges; successors = env.successors
    origin = env.origin; dest = env.dest
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    let_to_go = _precompute_let_to_go(dest); B = budget

    theta = defaultdict(float); tau = float(fixed_tau)
    tau_initial = tau
    max_actions = max((len(v) for v in successors.values()), default=1)
    H_target = -0.2 * np.log(max_actions)

    q_net = DACQNetwork(_NUM_NODES).to(device)
    q_target = DACQNetwork(_NUM_NODES).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_opt = optim.Adam(q_net.parameters(), lr=lr_critic)
    replay_buffer = deque(maxlen=buffer_size)

    #  Critic warm-start: supervised along Dijkstra path 
    import heapq as _hq_critic
    _dist_c = {origin: 0.0}; _prev_c = {}; _pq_c = [(0.0, origin)]
    while _pq_c:
        _d, _u = _hq_critic.heappop(_pq_c)
        if _d > _dist_c.get(_u, float('inf')): continue
        for _v in successors.get(_u, []):
            _mt, __ = edges[(_u, _v)]; _nd = _d + max(1, round(_mt))
            if _nd < _dist_c.get(_v, float('inf')): _dist_c[_v] = _nd; _prev_c[_v] = _u; _hq_critic.heappush(_pq_c, (_nd, _v))
    _ws_pairs_dac = []
    if use_dijkstra_warm_start and dest in _prev_c:
        _pn_dac = [dest]; _n_dac = dest
        while _n_dac in _prev_c: _pn_dac.append(_prev_c[_n_dac]); _n_dac = _prev_c[_n_dac]
        _pn_dac.reverse()
        _rem_dac = budget
        for _i in range(len(_pn_dac) - 1):
            _u, _v = _pn_dac[_i], _pn_dac[_i+1]
            for _bb in range(max(0, _rem_dac-10), _rem_dac + 4):
                _ws_pairs_dac.append((_u, _bb, _v))
            _mt, __ = edges[(_u, _v)]; _rem_dac -= max(1, round(_mt))
        for _ in range(50):
            _ws_loss = torch.tensor(0.0, device=device); _ws_n = 0
            for _u, _b, _a in _ws_pairs_dac:
                acts_ws = successors.get(_u, [])
                if len(acts_ws) <= 1: continue
                n_src = torch.full((len(acts_ws),), _u, device=device)
                a_t = torch.tensor(acts_ws, device=device)
                bb_t = torch.full((len(acts_ws),), _b, device=device)
                let_t = torch.tensor([let_to_go.get(_u, 0)], device=device).expand(len(acts_ws))
                deg_t = torch.tensor([_out_degree.get(_u, 0)], device=device).expand(len(acts_ws))
                qvs = q_net(n_src, a_t, bb_t, let_t, B, deg_t)
                targets = torch.zeros(len(acts_ws), device=device)
                for _j, _aa in enumerate(acts_ws):
                    if _aa == _a: targets[_j] = 1.0
                _ws_loss = _ws_loss + (qvs - targets).pow(2).mean(); _ws_n += 1
            if _ws_n == 0: continue
            _ws_loss = _ws_loss / _ws_n
            q_opt.zero_grad(); _ws_loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 5.0); q_opt.step()
        q_target.load_state_dict(q_net.state_dict())
    # Optional replay-only warm-start: one Dijkstra expert trajectory. This does
    # not initialize either the critic parameters or the actor table.
    expert_replay_path = []
    expert_replay_buffer = []
    expert_samples_used = 0
    if (use_dijkstra_replay_warm_start or use_persistent_expert_replay) and dest in _prev_c:
        expert_replay_path = [dest]; _expert_node = dest
        while _expert_node in _prev_c:
            expert_replay_path.append(_prev_c[_expert_node]); _expert_node = _prev_c[_expert_node]
        expert_replay_path.reverse()
        # The mean shortest path costs 119 for B=116. Draw a separate,
        # reproducible travel-time realization until the expert trajectory is
        # genuinely on-time; do not manufacture a terminal reward.
        _expert_rng = np.random.default_rng(seed + 3000007)
        _expert_times = None
        for _ in range(100000):
            _candidate_times = []; _follows_expert_path = True
            for _u, _v in zip(expert_replay_path[:-1], expert_replay_path[1:]):
                # Replicate evaluator execution uncertainty with an isolated RNG.
                _actual = _v
                if (_u in getattr(env, 'uncertain_edges', {}) and
                        _v == env.uncertain_edges.get(_u) and
                        _expert_rng.random() >= env.exec_prob):
                    _alternatives = [a for a in successors.get(_u, []) if a != _v]
                    if _alternatives:
                        _actual = int(_expert_rng.choice(_alternatives))
                if _actual != _v:
                    _follows_expert_path = False; break
                _mean_t, _sigma = edges[(_u, _actual)]
                if _sigma <= 0:
                    _sampled = max(1, round(_mean_t))
                else:
                    _cv2 = (_sigma / _mean_t) ** 2
                    _mu = np.log(_mean_t) - 0.5 * np.log(1.0 + _cv2)
                    _sd = np.sqrt(np.log(1.0 + _cv2))
                    _sampled = max(1, round(_expert_rng.lognormal(_mu, _sd)))
                _candidate_times.append(_sampled)
            if _follows_expert_path and sum(_candidate_times) <= budget:
                _expert_times = _candidate_times; break
        if _expert_times is None:
            raise RuntimeError('Could not sample an on-time Dijkstra expert trajectory')
        _expert_budget = budget; _expert_visited = {origin}
        for (_u, _v), _sampled in zip(zip(expert_replay_path[:-1], expert_replay_path[1:]), _expert_times):
            _next_budget = _expert_budget - _sampled
            _next_visited = _expert_visited | {_v}
            _next_actions = tuple(a for a in successors.get(_v, []) if a not in _next_visited)
            _reward = 1.0 if (_v == dest and _next_budget >= 0) else 0.0
            _expert_transition = (_u, _expert_budget, _v, _reward, _v, _next_budget, _next_actions)
            expert_replay_buffer.append(_expert_transition)
            if use_dijkstra_replay_warm_start:
                replay_buffer.append(_expert_transition)
            _expert_budget = _next_budget; _expert_visited = _next_visited
    replay_initial_size = len(replay_buffer)
    replay_initial_successes = sum(1 for x in replay_buffer if x[3] > 0)
    print(f'  [DAC] Dijkstra parameter warm-start={use_dijkstra_warm_start}; '
          f'critic nodes={len(set(p[0] for p in _ws_pairs_dac)) if _ws_pairs_dac else 0}; '
          f'replay warm-start={use_dijkstra_replay_warm_start}; persistent expert={use_persistent_expert_replay}; '
          f'initial replay={replay_initial_size}; expert size={len(expert_replay_buffer)}')

    def _Q(node, b, action):
        with torch.no_grad():
            n = torch.tensor([node], device=device); a = torch.tensor([action], device=device)
            bb = torch.tensor([b], device=device)
            let = torch.tensor([let_to_go.get(node, 0)], device=device)
            deg = torch.tensor([_out_degree.get(node, 0)], device=device)
            return float(q_net(n, a, bb, let, B, deg).item())

    def _Q_target_val(node, b, action):
        with torch.no_grad():
            n = torch.tensor([node], device=device); a = torch.tensor([action], device=device)
            bb = torch.tensor([b], device=device)
            let = torch.tensor([let_to_go.get(node, 0)], device=device)
            deg = torch.tensor([_out_degree.get(node, 0)], device=device)
            return float(q_target(n, a, bb, let, B, deg).item())

    def _sample_tt(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0: return float(mean_t)
        cv2 = (sigma/mean_t)**2
        return float(max(rng.lognormal(np.log(mean_t)-0.5*np.log(1+cv2), np.sqrt(np.log(1+cv2))), 0.01))

    def _entropy(probs):
        probs = np.asarray(probs, dtype=np.float64); probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log(probs)))

    max_steps = len(env.nodes) * 2
    training_successes = 0
    first_successful_episode = None
    critic_losses = []
    q_count = 0; q_sum = 0.0; q_sumsq = 0.0; q_min_seen = float('inf'); q_max_seen = float('-inf')
    actor_entropy_checkpoints, tau_checkpoints = [], []
    origin_actor_evolution = []
    actor_losses, theta_gradient_norms = [], []
    old_to_new_kls, rkl_projection_changes = [], []
    actor_update_count = 0
    actor_checkpoint_episodes = {1, 100, 1000, 5000, 10000, 20000, episodes}
    critic_action_gap_evolution = []
    visited_states, updated_state_action_pairs = set(), set()
    actor_updated_states = set()
    replay_sampled_states = set()
    tau_min = tau_max = tau
    actor_learning_audit = None

    def _theta_stats(previous=None):
        keys = set(theta.keys())
        if previous is not None:
            keys.update(previous.keys())
        vals = np.asarray([float(theta.get(k, 0.0)) for k in keys], dtype=np.float64) if keys else np.asarray([0.0])
        if previous is None:
            delta_l2 = 0.0
        else:
            delta_l2 = float(np.sqrt(sum((float(theta.get(k, 0.0)) - float(previous.get(k, 0.0))) ** 2 for k in keys)))
        return {
            'theta_count': int(len(keys)),
            'theta_mean': float(vals.mean()),
            'theta_std': float(vals.std()),
            'theta_min': float(vals.min()),
            'theta_max': float(vals.max()),
            'theta_l2_norm': float(np.linalg.norm(vals)),
            'delta_theta_l2_from_previous_checkpoint': delta_l2,
        }

    def _origin_policy_stats():
        oa = list(successors.get(origin, []))
        op = _actor_probs(theta, origin, budget, oa)
        return {
            'actions': [int(a) for a in oa],
            'pi': {int(a): float(p) for a, p in zip(oa, op)},
            'entropy': _entropy(op),
            'max_probability': float(op.max()) if len(op) else 0.0,
        }

    # Warm-start theta along Dijkstra
    import heapq as _hq
    _dist = {origin: 0.0}; _prev = {}; _pq = [(0.0, origin)]
    while _pq:
        _d, _u = _hq.heappop(_pq)
        if _d > _dist.get(_u, float('inf')): continue
        for _v in successors.get(_u, []):
            _mt, __ = edges[(_u, _v)]; _nd = _d + max(1, round(_mt))
            if _nd < _dist.get(_v, float('inf')): _dist[_v] = _nd; _prev[_v] = _u; _hq.heappush(_pq, (_nd, _v))
    if use_dijkstra_warm_start and dest in _prev:
        _path = []; _n = dest
        while _n in _prev: _path.append(_n); _n = _prev[_n]
        _path.append(origin); _path.reverse(); _rem = budget
        for _i in range(len(_path)-1):
            _node, _act = _path[_i], _path[_i+1]
            _min_b = int(_dist[dest]-_dist[_node])
            for _bb in range(_min_b, _rem+4): theta[(_node, _bb, _act)] = 2.0
            _mt, __ = edges[(_node, _act)]; _rem -= max(1, round(_mt))

    if evaluate_actor_learning_audit:
        actor_learning_audit = {
            'checkpoint_interval': 100,
            'note': 'Actor is tabular theta, not a torch optimizer. Gradient stats use the realized analytic theta delta divided by lr_actor immediately before assignment.',
            'checkpoints': [],
            'update_records': [],
        }
        _actor_audit_prev_theta = dict(theta)
        _actor_audit_window = {
            'gradient_l2': [], 'gradient_max_abs': [], 'gradient_mean_abs': [],
            'advantage_mean': [], 'advantage_std': [], 'advantage_min': [], 'advantage_max': [],
            'advantage_abs_lt_1e_6_pct': [], 'update_mean_abs': [], 'update_max_abs': [],
        }
        _init_stats = _theta_stats(None)
        _init_stats.update({'episode': 0, 'origin_policy': _origin_policy_stats()})
        actor_learning_audit['checkpoints'].append(_init_stats)
    else:
        _actor_audit_prev_theta = None
        _actor_audit_window = None

    for ep in range(episodes):
        node, b = origin, budget; visited = {node}; ep_trans = []
        visited_states.add((node, b))
        for _ in range(max_steps):
            if node == dest or b < 0: break
            all_acts = successors.get(node, []); acts = [a for a in all_acts if a not in visited]
            if not acts: break
            logits = np.array([theta[(node, b, a)] for a in acts])
            logits -= logits.max(); probs = np.exp(logits); probs /= probs.sum()
            idx = int(rng.choice(len(acts), p=probs)); action = acts[idx]
            t = _sample_tt(node, action); b_next = b - round(t)
            r_val = 1.0 if (action == dest and b_next >= 0) else 0.0
            next_visited = visited | {action}
            next_actions = tuple(a for a in successors.get(action, []) if a not in next_visited)
            ep_trans.append((node, b, action, r_val, action, b_next, next_actions))
            visited = next_visited; node = action; b = b_next
            visited_states.add((node, b))

        episode_success = bool(node == dest and b >= 0)
        if episode_success:
            training_successes += 1
            if first_successful_episode is None:
                first_successful_episode = ep + 1
        for trans in ep_trans: replay_buffer.append(trans)

        # Critic update (zeta=0: NO entropy in Bellman target)
        normal_batch_size = batch_size
        expert_batch_size = 0
        if use_persistent_expert_replay and expert_replay_buffer:
            expert_batch_size = int(round(batch_size * expert_replay_fraction))
            expert_batch_size = min(batch_size, max(1, expert_batch_size))
            normal_batch_size = batch_size - expert_batch_size
        batch = []

        if len(replay_buffer) >= max(1, normal_batch_size):
            indices = rng.integers(0, len(replay_buffer), size=normal_batch_size)
            batch = [replay_buffer[int(idx)] for idx in indices]
            if expert_batch_size:
                expert_indices = rng.integers(0, len(expert_replay_buffer), size=expert_batch_size)
                batch.extend(expert_replay_buffer[int(idx)] for idx in expert_indices)
                expert_samples_used += expert_batch_size
            replay_sampled_states.update((x[0], x[1]) for x in batch)
            updated_state_action_pairs.update((x[0], x[1], x[2]) for x in batch)
            targets = [float(x[3]) for x in batch]
            flat_nodes, flat_budgets, flat_actions, flat_probs, slices = [], [], [], [], []
            for i, (_, _, _, s_r, s_next, s_nb, next_actions) in enumerate(batch):
                if (s_next == dest and s_nb >= 0) or s_nb < 0 or not next_actions:
                    continue
                probs_next = _actor_probs(theta, s_next, s_nb, next_actions)
                start = len(flat_actions)
                flat_nodes.extend([s_next] * len(next_actions)); flat_budgets.extend([s_nb] * len(next_actions))
                flat_actions.extend(next_actions); flat_probs.extend(probs_next.tolist())
                slices.append((i, start, len(flat_actions)))
            if flat_actions:
                with torch.no_grad():
                    fn = torch.tensor(flat_nodes, device=device); fa = torch.tensor(flat_actions, device=device)
                    fb = torch.tensor(flat_budgets, device=device)
                    fl = torch.tensor([let_to_go.get(n, 0) for n in flat_nodes], device=device)
                    fd = torch.tensor([_out_degree.get(n, 0) for n in flat_nodes], device=device)
                    flat_q = q_target(fn, fa, fb, fl, B, fd).cpu().numpy()
                flat_probs_np = np.asarray(flat_probs)
                for i, start, end in slices:
                    targets[i] += gamma * float(np.dot(flat_probs_np[start:end], flat_q[start:end]))
            nodes_t = torch.tensor([x[0] for x in batch], device=device)
            actions_t = torch.tensor([x[2] for x in batch], device=device)
            budgets_t = torch.tensor([x[1] for x in batch], device=device)
            lets_t = torch.tensor([let_to_go.get(x[0], 0) for x in batch], device=device)
            degrees_t = torch.tensor([_out_degree.get(x[0], 0) for x in batch], device=device)
            pred = q_net(nodes_t, actions_t, budgets_t, lets_t, B, degrees_t)
            loss = (pred - torch.tensor(targets, device=device)).pow(2).mean()
            q_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 5.0); q_opt.step()
            critic_losses.append(float(loss.detach().cpu()))
            q_batch = pred.detach().cpu().numpy().astype(np.float64)
            q_count += q_batch.size; q_sum += float(q_batch.sum()); q_sumsq += float(np.square(q_batch).sum())
            q_min_seen = min(q_min_seen, float(q_batch.min())); q_max_seen = max(q_max_seen, float(q_batch.max()))

        # Actor update (NPG + RKL with adaptive tau)
        if actor_state_source == 'replay':
            actor_state_iter = [(x[0], x[1]) for x in batch]
        else:
            actor_state_iter = [(x[0], x[1]) for x in ep_trans]

        actor_states = []
        seen = set()
        for s_node, s_b in actor_state_iter:
            sk = (s_node, s_b)
            if sk in seen:
                continue
            seen.add(sk)
            acts = successors.get(s_node, [])
            if acts:
                actor_states.append((s_node, s_b, tuple(acts)))

        q_values_by_state = {}
        flat_nodes, flat_budgets, flat_actions, flat_slices = [], [], [], []
        for s_node, s_b, acts in actor_states:
            start = len(flat_actions)
            flat_nodes.extend([s_node] * len(acts))
            flat_budgets.extend([s_b] * len(acts))
            flat_actions.extend(acts)
            flat_slices.append(((s_node, s_b), start, len(flat_actions)))
        if flat_actions:
            with torch.no_grad():
                fn = torch.tensor(flat_nodes, device=device)
                fa = torch.tensor(flat_actions, device=device)
                fb = torch.tensor(flat_budgets, device=device)
                fl = torch.tensor([let_to_go.get(n, 0) for n in flat_nodes], device=device)
                fd = torch.tensor([_out_degree.get(n, 0) for n in flat_nodes], device=device)
                fq = q_net(fn, fa, fb, fl, B, fd).cpu().numpy().astype(np.float64)
            for sk, start, end in flat_slices:
                q_values_by_state[sk] = fq[start:end]

        episode_entropies = []
        for s_node, s_b, acts in actor_states:
            sk = (s_node, s_b)
            logits = np.array([theta[(s_node, s_b, a)] for a in acts])
            lm = logits.max(); ls = logits - lm; probs = np.exp(ls); probs /= probs.sum()
            qs = q_values_by_state[sk]
            adv = qs - float(np.dot(probs, qs))
            log_pi = ls - np.log(probs.sum()); logits_half = log_pi + lr_actor * qs
            lhm = logits_half.max(); lhs = logits_half - lhm
            pi_half = np.exp(lhs); pi_half /= pi_half.sum()
            af = 1.0/(1.0+lr_actor*tau); pn = pi_half**af; Z = pn.sum()
            pi_new = pn/Z if Z>0 else np.ones_like(pi_half)/len(pi_half)
            nl = np.log(np.maximum(pi_new, 1e-15)) + lm
            old_theta = logits.copy()
            theta_delta = nl - old_theta
            theta_grad = theta_delta / lr_actor
            if evaluate_actor_learning_audit:
                abs_adv = np.abs(adv)
                abs_delta = np.abs(theta_delta)
                grad_abs = np.abs(theta_grad)
                update_record = {
                    'episode': ep + 1,
                    'node': int(s_node),
                    'budget': int(s_b),
                    'actions': [int(a) for a in acts],
                    'advantage_mean': float(adv.mean()),
                    'advantage_std': float(adv.std()),
                    'advantage_min': float(adv.min()),
                    'advantage_max': float(adv.max()),
                    'advantage_abs_lt_1e_6_pct': float(np.mean(abs_adv < 1e-6) * 100.0),
                    'actor_gradient_l2_norm': float(np.linalg.norm(theta_grad)),
                    'actor_gradient_max_abs': float(grad_abs.max()),
                    'actor_gradient_mean_abs': float(grad_abs.mean()),
                    'update_mean_abs': float(abs_delta.mean()),
                    'update_max_abs': float(abs_delta.max()),
                }
                actor_learning_audit['update_records'].append(update_record)
                _actor_audit_window['gradient_l2'].append(update_record['actor_gradient_l2_norm'])
                _actor_audit_window['gradient_max_abs'].append(update_record['actor_gradient_max_abs'])
                _actor_audit_window['gradient_mean_abs'].append(update_record['actor_gradient_mean_abs'])
                _actor_audit_window['advantage_mean'].append(update_record['advantage_mean'])
                _actor_audit_window['advantage_std'].append(update_record['advantage_std'])
                _actor_audit_window['advantage_min'].append(update_record['advantage_min'])
                _actor_audit_window['advantage_max'].append(update_record['advantage_max'])
                _actor_audit_window['advantage_abs_lt_1e_6_pct'].append(update_record['advantage_abs_lt_1e_6_pct'])
                _actor_audit_window['update_mean_abs'].append(update_record['update_mean_abs'])
                _actor_audit_window['update_max_abs'].append(update_record['update_max_abs'])
            for j, a in enumerate(acts): theta[(s_node, s_b, a)] = float(nl[j])
            eps = 1e-15
            old_to_new_kls.append(float(np.sum(probs * (np.log(np.maximum(probs, eps)) - np.log(np.maximum(pi_new, eps))))))
            projection_rkl = float(np.sum(pi_new * (np.log(np.maximum(pi_new, eps)) - np.log(np.maximum(pi_half, eps)))))
            rkl_projection_changes.append(projection_rkl)
            actor_losses.append(projection_rkl - lr_actor * tau * _entropy(pi_new))
            theta_gradient_norms.append(float(np.linalg.norm(theta_delta) / lr_actor))
            actor_update_count += 1
            actor_updated_states.add(sk)
            # Adaptive entropy: tau <- tau + lr_ent * (H_current - H_target)
            if use_adaptive_tau:
                tau += lr_ent * (_entropy(pi_new) - H_target)
            episode_entropies.append(_entropy(pi_new))
            tau_min = min(tau_min, tau); tau_max = max(tau_max, tau)

        if evaluate_actor_learning_audit and ((ep + 1) % 100 == 0 or ep + 1 == episodes):
            checkpoint = _theta_stats(_actor_audit_prev_theta)
            checkpoint['episode'] = ep + 1
            checkpoint['origin_policy'] = _origin_policy_stats()
            for key, values in _actor_audit_window.items():
                checkpoint[key + '_mean'] = float(np.mean(values)) if values else float('nan')
                checkpoint[key + '_max'] = float(np.max(values)) if values else float('nan')
                checkpoint[key + '_min'] = float(np.min(values)) if values else float('nan')
            actor_learning_audit['checkpoints'].append(checkpoint)
            _actor_audit_prev_theta = dict(theta)
            for values in _actor_audit_window.values():
                values.clear()

        if ep == 0 or (ep + 1) % diagnostic_interval == 0 or ep + 1 == episodes:
            op = _actor_probs(theta, origin, budget, successors.get(origin, []))
            actor_entropy_checkpoints.append({'episode': ep + 1,
                'mean_updated_state_entropy': float(np.mean(episode_entropies)) if episode_entropies else float('nan'),
                'origin_entropy': _entropy(op)})
            tau_checkpoints.append({'episode': ep + 1, 'tau': float(tau)})
        if ep + 1 in actor_checkpoint_episodes:
            oa = list(successors.get(origin, []))
            op = _actor_probs(theta, origin, budget, oa)
            origin_actor_evolution.append({'episode': ep + 1,
                'theta': {int(a): float(theta[(origin, budget, a)]) for a in oa},
                'pi': {int(a): float(p) for a, p in zip(oa, op)},
                'entropy': _entropy(op)})
            origin_q = {int(a): _Q(origin, budget, a) for a in oa}
            origin_q_values = np.asarray(list(origin_q.values()), dtype=np.float64)
            state_gaps = []
            nonzero_count = 0
            for vs_node, vs_budget in visited_states:
                vs_actions = successors.get(vs_node, [])
                if not vs_actions:
                    continue
                vs_q = np.asarray([_Q(vs_node, vs_budget, a) for a in vs_actions], dtype=np.float64)
                state_gaps.append(float(vs_q.max() - vs_q.min()))
            for up_node, up_budget, up_action in updated_state_action_pairs:
                nonzero_count += int(abs(_Q(up_node, up_budget, up_action)) > 1e-12)
            critic_action_gap_evolution.append({'episode': ep + 1,
                'origin_q': origin_q,
                'origin_action_gap': float(origin_q_values.max() - origin_q_values.min()),
                'origin_q_mean': float(origin_q_values.mean()),
                'origin_q_std': float(origin_q_values.std()),
                'visited_state_count': len(visited_states),
                'updated_state_action_pair_count': len(updated_state_action_pairs),
                'nonzero_q_fraction': nonzero_count / len(updated_state_action_pairs) if updated_state_action_pairs else 0.0,
                'average_visited_state_action_gap': float(np.mean(state_gaps)) if state_gaps else 0.0})

        # Target network soft update (tau=0.005)
        with torch.no_grad():
            for pt, po in zip(q_target.parameters(), q_net.parameters()):
                pt.data = 0.005 * po.data + 0.995 * pt.data

    path = [origin]; node, b = origin, budget; visited = {node}
    for _ in range(max_steps):
        if node == dest or b < 0: break
        acts = [a for a in successors.get(node, []) if a not in visited]
        if not acts: break
        action = max(acts, key=lambda a: theta[(node, b, a)])
        mean_t, _ = edges[(node, action)]; b -= round(mean_t)
        visited.add(action); path.append(action); node = action
    origin_acts = successors.get(origin, [])
    origin_probs_arr = _actor_probs(theta, origin, budget, origin_acts)
    origin_actor_probabilities = {int(a): float(p) for a, p in zip(origin_acts, origin_probs_arr)}

    # Pure actor-policy Monte Carlo. There is deliberately no evaluation fallback.
    def _run_deployment_eval(mode, threshold=None, log_first_n=0):
        dijkstra_next_hop = _precompute_dijkstra_next_hop(dest)
        dijkstra_reachable = set(dijkstra_next_hop.keys()) | {dest}
        policy_rng = np.random.default_rng(seed + 3000003)
        old_eval_env_rng = env.rng
        env.rng = np.random.default_rng(seed + 4000003)
        successes = 0
        reached_dest = 0
        budget_failures = 0
        no_action_failures = 0
        max_step_failures = 0
        dac_action_count = 0
        fallback_count = 0
        rollouts_using_fallback = 0
        fallback_attempt_counts = {
            'executed': 0,
            'next_hop_missing': 0,
            'next_hop_not_outgoing': 0,
            'next_hop_excluded_only_visited': 0,
            'node_cannot_reach_destination': 0,
            'no_legal_actions': 0,
        }
        action_count = 0
        rollout_logs = []
        try:
            for rollout_idx in range(mc_episodes):
                m_node, m_b, m_visited = origin, budget, {origin}
                used_fallback = False
                rollout_log = []
                termination_reason = 'max_steps'
                for step_idx in range(max_steps):
                    if m_node == dest:
                        reached_dest += 1
                        if m_b >= 0:
                            successes += 1
                            termination_reason = 'success'
                        else:
                            budget_failures += 1
                            termination_reason = 'destination_over_budget'
                        break
                    if m_b < 0:
                        budget_failures += 1
                        termination_reason = 'budget_negative'
                        break
                    outgoing = list(successors.get(m_node, []))
                    legal = [a for a in outgoing if a not in m_visited]
                    if not legal:
                        no_action_failures += 1
                        termination_reason = 'no_legal_actions'
                        if mode == 'q_fallback':
                            fallback_attempt_counts['no_legal_actions'] += 1
                        break

                    q_vals = np.asarray([_Q(m_node, m_b, a) for a in legal], dtype=np.float64)
                    order = np.argsort(q_vals)
                    qmax = float(q_vals[order[-1]])
                    q2 = float(q_vals[order[-2]]) if len(q_vals) >= 2 else qmax
                    q_gap = qmax - q2 if len(q_vals) >= 2 else float('inf')
                    selected_by = mode
                    trigger = False
                    intended = None
                    fallback_status = None
                    if mode == 'actor':
                        probs = _actor_probs(theta, m_node, m_b, legal)
                        intended = int(policy_rng.choice(legal, p=probs))
                        dac_action_count += 1
                    elif mode == 'q_greedy':
                        intended = int(legal[int(np.argmax(q_vals))])
                        dac_action_count += 1
                    elif mode == 'dijkstra':
                        fb = dijkstra_next_hop.get(m_node)
                        if fb in legal:
                            intended = int(fb)
                            fallback_count += 1
                            selected_by = 'dijkstra'
                            fallback_status = 'executed'
                        else:
                            fallback_status = 'next_hop_missing_or_illegal'
                            no_action_failures += 1
                            termination_reason = fallback_status
                            break
                    elif mode == 'q_fallback':
                        intended = int(legal[int(np.argmax(q_vals))])
                        trigger = bool(q_gap < float(threshold))
                        if trigger:
                            fb = dijkstra_next_hop.get(m_node)
                            if m_node not in dijkstra_reachable:
                                fallback_attempt_counts['node_cannot_reach_destination'] += 1
                                fallback_status = 'node_cannot_reach_destination'
                            elif fb is None:
                                fallback_attempt_counts['next_hop_missing'] += 1
                                fallback_status = 'next_hop_missing'
                            elif fb not in outgoing:
                                fallback_attempt_counts['next_hop_not_outgoing'] += 1
                                fallback_status = 'next_hop_not_outgoing'
                            elif fb not in legal:
                                fallback_attempt_counts['next_hop_excluded_only_visited'] += 1
                                fallback_status = 'next_hop_excluded_only_visited'
                            else:
                                intended = int(fb)
                                fallback_count += 1
                                used_fallback = True
                                selected_by = 'dijkstra_fallback'
                                fallback_status = 'executed'
                                fallback_attempt_counts['executed'] += 1
                        if selected_by == 'q_fallback':
                            selected_by = 'q_greedy'
                            dac_action_count += 1
                    else:
                        raise ValueError(f'unknown deployment mode: {mode}')

                    actual = env.sample_executed_action(m_node, intended)
                    travel_time = env.sample_travel_time(m_node, actual)
                    next_node = actual
                    if rollout_idx < log_first_n:
                        rollout_log.append({
                            'step': step_idx,
                            'current_node': int(m_node),
                            'remaining_budget': int(m_b),
                            'legal_actions': [int(a) for a in legal],
                            'q_values': {int(a): float(q) for a, q in zip(legal, q_vals)},
                            'q_gap': float(q_gap),
                            'trigger_result': bool(trigger),
                            'selected_intended_action': int(intended),
                            'selected_by': selected_by,
                            'fallback_status': fallback_status,
                            'executed_action_after_uncertainty': int(actual),
                            'sampled_travel_time': int(travel_time),
                            'next_node': int(next_node),
                        })
                    m_b -= travel_time
                    m_node = next_node
                    m_visited.add(next_node)
                    action_count += 1
                else:
                    max_step_failures += 1
                if used_fallback:
                    rollouts_using_fallback += 1
                if rollout_idx < log_first_n:
                    rollout_log.append({'termination_reason': termination_reason, 'final_node': int(m_node), 'remaining_budget': int(m_b)})
                    rollout_logs.append({'rollout': rollout_idx, 'steps': rollout_log})
        finally:
            env.rng = old_eval_env_rng
        total_actions = dac_action_count + fallback_count
        return {
            'mode': mode,
            'threshold': float(threshold) if threshold is not None else None,
            'mc': successes / mc_episodes if mc_episodes else float('nan'),
            'success_count': successes,
            'reached_destination_count': reached_dest,
            'budget_failures': budget_failures,
            'no_action_failures': no_action_failures,
            'max_step_failures': max_step_failures,
            'failed_trajectories': mc_episodes - successes,
            'action_count': action_count,
            'dac_action_count': dac_action_count,
            'fallback_count': fallback_count,
            'fallback_ratio': fallback_count / total_actions if total_actions else 0.0,
            'average_fallback_actions_per_rollout': fallback_count / mc_episodes if mc_episodes else 0.0,
            'rollouts_using_fallback': rollouts_using_fallback,
            'rollouts_using_fallback_ratio': rollouts_using_fallback / mc_episodes if mc_episodes else 0.0,
            'fallback_attempt_counts': fallback_attempt_counts,
            'policy_seed': seed + 3000003,
            'env_seed': seed + 4000003,
            'rollout_logs': rollout_logs,
        }

    def _deterministic_dijkstra_sanity():
        dijkstra_next_hop = _precompute_dijkstra_next_hop(dest)
        node, remaining = origin, budget
        visited = {node}
        path_det = [node]
        total_time = 0
        termination_reason = 'max_steps'
        for _ in range(max_steps):
            if node == dest:
                termination_reason = 'success' if remaining >= 0 else 'destination_over_budget'
                break
            if remaining < 0:
                termination_reason = 'budget_negative'
                break
            fb = dijkstra_next_hop.get(node)
            if fb is None:
                termination_reason = 'next_hop_missing'
                break
            if fb not in successors.get(node, []):
                termination_reason = 'next_hop_not_outgoing'
                break
            if fb in visited:
                termination_reason = 'next_hop_visited'
                break
            t = max(1, round(edges[(node, fb)][0]))
            total_time += t
            remaining -= t
            node = fb
            visited.add(node)
            path_det.append(node)
        return {
            'path': [int(x) for x in path_det],
            'reached_destination': bool(node == dest and remaining >= 0),
            'path_length': len(path_det),
            'total_deterministic_travel_time': int(total_time),
            'remaining_budget': int(remaining),
            'termination_reason': termination_reason,
        }

    mc_rng = np.random.default_rng(seed + 1000003)
    mc_successes = 0
    old_env_rng = env.rng
    env.rng = np.random.default_rng(seed + 2000003)
    try:
        for _ in range(mc_episodes):
            m_node, m_b, m_visited = origin, budget, {origin}
            for _ in range(max_steps):
                if m_node == dest:
                    mc_successes += int(m_b >= 0); break
                if m_b < 0: break
                m_acts = [a for a in successors.get(m_node, []) if a not in m_visited]
                if not m_acts: break
                m_probs = _actor_probs(theta, m_node, m_b, m_acts)
                intended = int(mc_rng.choice(m_acts, p=m_probs))
                actual = env.sample_executed_action(m_node, intended)
                m_b -= env.sample_travel_time(m_node, actual)
                m_node = actual; m_visited.add(actual)
    finally:
        env.rng = old_env_rng
    prob = mc_successes / mc_episodes if mc_episodes else float('nan')

    deployment_actor_eval = None
    deployment_pure_eval = None
    deployment_fallback_eval = None
    deployment_dijkstra_eval = None
    deployment_dijkstra_sanity = None
    fallback_threshold_sensitivity = []
    if evaluate_dijkstra_fallback:
        deployment_dijkstra_sanity = _deterministic_dijkstra_sanity() if deployment_audit_detail else None
        log_first_n = 10 if deployment_audit_detail else 0
        deployment_fallback_eval = _run_deployment_eval('q_fallback', threshold=fallback_q_gap_threshold, log_first_n=log_first_n)
        if deployment_audit_detail:
            deployment_actor_eval = _run_deployment_eval('actor')
            deployment_pure_eval = _run_deployment_eval('q_greedy')
            deployment_dijkstra_eval = _run_deployment_eval('dijkstra')
            for _thr in (1e-5, 1e-4, 2e-4, 5e-4, 1e-3, 1e-2):
                fallback_threshold_sensitivity.append(_run_deployment_eval('q_fallback', threshold=_thr))
        print('  [DAC Deployment] DAC + Dijkstra Safety Fallback')
        print(f'  [DAC Deployment] Threshold: {fallback_q_gap_threshold}')
        print('  [DAC Deployment] fallback ratio='
              f"{deployment_fallback_eval['fallback_ratio']:.6f} "
              f"({deployment_fallback_eval['fallback_count']}/"
              f"{deployment_fallback_eval['dac_action_count'] + deployment_fallback_eval['fallback_count']})")
    def _audit_q_confidence():
        decision_records = []
        first20_rollouts = []
        first_failed_decision = None
        audit_env_seed = seed + 4000003
        old_audit_env_rng = env.rng
        env.rng = np.random.default_rng(audit_env_seed)
        try:
            for rollout_idx in range(mc_episodes):
                m_node, m_b, m_visited = origin, budget, {origin}
                rollout_steps = []
                rollout_decisions = []
                reached = False
                for step_idx in range(max_steps):
                    if m_node == dest:
                        reached = m_b >= 0
                        break
                    if m_b < 0:
                        break
                    m_acts = [a for a in successors.get(m_node, []) if a not in m_visited]
                    if not m_acts:
                        break
                    q_vals = np.asarray([_Q(m_node, m_b, a) for a in m_acts], dtype=np.float64)
                    order = np.argsort(-q_vals)
                    best_idx = int(order[0])
                    second_idx = int(order[1]) if len(order) > 1 else None
                    chosen = int(m_acts[best_idx])
                    second = int(m_acts[second_idx]) if second_idx is not None else None
                    qmax = float(q_vals[best_idx])
                    q2 = float(q_vals[second_idx]) if second_idx is not None else None
                    gap = float(qmax - q2) if q2 is not None else 0.0
                    spread = float(q_vals.max() - q_vals.min())
                    actual = env.sample_executed_action(m_node, chosen)
                    travel_time = env.sample_travel_time(m_node, actual)
                    rec = {
                        'rollout': rollout_idx,
                        'step': step_idx,
                        'node': int(m_node),
                        'budget': int(m_b),
                        'candidate_actions': [int(a) for a in m_acts],
                        'q_values': {int(a): float(q) for a, q in zip(m_acts, q_vals)},
                        'chosen_action': chosen,
                        'second_best_action': second,
                        'qmax': qmax,
                        'q2': q2,
                        'gap': gap,
                        'spread': spread,
                        'travel_time': int(travel_time),
                        'actual_action': int(actual),
                        'execution_override': bool(actual != chosen),
                        'let_distance_chosen': int(let_to_go.get(chosen, 10**9)),
                        'let_distance_second': int(let_to_go.get(second, 10**9)) if second is not None else None,
                    }
                    decision_records.append(rec)
                    rollout_decisions.append(rec)
                    rollout_steps.append({
                        'step': step_idx,
                        'node': int(m_node),
                        'budget': int(m_b),
                        'chosen_action': chosen,
                        'second_action': second,
                        'gap': gap,
                        'travel_time': int(travel_time),
                        'actual_action': int(actual),
                        'execution_override': bool(actual != chosen),
                        'destination_reached_after_step': bool(actual == dest and m_b - travel_time >= 0),
                    })
                    m_b -= travel_time
                    m_node = actual
                    m_visited.add(actual)
                if rollout_idx < 20:
                    first20_rollouts.append({
                        'rollout': rollout_idx,
                        'steps': rollout_steps,
                        'reached_destination': bool(reached),
                    })
                if not reached and first_failed_decision is None and rollout_decisions:
                    first_failed_decision = dict(rollout_decisions[0])
        finally:
            env.rng = old_audit_env_rng

        gaps = np.asarray([r['gap'] for r in decision_records], dtype=np.float64)
        bins = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, float('inf')]
        labels = ['0-1e-5', '1e-5-1e-4', '1e-4-1e-3', '1e-3-1e-2', '1e-2-1e-1', '>1e-1']
        histogram = {}
        for lo, hi, label in zip(bins[:-1], bins[1:], labels):
            if hi == float('inf'):
                count = int(np.sum(gaps > lo))
            elif lo == 0.0:
                count = int(np.sum((gaps >= lo) & (gaps < hi)))
            else:
                count = int(np.sum((gaps >= lo) & (gaps < hi)))
            histogram[label] = count
        confidence = {}
        for threshold in [1e-5, 1e-4, 1e-3, 1e-2]:
            count = int(np.sum(gaps < threshold))
            confidence[f'gap<{threshold:g}'] = {
                'count': count,
                'percentage': count / len(gaps) if len(gaps) else 0.0,
            }

        counterfactual = None
        if first_failed_decision is not None:
            cf_node = first_failed_decision['node']
            cf_budget = first_failed_decision['budget']
            greedy_action = first_failed_decision['chosen_action']
            second_action = first_failed_decision['second_best_action']

            def _run_forced_once(first_action, sim_seed):
                old_cf_rng = env.rng
                env.rng = np.random.default_rng(sim_seed)
                try:
                    node_cf, b_cf, visited_cf = cf_node, cf_budget, {cf_node}
                    forced = False
                    for _ in range(max_steps):
                        if node_cf == dest:
                            return b_cf >= 0
                        if b_cf < 0:
                            return False
                        acts_cf = [a for a in successors.get(node_cf, []) if a not in visited_cf]
                        if not acts_cf:
                            return False
                        if not forced and first_action in acts_cf:
                            intended_cf = int(first_action)
                            forced = True
                        else:
                            q_cf = np.asarray([_Q(node_cf, b_cf, a) for a in acts_cf], dtype=np.float64)
                            intended_cf = int(acts_cf[int(np.argmax(q_cf))])
                        actual_cf = env.sample_executed_action(node_cf, intended_cf)
                        b_cf -= env.sample_travel_time(node_cf, actual_cf)
                        node_cf = actual_cf
                        visited_cf.add(actual_cf)
                    return False
                finally:
                    env.rng = old_cf_rng

            greedy_success = 0
            second_success = 0
            for sim in range(100):
                greedy_success += int(_run_forced_once(greedy_action, seed + 5000003 + sim))
                if second_action is not None:
                    second_success += int(_run_forced_once(second_action, seed + 5000003 + sim))
            counterfactual = {
                'state': first_failed_decision,
                'greedy_success': greedy_success,
                'second_best_success': second_success,
                'n_simulations': 100,
            }

        return {
            'decision_count': len(decision_records),
            'gap_stats': {
                'mean': float(np.mean(gaps)) if len(gaps) else float('nan'),
                'median': float(np.median(gaps)) if len(gaps) else float('nan'),
                'p95': float(np.percentile(gaps, 95)) if len(gaps) else float('nan'),
                'p99': float(np.percentile(gaps, 99)) if len(gaps) else float('nan'),
                'min': float(np.min(gaps)) if len(gaps) else float('nan'),
                'max': float(np.max(gaps)) if len(gaps) else float('nan'),
            },
            'histogram': histogram,
            'confidence_counts': confidence,
            'first20_rollouts': first20_rollouts,
            'first_failed_decision': first_failed_decision,
            'counterfactual': counterfactual,
            'decision_records': decision_records,
            'env_seed': audit_env_seed,
        }

    q_confidence_audit = _audit_q_confidence() if evaluate_q_confidence_audit else None
    # Build Q-table wrapper and populate cache for greedy path
    q_table = _DACQWrapper(q_net, let_to_go, B, device)
    for i, p_node in enumerate(path[:-1]):
        _approx_b = budget
        for _j in range(i):
            _mt_e = edges.get((path[_j], path[_j+1]), (1, 0))
            _approx_b -= round(_mt_e[0])
        _ = q_table.get((p_node, max(0, _approx_b), path[i+1]), 0.0)

    # Diagnostics
    n_params = sum(p.numel() for p in q_net.parameters())
    if q_count:
        q_mean = q_sum / q_count
        q_std = float(np.sqrt(max(0.0, q_sumsq / q_count - q_mean * q_mean)))
        q_min = q_min_seen
        q_max = q_max_seen
    else:
        q_mean = q_std = q_min = q_max = float('nan')

    reaches_dest = path[-1] == dest
    first_action = path[1] if len(path) > 1 else None

    replay_successful = sum(1 for x in replay_buffer if x[3] > 0)
    replay_terminal_failures = sum(1 for x in replay_buffer
                                   if x[3] <= 0 and (x[5] < 0 or not x[6]))
    finite_checks = np.asarray(critic_losses + [q_mean, q_std, q_min, q_max, tau], dtype=float)
    return {'prob': prob, 'mc_pure_actor': prob, 'mc_episodes': mc_episodes,
            'path': path, 'q_table': q_table,
            'n_params': n_params,
            'q_mean': q_mean, 'q_std': q_std, 'q_min': q_min, 'q_max': q_max,
            'reaches_dest': reaches_dest, 'first_action': first_action,
            'path_len': len(path),
            'training_success_count': training_successes,
            'training_success_ratio': training_successes / episodes if episodes else float('nan'),
            'first_successful_episode': first_successful_episode,
            'critic_loss_mean': float(np.mean(critic_losses)) if critic_losses else float('nan'),
            'critic_loss_std': float(np.std(critic_losses)) if critic_losses else float('nan'),
            'origin_actor_probabilities': origin_actor_probabilities,
            'origin_actor_evolution': origin_actor_evolution,
            'critic_action_gap_evolution': critic_action_gap_evolution,
            'actor_loss_mean': float(np.mean(actor_losses)) if actor_losses else float('nan'),
            'actor_loss_std': float(np.std(actor_losses)) if actor_losses else float('nan'),
            'theta_gradient_norm_mean': float(np.mean(theta_gradient_norms)) if theta_gradient_norms else float('nan'),
            'theta_gradient_norm_std': float(np.std(theta_gradient_norms)) if theta_gradient_norms else float('nan'),
            'actor_update_count': actor_update_count,
            'average_old_to_new_kl': float(np.mean(old_to_new_kls)) if old_to_new_kls else float('nan'),
            'average_rkl_projection_change': float(np.mean(rkl_projection_changes)) if rkl_projection_changes else float('nan'),
            'use_adaptive_tau': use_adaptive_tau,
            'fixed_tau': float(fixed_tau),
            'actor_entropy_checkpoints': actor_entropy_checkpoints,
            'tau_initial': float(tau_initial), 'tau_final': float(tau),
            'tau_min': float(tau_min), 'tau_max': float(tau_max),
            'tau_checkpoints': tau_checkpoints,
            'replay_initial_size': replay_initial_size,
            'replay_initial_success_transition_count': replay_initial_successes,
            'replay_initial_successful_transition_ratio': replay_initial_successes / replay_initial_size if replay_initial_size else 0.0,
            'dijkstra_expert_path': expert_replay_path,
            'normal_replay_size': len(replay_buffer),
            'persistent_expert_replay_size': len(expert_replay_buffer) if use_persistent_expert_replay else 0,
            'persistent_expert_success_transition_count': sum(1 for x in expert_replay_buffer if x[3] > 0) if use_persistent_expert_replay else 0,
            'expert_replay_fraction': float(expert_replay_fraction) if use_persistent_expert_replay else 0.0,
            'expert_samples_used': expert_samples_used,
            'success_transitions_retained_at_end': replay_successful + (sum(1 for x in expert_replay_buffer if x[3] > 0) if use_persistent_expert_replay else 0),
            'replay_buffer_size': len(replay_buffer),
            'replay_terminal_success_transition_count': replay_successful,
            'replay_terminal_failure_transition_count': replay_terminal_failures,
            'replay_successful_transition_ratio': replay_successful / len(replay_buffer) if replay_buffer else 0.0,
            'actor_state_source': actor_state_source,
            'evaluate_dijkstra_fallback': evaluate_dijkstra_fallback,
            'fallback_q_gap_threshold': float(fallback_q_gap_threshold),
            'deployment_name': 'DAC + Dijkstra Safety Fallback',
            'deployment_actor_eval': deployment_actor_eval,
            'deployment_pure_eval': deployment_pure_eval,
            'deployment_fallback_eval': deployment_fallback_eval,
            'deployment_dijkstra_eval': deployment_dijkstra_eval,
            'deployment_dijkstra_sanity': deployment_dijkstra_sanity,
            'fallback_threshold_sensitivity': fallback_threshold_sensitivity,
            'evaluate_q_confidence_audit': evaluate_q_confidence_audit,
            'q_confidence_audit': q_confidence_audit,
            'actor_learning_audit': actor_learning_audit,
            'unique_actor_updated_state_count': len(actor_updated_states),
            'unique_replay_sampled_state_count': len(replay_sampled_states),
            'actor_replay_state_overlap_count': len(actor_updated_states & replay_sampled_states),
            'actor_replay_state_overlap_ratio': (
                len(actor_updated_states & replay_sampled_states) / len(actor_updated_states)
                if actor_updated_states else 0.0
            ),
            'has_nan': bool(np.isnan(finite_checks).any()),
            'has_divergence': bool((~np.isfinite(finite_checks)).any() or np.max(np.abs(finite_checks), initial=0) > 1e6),
            'use_dijkstra_warm_start': use_dijkstra_warm_start,
            'use_dijkstra_replay_warm_start': use_dijkstra_replay_warm_start,
            'use_persistent_expert_replay': use_persistent_expert_replay}









