"""
segac_routing2.py - SEGAC GPG with one-time Dijkstra BC initialization.

RL phase remains the GPG-style loop:
  collect complete episodes -> replay sample -> recompute current log-prob
  -> cumulative importance ratio -> trajectory-level return update.

The Dijkstra expert data is used only before RL for behavior-cloning
initialization. It is not inserted into replay and is not used after RL starts.
"""

from collections import deque
from pathlib import Path
import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WITH_WIS = False

_NET_DIR = Path(__file__).parent.parent / 'Networks' / 'Networks' / 'Anaheim'
_TIME_STEP = 0.25
_net_df = __import__('pandas').read_csv(_NET_DIR / 'Anaheim_network.csv')
_nodes_set = set()
for _, _row in _net_df.iterrows():
    _nodes_set.add(int(_row['From']))
    _nodes_set.add(int(_row['To']))
_NUM_NODES = max(_nodes_set) + 1 if _nodes_set else 1


def _precompute_let_to_go(dest):
    """Compatibility helper for local checks; not used by SEGAC training."""
    reverse = {}
    for _, row in _net_df.iterrows():
        u, v = int(row['From']), int(row['To'])
        reverse.setdefault(v, []).append((u, max(1, round(float(row['Cost']) / _TIME_STEP))))
    dist = {dest: 0}
    pq = [(0, dest)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for pred, cost in reverse.get(u, []):
            nd = d + cost
            if nd < dist.get(pred, float('inf')):
                dist[pred] = nd
                heapq.heappush(pq, (nd, pred))
    return dist


class SEGACPolicyNetwork(nn.Module):
    """Reference-style Policy: dense MLP from state observation to all-node logits."""

    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.net = nn.Sequential(
            nn.Linear(num_nodes * 2 + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes))

    def forward(self, state):
        return self.net(state)


class _CriticBaseline(nn.Module):
    """Compatibility class; v4 RL uses the GPG.py default baseline 0.1."""

    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_nodes * 2 + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, state):
        return self.model(state).squeeze(-1)


SEGACValueNetwork = _CriticBaseline


def _dijkstra_path(edges, successors, origin, dest):
    dist = {origin: 0.0}
    prev = {}
    pq = [(0.0, origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        if u == dest:
            break
        for v in successors.get(u, []):
            mean_t, _ = edges[(u, v)]
            nd = d + max(1, round(mean_t))
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dest not in prev:
        return [], float('inf')
    path = [dest]
    n = dest
    while n in prev:
        n = prev[n]
        path.append(n)
    path.reverse()
    return path, dist[dest]


def run_segac_routing(env, budget: int,
                      episodes: int = 50000, lr_actor: float = 0.01,
                      lr_critic: float = 0.1, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    edges = env.edges
    successors = env.successors
    origin = env.origin
    dest = env.dest
    nodes = list(env.nodes)
    num_nodes = max(max(nodes, default=0), origin, dest, _NUM_NODES - 1) + 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pi_net = SEGACPolicyNetwork(num_nodes).to(device)
    pi_opt = optim.Adam(pi_net.parameters(), lr=lr_actor)

    replay_memory = deque(maxlen=10000)
    replay_batch_size = 64
    collect_batch_size = 64
    progress_interval = 1000
    max_steps = len(nodes) * 2

    ratio_vals = []
    cum_ratio_vals = []
    baseline_vals = []
    return_vals = []
    gpg_losses = []
    sampled_episode_count = 0
    success_count = 0
    failure_count = 0

    print('  [SEGAC-v4]')
    print('  One-time Dijkstra BC warm-start: Enabled')
    print('  Episode replay GPG: Enabled')
    print('  Replay capacity: 10000')
    print(f'  Replay batch size: {replay_batch_size}')
    print(f'  Collect batch size: {collect_batch_size}')
    print(f'  WITH_WIS: {WITH_WIS}')

    def _state_tensor(node, b):
        state = np.zeros(num_nodes * 2 + 1, dtype=np.float32)
        if 0 <= node < num_nodes:
            state[node] = 1.0
        if 0 <= dest < num_nodes:
            state[num_nodes + dest] = 1.0
        state[-1] = float(b) / max(float(budget), 1.0)
        return torch.tensor(state, device=device).unsqueeze(0)

    def _masked_logits(node, b, acts):
        logits_all = pi_net(_state_tensor(node, b)).squeeze(0)
        acts_t = torch.tensor(acts, device=device, dtype=torch.long)
        return logits_all[acts_t]

    def _action_distribution(node, b, acts):
        act_logits = _masked_logits(node, b, acts)
        probs = torch.softmax(act_logits, dim=0)
        return act_logits, probs

    def _sample_tt(u, v):
        mean_t, sigma = edges[(u, v)]
        if sigma <= 0:
            return float(mean_t)
        cv2 = (sigma / mean_t) ** 2
        mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        return float(max(rng.lognormal(mu_ln, sigma_ln), 0.01))

    def _current_log_prob(step):
        act_logits = _masked_logits(step['state'][0], step['state'][1], step['feasible_actions'])
        log_probs = act_logits - torch.logsumexp(act_logits, dim=0)
        return log_probs[step['feasible_actions'].index(step['action'])]

    def _policy_action(node, b, no_revisit=None):
        acts = successors.get(node, [])
        if no_revisit is not None:
            filtered = [a for a in acts if a not in no_revisit]
            if filtered:
                acts = filtered
        if not acts:
            return None
        with torch.no_grad():
            _, probs_t = _action_distribution(node, b, acts)
            return acts[int(torch.argmax(probs_t).item())]

    deployment_filter_stats = {
        'total_decisions': 0,
        'cycle_filtered_count': 0,
        'all_visited_fallback_count': 0,
    }

    def _deployment_policy_action(node, b, visited=None):
        acts = list(successors.get(node, []))
        if not acts:
            return None
        deployment_filter_stats['total_decisions'] += 1
        with torch.no_grad():
            _, probs_t = _action_distribution(node, b, acts)
            order = torch.argsort(probs_t, descending=True).detach().cpu().numpy().tolist()
        if visited is not None:
            visited_set = set(visited)
            for idx in order:
                action = acts[int(idx)]
                if action not in visited_set:
                    if int(idx) != int(order[0]):
                        deployment_filter_stats['cycle_filtered_count'] += 1
                    return action
            deployment_filter_stats['all_visited_fallback_count'] += 1
        return acts[int(order[0])]

    def _greedy_path():
        path = [origin]
        node, b = origin, budget
        visited = {node}
        for _ in range(max_steps):
            if node == dest or b < 0:
                break
            action = _policy_action(node, b, visited)
            if action is None:
                break
            mean_t, _ = edges[(node, action)]
            b -= max(1, round(mean_t))
            visited.add(action)
            path.append(action)
            node = action
        return path

    def _quick_mc(n_eval=50, seed_offset=9000):
        vals = []
        for i in range(n_eval):
            local_rng = np.random.default_rng(seed + seed_offset + i)
            node, b = origin, budget
            for _ in range(max_steps):
                if node == dest or b < 0:
                    break
                action = _policy_action(node, b, None)
                if action is None or (node, action) not in edges:
                    break
                mean_t, sigma = edges[(node, action)]
                if sigma <= 0:
                    travel_time = float(mean_t)
                else:
                    cv2 = (sigma / mean_t) ** 2
                    mu_ln = np.log(mean_t) - 0.5 * np.log(1.0 + cv2)
                    sigma_ln = np.sqrt(np.log(1.0 + cv2))
                    travel_time = float(max(local_rng.lognormal(mu_ln, sigma_ln), 0.01))
                b -= max(1, round(travel_time))
                node = action
            vals.append(float(node == dest and b >= 0))
        return float(np.mean(vals)) if vals else 0.0

    def _entropy_origin():
        acts = successors.get(origin, [])
        if len(acts) <= 1:
            return 0.0
        with torch.no_grad():
            _, probs_t = _action_distribution(origin, budget, acts)
            ent = -(probs_t * torch.log(probs_t + 1e-12)).sum()
            return float(ent.item())

    def _agreement(dijkstra_path):
        if len(dijkstra_path) < 2:
            return 0.0
        agree = 0
        total = 0
        b = budget
        for i in range(len(dijkstra_path) - 1):
            node = dijkstra_path[i]
            expert = dijkstra_path[i + 1]
            acts = successors.get(node, [])
            if len(acts) > 1:
                total += 1
                if _policy_action(node, b, None) == expert:
                    agree += 1
            mean_t, _ = edges[(node, expert)]
            b -= max(1, round(mean_t))
        return agree / max(total, 1)

    dijk_path, dijk_cost = _dijkstra_path(edges, successors, origin, dest)
    bc_before_path = _greedy_path()
    bc_before_mc = _quick_mc()

    expert_trajs = []
    if dijk_path:
        n_expert = 100
        min_b = int(dijk_cost)
        for t in range(n_expert):
            frac = t / max(n_expert - 1, 1)
            b = min_b + int(frac * max(budget + 10 - min_b, 0))
            traj = []
            node = origin
            for i in range(len(dijk_path) - 1):
                action = dijk_path[i + 1]
                acts = successors.get(node, [])
                if action not in acts:
                    break
                traj.append((node, max(0, b), action))
                mean_t, _ = edges[(node, action)]
                b -= max(1, round(mean_t))
                node = action
            if traj:
                expert_trajs.append(traj)

    bc_losses = []
    if expert_trajs:
        bc_opt = optim.Adam(pi_net.parameters(), lr=lr_actor)
        for _ in range(20):
            bc_opt.zero_grad()
            bc_loss = torch.tensor(0.0, device=device)
            n_bc = 0
            for traj in expert_trajs:
                for node, b, action in traj:
                    acts = successors.get(node, [])
                    if len(acts) <= 1 or action not in acts:
                        continue
                    act_logits = _masked_logits(node, b, acts)
                    log_probs = act_logits - torch.logsumexp(act_logits, dim=0)
                    bc_loss = bc_loss - log_probs[acts.index(action)]
                    n_bc += 1
            if n_bc == 0:
                continue
            bc_loss = bc_loss / n_bc
            bc_loss.backward()
            bc_opt.step()
            bc_losses.append(float(bc_loss.detach().item()))

    bc_after_path = _greedy_path()
    bc_after_mc = _quick_mc(seed_offset=9500)
    bc_agreement = _agreement(dijk_path)
    print(
        f'  BC init: experts={len(expert_trajs)} '
        f'mc {bc_before_mc:.4f}->{bc_after_mc:.4f} '
        f'agreement={bc_agreement:.3f}',
        flush=True,
    )

    collected_episodes = 0
    last_actor_loss = float('nan')
    while collected_episodes < episodes:
        batch_target = min(collect_batch_size, episodes - collected_episodes)
        for _ in range(batch_target):
            episode = []
            node, b = origin, budget
            visited = {node}
            total_travel_time = 0.0
            for _step in range(max_steps):
                if node == dest or b < 0:
                    break
                acts = [a for a in successors.get(node, []) if a not in visited]
                if not acts:
                    break
                with torch.no_grad():
                    _, probs_t = _action_distribution(node, b, acts)
                    probs = probs_t.cpu().numpy()
                idx = int(rng.choice(len(acts), p=probs))
                action = acts[idx]
                old_log_prob = float(np.log(max(float(probs[idx]), 1e-12)))
                travel_time = _sample_tt(node, action)
                b_next = b - max(1, round(travel_time))
                total_travel_time += travel_time
                episode.append({
                    'state': (node, b),
                    'action': action,
                    'feasible_actions': list(acts),
                    'old_log_prob': old_log_prob,
                    'next_state': (action, b_next),
                })
                visited.add(action)
                node, b = action, b_next

            collected_episodes += 1
            if not episode:
                failure_count += 1
                continue
            on_time = bool(node == dest and total_travel_time <= budget)
            reward = 1.0 if on_time else 0.0
            replay_memory.append({
                'steps': episode,
                'reward': reward,
                'travel_time': total_travel_time,
                'success': on_time,
            })
            success_count += int(on_time)
            failure_count += int(not on_time)

        if replay_memory:
            n_sample = min(replay_batch_size, len(replay_memory))
            sample_idx = rng.choice(len(replay_memory), size=n_sample, replace=False)
            sampled_episodes = [replay_memory[int(i)] for i in np.atleast_1d(sample_idx)]
            sampled_episode_count += len(sampled_episodes)

            pi_opt.zero_grad()
            policy_losses = []
            for ep_data in sampled_episodes:
                baseline = 0.1
                trajectory_return = float(ep_data['reward']) - baseline
                baseline_vals.append(baseline)
                return_vals.append(trajectory_return)
                cur_log_probs = []
                old_log_probs = []
                for step in ep_data['steps']:
                    cur_log_probs.append(_current_log_prob(step))
                    old_log_probs.append(step['old_log_prob'])
                if not cur_log_probs:
                    continue
                cur_log_probs_t = torch.stack(cur_log_probs).view(-1, 1)
                old_log_probs_t = torch.tensor(old_log_probs, device=device).view(-1, 1)
                ratio = torch.exp(cur_log_probs_t) / torch.exp(old_log_probs_t)
                rho = torch.cumprod(ratio, dim=0)
                if WITH_WIS:
                    rho = rho / rho.mean()
                R = torch.tensor(trajectory_return, device=device)
                policy_losses.append((-cur_log_probs_t * R.detach() * rho.detach()).sum().view(1, -1))
                with torch.no_grad():
                    ratio_vals.extend(ratio.view(-1).cpu().numpy().tolist())
                    cum_ratio_vals.extend(rho.view(-1).cpu().numpy().tolist())

            if policy_losses:
                policy_loss = torch.cat(policy_losses).mean()
                policy_loss.backward()
                pi_opt.step()
                last_actor_loss = float(policy_loss.item())
                gpg_losses.append(last_actor_loss)

        if collected_episodes % progress_interval == 0 or collected_episodes == episodes:
            recent_success = success_count / max(success_count + failure_count, 1)
            print(
                f'  [SEGAC-v4] ep {collected_episodes}/{episodes} '
                f'replay={len(replay_memory)} actor_loss={last_actor_loss:.4f} '
                f'baseline=0.1000 recent_success={recent_success:.4f}',
                flush=True,
            )

    path = _greedy_path()

    def segac_neural_policy(node, bud, visited=None):
        return _deployment_policy_action(node, bud, visited)

    segac_neural_policy.deployment_filter_stats = deployment_filter_stats

    v_pred = 0.1
    ratio_arr = np.array(ratio_vals if ratio_vals else [float('nan')], dtype=float)
    cum_ratio_arr = np.array(cum_ratio_vals if cum_ratio_vals else [float('nan')], dtype=float)
    baseline_arr = np.array(baseline_vals if baseline_vals else [float('nan')], dtype=float)
    return_arr = np.array(return_vals if return_vals else [float('nan')], dtype=float)
    gpg_arr = np.array(gpg_losses if gpg_losses else [float('nan')], dtype=float)

    return {
        'prob': v_pred,
        'path': path,
        'v_pred': v_pred,
        'n_params': sum(p.numel() for p in pi_net.parameters()),
        'reaches_dest': bool(path and path[-1] == dest),
        'policy_func': segac_neural_policy,
        'episode_replay_enabled': True,
        'replay_buffer_size': len(replay_memory),
        'sampled_episode_count': sampled_episode_count,
        'importance_ratio_mean': float(np.nanmean(ratio_arr)),
        'importance_ratio_std': float(np.nanstd(ratio_arr)),
        'importance_ratio_min': float(np.nanmin(ratio_arr)),
        'importance_ratio_max': float(np.nanmax(ratio_arr)),
        'cumulative_ratio_mean': float(np.nanmean(cum_ratio_arr)),
        'cumulative_ratio_std': float(np.nanstd(cum_ratio_arr)),
        'cumulative_ratio_min': float(np.nanmin(cum_ratio_arr)),
        'cumulative_ratio_max': float(np.nanmax(cum_ratio_arr)),
        'gpg_loss': float(np.nanmean(gpg_arr)),
        'gpg_loss_last': gpg_losses[-1] if gpg_losses else float('nan'),
        'baseline_mean': float(np.nanmean(baseline_arr)),
        'baseline_std': float(np.nanstd(baseline_arr)),
        'trajectory_return_mean': float(np.nanmean(return_arr)),
        'trajectory_return_std': float(np.nanstd(return_arr)),
        'baseline_floor': 0.1,
        'success_trajectory_count': success_count,
        'failure_trajectory_count': failure_count,
        'training_success_rate': success_count / max(success_count + failure_count, 1),
        'with_wis': WITH_WIS,
        'expert_traj_count': len(expert_trajs),
        'bc_loss': bc_losses[-1] if bc_losses else float('nan'),
        'bc_loss_start': bc_losses[0] if bc_losses else float('nan'),
        'bc_before_mc': bc_before_mc,
        'bc_after_mc': bc_after_mc,
        'bc_before_path': bc_before_path,
        'bc_after_path': bc_after_path,
        'bc_agreement': bc_agreement,
        'dijkstra_agreement_bc': bc_agreement,
        'dijkstra_agreement_rl': _agreement(dijk_path),
        'periodic_expert_bc': False,
        'return_positive_ratio': float(np.mean(return_arr > 0)) if return_arr.size else float('nan'),
        'return_zero_ratio': float(np.mean(return_arr == 0)) if return_arr.size else float('nan'),
        'return_negative_ratio': float(np.mean(return_arr < 0)) if return_arr.size else float('nan'),
        'nonzero_loss_trajectory_ratio': float(np.mean(np.abs(return_arr) > 1e-12)) if return_arr.size else float('nan'),
        'optimizer_step_count': len(gpg_losses),
        'gradient_norm_mean': float('nan'),
        'gradient_norm_last': float('nan'),
        'actor_drift_mean': float('nan'),
        'actor_drift_last': float('nan'),
        'entropy_origin': _entropy_origin(),
        'deployment_filter_stats': deployment_filter_stats,
        'deployment_filter_total_decisions': deployment_filter_stats['total_decisions'],
        'deployment_cycle_filtered_count': deployment_filter_stats['cycle_filtered_count'],
        'deployment_all_visited_fallback_count': deployment_filter_stats['all_visited_fallback_count'],
    }
