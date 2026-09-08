"""
eu_rac.py - Neural EU-RAC with a GCN policy actor and parameterized value function.

The implementation is network-agnostic. The environment object supplies the graph,
edge travel-time distributions, origin, destination, budget, and execution uncertainty.
"""

import heapq
import copy
import random
import hashlib
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _extract_graph_data(env):
    """Extract graph statistics from a routing environment."""
    mean_edges = {tuple(k): float(v[0]) for k, v in env.edges.items()}
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    successors = defaultdict(list)

    for u, v in mean_edges:
        out_degree[u] += 1
        in_degree[v] += 1
        successors[u].append(v)

    nodes = set(getattr(env, 'nodes', set()))
    for u, v in mean_edges:
        nodes.add(u)
        nodes.add(v)
    if not nodes:
        raise ValueError('Environment must provide at least one edge or node.')

    num_nodes = max(int(n) for n in nodes) + 1
    max_out_degree = max(out_degree.values()) if out_degree else 1
    max_in_degree = max(in_degree.values()) if in_degree else 1
    return {
        'mean_edges': mean_edges,
        'out_degree': out_degree,
        'in_degree': in_degree,
        'successors': successors,
        'num_nodes': num_nodes,
        'max_out_degree': max_out_degree,
        'max_in_degree': max_in_degree,
    }


def _build_normalized_adjacency(successors, num_nodes):
    """Build D^(-1/2) * (A + I) * D^(-1/2) for a directed graph."""
    rows, cols = [], []
    for u, vs in successors.items():
        for v in vs:
            rows.append(int(u))
            cols.append(int(v))
    for i in range(num_nodes):
        rows.append(i)
        cols.append(i)

    idx = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes))

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row_scale = deg_inv_sqrt[rows]
    col_scale = deg_inv_sqrt[cols]
    norm_vals = row_scale * col_scale
    return torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes))


def _precompute_let_to_go(dest, mean_edges):
    """Run Dijkstra from the destination backwards to all nodes."""
    rev = defaultdict(list)
    for (u, v), cost in mean_edges.items():
        rev.setdefault(v, []).append((u, max(1, round(cost))))
    dist = {dest: 0}
    pq = [(0, dest)]
    visited = set()
    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for pred, cost in rev.get(node, []):
            nd = d + cost
            if nd < dist.get(pred, float('inf')):
                dist[pred] = nd
                heapq.heappush(pq, (nd, pred))
    return dist

class ValueNetwork(nn.Module):
    def __init__(self, num_nodes, embed_dim=32, hidden_dim=64, max_out_degree=1):
        super().__init__()
        self.max_out_degree = max(max_out_degree, 1)
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        input_dim = embed_dim + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, node_ids, budget_remaining, let_to_go, budget_max, out_degrees):
        emb = self.node_embedding(node_ids)
        b_norm = (budget_remaining.float() / budget_max).unsqueeze(1)
        let_norm = (let_to_go.float() / budget_max).unsqueeze(1)
        slack_norm = ((budget_remaining.float() - let_to_go.float()) / budget_max).unsqueeze(1)
        deg_norm = (out_degrees.float() / self.max_out_degree).unsqueeze(1)
        x = torch.cat([emb, b_norm, let_norm, slack_norm, deg_norm], dim=1)
        return self.net(x).squeeze(-1)


#  GCN Graph Encoder 
class GCNEncoder(nn.Module):
    """2-layer GCN: node features to node embeddings (32-dim).
    Uses precomputed normalized adjacency (sparse)."""
    def __init__(self, node_feat_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = nn.Linear(node_feat_dim, hidden_dim, bias=True)
        self.conv2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x, adj_norm):
        # x: (num_nodes, node_feat_dim)
        h = torch.sparse.mm(adj_norm, x)
        h = self.conv1(h)
        h = F.relu(h)
        h = torch.sparse.mm(adj_norm, h)
        h = self.conv2(h)
        h = F.relu(h)
        return h  # (num_nodes, out_dim)


#  Action Scorer 
class ActionScorer(nn.Module):
    """Scores feasible actions from destination-conditioned features.
    Input: h_i(32) + h_j(32) + h_dest(32) + (h_j-h_dest)(32)
           + b_norm(1) + LET_j_norm(1) + slack_after_norm(1)
           + edge_mean_norm(1) + edge_std_norm(1) + is_uncertain(1) = 134
    Output: scalar logit per action."""
    def __init__(self, input_dim=134, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1))
        # Negative bias: non-Dijkstra logits stay < 0, CE pushes Dijkstra into [0,10] range
        nn.init.constant_(self.net[4].bias, -3.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)


#  Build static node features 
def _build_static_node_features(dest, budget_max, let_to_go, graph):
    """Build static node features for GCN input."""
    num_nodes = graph['num_nodes']
    out_degree = graph['out_degree']
    in_degree = graph['in_degree']
    max_out_degree = max(graph['max_out_degree'], 1)
    max_in_degree = max(graph['max_in_degree'], 1)
    feats = np.zeros((num_nodes, 4), dtype=np.float32)
    budget_scale = max(budget_max, 1)
    for i in range(num_nodes):
        feats[i, 0] = out_degree.get(i, 0) / max_out_degree
        feats[i, 1] = in_degree.get(i, 0) / max_in_degree
        feats[i, 2] = let_to_go.get(i, 0) / budget_scale
        feats[i, 3] = 1.0
    return feats


#  EURAC with GCN Actor 
class EURAC:
    def __init__(self, env, p_intended=0.8, lr_e=0.2, lr_d=0.2, lr_actor=1e-4,
                 entropy_coef=0.01, lambda_td=0.9,
                 value_lr=1e-3, warm_start_lr=0.1, kl_coef=0.1,
                 actor_grad_clip=5.0, device=None,
                 auto_checkpoint_dir=None, auto_checkpoint_interval=10000,
                 auto_resume=True, update_freq=5):
        self.env = env
        self.p_intended = p_intended
        self.lr_e = lr_e
        self.lr_d = lr_d
        self.lr_actor = lr_actor
        self.warm_start_lr = warm_start_lr
        self.entropy_coef = entropy_coef
        self.lambda_td = lambda_td
        self.kl_coef = kl_coef
        self.actor_grad_clip = actor_grad_clip
        self.update_freq = update_freq
        self.Q_e = defaultdict(float)
        self.Q_d = defaultdict(float)
        self.e_Qe = defaultdict(float)
        self.e_Qd = defaultdict(float)

        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._graph = _extract_graph_data(env)
        self._mean_edges = self._graph['mean_edges']
        self._out_degree = self._graph['out_degree']
        self._num_nodes = self._graph['num_nodes']
        self._max_out_degree = self._graph['max_out_degree']
        self._let_to_go = _precompute_let_to_go(env.dest, self._mean_edges)
        self._budget_max = env.budget

        self._static_node_feats = torch.tensor(
            _build_static_node_features(env.dest, env.budget, self._let_to_go, self._graph),
            device=self._device)
        self._adj_norm = _build_normalized_adjacency(
            self._graph['successors'], self._num_nodes).to(self._device)

        self.gcn = GCNEncoder(node_feat_dim=4, hidden_dim=64, out_dim=32).to(self._device)
        self.action_scorer = ActionScorer(input_dim=134, hidden_dim=64).to(self._device)
        # During warm-start: optimize GCN + action_scorer together
        # After warm-start (RL): freeze GCN, only optimize action_scorer
        self.pi_optimizer = optim.Adam(
            list(self.gcn.parameters()) + list(self.action_scorer.parameters()),
            lr=warm_start_lr)
        self._gcn_frozen = False
        self._cached_embs = None  # frozen GCN embeddings
        self.ref_scorer = None
        self._ref_node_embs = None
        self._warm_actor_state = None

        #  Edge features for fast lookup (static per run) 
        self._edge_cache = {}

        #  V_phi (same as eu_rac2) 
        self.v_net = ValueNetwork(self._num_nodes, max_out_degree=self._max_out_degree).to(self._device)
        self.v_target = ValueNetwork(self._num_nodes, max_out_degree=self._max_out_degree).to(self._device)
        self.v_target.load_state_dict(self.v_net.state_dict())
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=value_lr)
        self._v_cache = {}
        self._v_target_cache = {}

        #  Diagnosis tracking (rolling windows via deque, maxlen = cap) 
        self._diag_advantages = deque(maxlen=1000)
        self._diag_entropies = deque(maxlen=1000)
        self._diag_max_probs = deque(maxlen=1000)
        self._diag_dijkstra_agreement_before = None
        self._diag_dijkstra_agreement_after = None
        self._diag_logit_means = deque(maxlen=500)
        self._diag_logit_stds = deque(maxlen=500)
        self._diag_raw_logit_means = deque(maxlen=500)   # pre-clamp
        self._diag_raw_logit_stds = deque(maxlen=500)    # pre-clamp
        self._diag_kl_ref_cur = deque(maxlen=1000)
        self._diag_kl_cur_ref = deque(maxlen=1000)
        self._diag_actor_losses = deque(maxlen=1000)
        self._diag_actor_updates = 0

        self._episodes_trained = 0
        self._auto_checkpoint_interval = auto_checkpoint_interval
        self._auto_checkpoint_path = self._make_auto_checkpoint_path(auto_checkpoint_dir)
        self._resumed_from_checkpoint = False
        if auto_resume and self._auto_checkpoint_path.exists():
            loaded_ep = self.load_checkpoint(self._auto_checkpoint_path)
            self._resumed_from_checkpoint = loaded_ep > 0
            if self._resumed_from_checkpoint:
                print(f'    [EU-RAC] auto-resumed {loaded_ep} episodes from '
                      f'{self._auto_checkpoint_path}')

    def _make_auto_checkpoint_path(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parent / 'checkpoints' / 'eurac_autocheckpoints'
        else:
            checkpoint_dir = Path(checkpoint_dir)
        uncertain_items = sorted(
            (int(k), int(v['intended']) if isinstance(v, dict) else int(v))
            for k, v in getattr(self.env, 'uncertain_edges', {}).items()
        )
        unc_sig = hashlib.sha1(repr(uncertain_items).encode('utf-8')).hexdigest()[:10]
        p_tag = int(round(float(self.p_intended) * 1000))
        env_p = int(round(float(getattr(self.env, 'exec_prob', self.p_intended)) * 1000))
        name = (f'eurac_od{self.env.origin}_{self.env.dest}_b{self.env.budget}'
                f'_p{p_tag}_envp{env_p}_u{unc_sig}_latest.pt')
        return checkpoint_dir / name

    def _finish_episode(self, outcome, train):
        if train:
            self._episodes_trained += 1
            interval = self._auto_checkpoint_interval
            if interval and self._episodes_trained % interval == 0:
                self.save_checkpoint(self._auto_checkpoint_path, episode=self._episodes_trained, n_episodes=None)
                print(f'    [EU-RAC] auto-checkpoint saved at {self._episodes_trained} episodes: {self._auto_checkpoint_path}')
        return outcome
    #  GCN embedding helper 
    def _get_node_embeddings(self):
        """GCN output with shape (num_nodes, 32). Cached when frozen (RL phase)."""
        if self._gcn_frozen and self._cached_embs is not None:
            return self._cached_embs
        with torch.set_grad_enabled(not self._gcn_frozen):
            embs = self.gcn(self._static_node_feats, self._adj_norm)
        if self._gcn_frozen:
            self._cached_embs = embs.detach()
        return embs

    #  Edge feature helper 
    def _get_edge_features(self, node, action):
        """Return (mean_norm, std_norm, is_uncertain) for an edge."""
        k = (node, action)
        if k in self._edge_cache:
            return self._edge_cache[k]
        mean_t, sigma = self.env.edges.get(k, (1.0, 0.0))
        B = max(self._budget_max, 1)
        mean_n = mean_t / B
        std_n = sigma / B if sigma > 0 else 0.0
        is_unc = 1.0 if node in self.env.uncertain_edges else 0.0
        self._edge_cache[k] = (mean_n, std_n, is_unc)
        return self._edge_cache[k]
    def _action_logits(self, node, budget, acts, node_embs):
        """Compute RAW logits for feasible actions. Returns tensor (len(acts),).
        Features include destination conditioning via h_dest and per-candidate LET."""
        if not acts:
            return torch.zeros(0, device=self._device)

        h_i = node_embs[node]  # (32,)
        acts_t = torch.tensor(acts, device=self._device)
        h_js = node_embs[acts_t]  # (na, 32)
        h_i_exp = h_i.unsqueeze(0).expand(len(acts), -1)  # (na, 32)

        # Destination conditioning
        h_dest = node_embs[self.env.dest]  # (32,)
        h_dest_exp = h_dest.unsqueeze(0).expand(len(acts), -1)  # (na, 32)
        h_diff = h_js - h_dest_exp  # (na, 32), candidate offset from the destination embedding.

        B = max(self._budget_max, 1)
        b_n = torch.full((len(acts), 1), budget / B, device=self._device)

        # LET from each candidate j to destination (not current node i)
        let_js = torch.tensor([self._let_to_go.get(a, 0) for a in acts],
                              device=self._device).unsqueeze(1) / B  # (na, 1)

        edge_feats = [self._get_edge_features(node, a) for a in acts]
        et = torch.tensor([f[0] for f in edge_feats], device=self._device).unsqueeze(1)
        es = torch.tensor([f[1] for f in edge_feats], device=self._device).unsqueeze(1)
        uf = torch.tensor([f[2] for f in edge_feats], device=self._device).unsqueeze(1)

        # slack after taking this action: budget - mean_t - LET(j, dest)
        slack_after = b_n - et - let_js  # (na, 1)

        # Input: h_i(32) + h_j(32) + h_dest(32) + h_diff(32)
        #        + b_n(1) + LET_j_norm(1) + slack_after(1)
        #        + edge_mean(1) + edge_std(1) + is_uncertain(1) = 134
        x = torch.cat([h_i_exp, h_js, h_dest_exp, h_diff,
                       b_n, let_js, slack_after, et, es, uf], dim=1)
        return self.action_scorer(x)

    def _action_logits_clamped(self, node, budget, acts, node_embs):
        """Compute squashed logits for policy softmax. Returns (raw, squashed).
        tanh preserves ordering and maps large raw values smoothly to [-10,10]."""
        raw = self._action_logits(node, budget, acts, node_embs)
        squashed = 10.0 * torch.tanh(raw / 10.0)
        return raw, squashed

    def _ref_action_logits(self, node, budget, acts):
        """Compute reference (warm-start) policy logits using frozen scorer."""
        if not acts:
            return torch.zeros(0, device=self._device), torch.zeros(0, device=self._device)
        # Re-use _action_logits feature computation but with ref_scorer
        h_i = self._ref_node_embs[node]  # (32,)
        acts_t = torch.tensor(acts, device=self._device)
        h_js = self._ref_node_embs[acts_t]  # (na, 32)
        h_i_exp = h_i.unsqueeze(0).expand(len(acts), -1)  # (na, 32)

        h_dest = self._ref_node_embs[self.env.dest]  # (32,)
        h_dest_exp = h_dest.unsqueeze(0).expand(len(acts), -1)  # (na, 32)
        h_diff = h_js - h_dest_exp  # (na, 32)

        B = max(self._budget_max, 1)
        b_n = torch.full((len(acts), 1), budget / B, device=self._device)
        let_js = torch.tensor([self._let_to_go.get(a, 0) for a in acts],
                              device=self._device).unsqueeze(1) / B
        edge_feats = [self._get_edge_features(node, a) for a in acts]
        et = torch.tensor([f[0] for f in edge_feats], device=self._device).unsqueeze(1)
        es = torch.tensor([f[1] for f in edge_feats], device=self._device).unsqueeze(1)
        uf = torch.tensor([f[2] for f in edge_feats], device=self._device).unsqueeze(1)
        slack_after = b_n - et - let_js
        x = torch.cat([h_i_exp, h_js, h_dest_exp, h_diff,
                       b_n, let_js, slack_after, et, es, uf], dim=1)
        raw = self.ref_scorer(x)
        squashed = 10.0 * torch.tanh(raw / 10.0)
        return raw, squashed

    #  Neural policy probabilities 
    def _neural_probs(self, node, budget, acts, node_embs):
        """Return (acts_list, probs_np) under neural policy (clamped logits to softmax)."""
        na = len(acts)
        if na == 0:
            return [], np.array([])
        if na == 1:
            return acts, np.array([1.0])
        raw, clamped = self._action_logits_clamped(node, budget, acts, node_embs)

        # Track raw + clamped statistics from same forward pass (rolling)
        self._diag_raw_logit_means.append(float(raw.mean().item()))
        self._diag_raw_logit_stds.append(float(raw.std().item()))
        self._diag_logit_means.append(float(clamped.mean().item()))
        self._diag_logit_stds.append(float(clamped.std().item()))

        probs = F.softmax(clamped, dim=0)
        return acts, probs.detach().cpu().numpy()

    def action_probs(self, state, feasible_actions=None):
        """Torch action probabilities for a state and feasible action list."""
        node, budget = state
        acts = feasible_actions if feasible_actions is not None else self.env.get_actions(node)
        if len(acts) == 0:
            return torch.zeros(0, device=self._device)
        if len(acts) == 1:
            return torch.ones(1, device=self._device)
        node_embs = self._get_node_embeddings()
        _, clamped = self._action_logits_clamped(node, budget, acts, node_embs)
        return F.softmax(clamped, dim=0)

    def _reference_probs(self, state, feasible_actions):
        """Frozen warm-start reference probabilities for the same feasible action mask."""
        if self.ref_scorer is None or self._ref_node_embs is None:
            return None
        if len(feasible_actions) == 0:
            return torch.zeros(0, device=self._device)
        if len(feasible_actions) == 1:
            return torch.ones(1, device=self._device)
        node, budget = state
        _, ref_clamped = self._ref_action_logits(node, budget, feasible_actions)
        return F.softmax(ref_clamped, dim=0)

    def _freeze_gcn_and_anchor(self):
        """Freeze the warm-start policy as the reference for conservative RL updates."""
        self._edge_cache.clear()
        self._gcn_frozen = True
        for p in self.gcn.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            self._cached_embs = self.gcn(self._static_node_feats, self._adj_norm).detach()
        self.pi_optimizer = optim.Adam(self.action_scorer.parameters(), lr=self.lr_actor)

        self.ref_scorer = copy.deepcopy(self.action_scorer)
        self.ref_scorer.eval()
        for p in self.ref_scorer.parameters():
            p.requires_grad_(False)
        self._ref_node_embs = self._cached_embs.clone()
        self._warm_actor_state = {
            k: v.detach().clone() for k, v in self.action_scorer.state_dict().items()
        }

    #  Warm-start (Dijkstra + cross-entropy for neural actor) 
    def warm_start(self, logit=4.0):
        if self._resumed_from_checkpoint:
            print('    [GCN warm-start] skipped because EURAC resumed from checkpoint')
            return
        edges = self.env.edges
        successors = self.env.successors
        origin = self.env.origin; dest = self.env.dest

        #  Step 1: Original Dijkstra warm-start (build tabular theta) 
        dist = {origin: 0.0}; prev = {}; pq = [(0.0, origin)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            for v in successors.get(u, []):
                mean_t, _ = edges[(u, v)]
                nd = d + max(1, round(mean_t))
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if dest not in prev:
            return

        path = []; n = dest
        while n in prev: path.append(n); n = prev[n]
        path.append(origin); path.reverse()

        budget_slack = 10; remaining = self.env.budget
        warm_start_pairs = []  # (node, budget, action)
        for i in range(len(path) - 1):
            node = path[i]; action = path[i + 1]
            max_b = remaining + budget_slack
            for b in range(0, max_b + 1):
                if b >= 0:
                    warm_start_pairs.append((node, b, action))
            mean_t, _ = edges[(node, action)]
            remaining -= max(1, round(mean_t))

        #  Step 2: Cross-entropy warm-start for neural actor 
        print(f'    [GCN warm-start] {len(warm_start_pairs)} (state, action) pairs, '
              f'{len(set(p[0] for p in warm_start_pairs))} unique nodes')

        # Group by node for efficiency
        from collections import defaultdict as _dd
        by_node = _dd(list)
        for node, b, action in warm_start_pairs:
            by_node[node].append((b, action))

        # Fast Dijkstra agreement checker (using current neural policy)
        def _fast_agree(node_embs_local):
            agree, total = 0, 0
            dijk_budget = self.env.budget
            dijk_node = origin
            dijk_visited = {dijk_node}
            for i in range(len(path) - 1):
                dijk_action = path[i + 1]
                total += 1
                acts = successors.get(dijk_node, [])
                if len(acts) <= 1:
                    dijk_node = dijk_action; continue
                _, probs = self._neural_probs(dijk_node, dijk_budget, acts, node_embs_local)
                my_action = acts[int(np.argmax(probs))]
                if my_action == dijk_action: agree += 1
                mean_t, _ = edges[(dijk_node, dijk_action)]
                dijk_budget -= max(1, round(mean_t))
                dijk_node = dijk_action
            return agree / max(total, 1) if total > 0 else 0.0

        # Pre-warm-start agreement
        with torch.no_grad():
            embs0 = self._get_node_embeddings()
            agree0 = _fast_agree(embs0)
        self._diag_dijkstra_agreement_before = float(agree0)
        print(f'    [GCN warm-start] agreement before CE: {agree0:.3f}')

        # Helper: compute greedy path under current neural policy
        def _greedy_path(node_embs_local):
            gp = [origin]; n = origin; b = self.env.budget; vis = {n}
            for _ in range(len(self.env.nodes) * 2):
                if n == dest or b < 0: break
                acts = [a for a in successors.get(n, []) if a not in vis]
                if not acts: break
                _, probs = self._neural_probs(n, b, acts, node_embs_local)
                a = acts[int(np.argmax(probs))]
                mean_t, _ = edges[(n, a)]; b -= max(1, round(mean_t))
                vis.add(a); gp.append(a); n = a
            return gp, gp[-1] == dest

        #  CE + margin training with early stopping 
        max_epochs = 80; check_interval = 5
        best_state = None; best_score = -1
        stopped_early = False; stopped_epoch = max_epochs
        ce_loss_val = float('inf'); margin_val = float('inf')
        lambda_m = 0.5; m_margin = 1.0

        for epoch in range(max_epochs):
            self.pi_optimizer.zero_grad()
            node_embs = self._get_node_embeddings()

            # --- CE loss (path_weight=5.0 for Dijkstra-path samples) ---
            path_nodes = set(path)  # nodes on Dijkstra path
            ce_loss = torch.tensor(0.0, device=self._device); n_ce = 0
            for node, pairs in by_node.items():
                all_acts = successors.get(node, [])
                if len(all_acts) <= 1: continue
                w = 5.0 if node in path_nodes else 1.0
                for b, action in pairs:
                    _, logits = self._action_logits_clamped(node, b, all_acts, node_embs)
                    log_probs = logits - torch.logsumexp(logits, dim=0)
                    try:
                        idx = all_acts.index(action)
                    except ValueError:
                        continue
                    ce_loss = ce_loss - w * log_probs[idx]
                    n_ce += w
            ce_loss = ce_loss / max(n_ce, 1)

            # --- Expert-action margin loss (path_weight=5.0, on raw logits) ---
            margin_loss = torch.tensor(0.0, device=self._device); n_mg = 0
            for node, pairs in by_node.items():
                all_acts = successors.get(node, [])
                if len(all_acts) <= 1: continue
                w = 5.0 if node in path_nodes else 1.0
                for b, action in pairs:
                    raw = self._action_logits(node, b, all_acts, node_embs)
                    try:
                        idx_star = all_acts.index(action)
                    except ValueError:
                        continue
                    z_star = raw[idx_star]
                    mask = torch.ones(len(all_acts), device=self._device)
                    mask[idx_star] = 0.0
                    z_other_max = (raw + (1.0 - mask) * (-1e9)).max()
                    margin_loss = margin_loss + w * torch.clamp(m_margin + z_other_max - z_star, min=0)
                    n_mg += w
            margin_loss = margin_loss / max(n_mg, 1)

            # --- L2 penalty ---
            l2_penalty = 0.0
            for node in by_node:
                all_acts = successors.get(node, [])
                if len(all_acts) <= 1: continue
                l = self._action_logits(node, by_node[node][0][0], all_acts, node_embs)
                l2_penalty = l2_penalty + (l ** 2).mean()
            l2_penalty = l2_penalty / max(len(by_node), 1)

            total_loss = ce_loss + lambda_m * margin_loss + 0.0005 * l2_penalty
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.gcn.parameters()) + list(self.action_scorer.parameters()), 5.0)
            self.pi_optimizer.step()
            ce_loss_val = ce_loss.item(); margin_val = margin_loss.item()

            # Periodic check: agreement + greedy path + divergence analysis
            if (epoch + 1) % check_interval == 0:
                with torch.no_grad():
                    embs_e = self._get_node_embeddings()
                    agree_e = _fast_agree(embs_e)
                    gp, gp_reaches = _greedy_path(embs_e)

                    # Find first divergence: walk both paths with same budget
                    div_node, div_budget, div_dijk, div_greedy = None, None, None, None
                    b_dijk = self.env.budget; b_greedy = self.env.budget
                    n_dijk = origin; n_greedy = origin
                    for step in range(min(len(path), len(gp)) - 1):
                        a_dijk = path[step + 1]
                        a_greedy = gp[step + 1] if step + 1 < len(gp) else None
                        if a_dijk != a_greedy:
                            div_node = n_dijk; div_budget = b_dijk
                            div_dijk = a_dijk; div_greedy = a_greedy
                            break
                        mean_t_d, _ = edges[(n_dijk, a_dijk)]
                        b_dijk -= max(1, round(mean_t_d)); n_dijk = a_dijk
                        if a_greedy:
                            mean_t_g, _ = edges[(n_greedy, a_greedy)]
                            b_greedy -= max(1, round(mean_t_g)); n_greedy = a_greedy

                    # Divergence diagnostics
                    if div_node is not None:
                        div_acts = successors.get(div_node, [])
                        if len(div_acts) > 1:
                            raw_d, sq_d = self._action_logits_clamped(div_node, div_budget, div_acts, embs_e)
                            probs_d = F.softmax(sq_d, dim=0)
                            div_raw_str = ", ".join(f"{a}:{raw_d[i].item():.2f}" for i, a in enumerate(div_acts))
                            div_prob_str = ", ".join(f"{a}:{probs_d[i].item():.3f}" for i, a in enumerate(div_acts))
                            z_star = raw_d[div_acts.index(div_dijk)].item() if div_dijk in div_acts else float('nan')
                            z_other = max([raw_d[i].item() for i, a in enumerate(div_acts) if a != div_dijk]) if div_dijk in div_acts else float('nan')
                            div_margin = z_star - z_other
                        else:
                            div_raw_str = "N/A"; div_prob_str = "N/A"; div_margin = float('nan')
                    else:
                        div_raw_str = "N/A"; div_prob_str = "N/A"; div_margin = float('nan')

                # Score: reaches_dest is top priority
                score = (2.0 if gp_reaches else 0.0) + agree_e
                if score > best_score:
                    best_score = score
                    best_state = {
                        'gcn': {k: v.clone() for k, v in self.gcn.state_dict().items()},
                        'scorer': {k: v.clone() for k, v in self.action_scorer.state_dict().items()},
                        'epoch': epoch + 1, 'agree': agree_e, 'reaches': gp_reaches,
                        'loss': ce_loss_val, 'margin': div_margin, 'path': gp,
                    }

                print(f'    [GCN warm-start] epoch {epoch+1}/{max_epochs}, '
                      f'CE={ce_loss_val:.4f}, agree={agree_e:.3f}, gp_reaches={gp_reaches}, gp_len={len(gp)}')
                print(f'      Dijkstra path: {"->".join(str(n) for n in path)}')
                print(f'      Greedy path:   {"->".join(str(n) for n in gp)}')
                if div_node is not None:
                    print(f'      First divergence at node={div_node}, budget={div_budget}: '
                          f'Dijkstra->{div_dijk}, greedy->{div_greedy}')
                    print(f'      Divergence raw logits: [{div_raw_str}]')
                    print(f'      Divergence probs:      [{div_prob_str}]')
                    print(f'      Expert action margin: {div_margin:.2f}')

                if gp_reaches:
                    stopped_early = True; stopped_epoch = epoch + 1
                    print(f'    [GCN warm-start] Early stop: greedy path reaches destination!')
                    break

        #  Restore best checkpoint 
        if best_state is not None:
            self.gcn.load_state_dict(best_state['gcn'])
            self.action_scorer.load_state_dict(best_state['scorer'])
            best_epoch = best_state['epoch']
            best_agree = best_state['agree']
            best_reaches = best_state['reaches']
            best_path = best_state['path']
            ce_loss_val = best_state['loss']
            print(f'    [GCN warm-start] Restored best checkpoint (epoch {best_epoch}): '
                  f'agree={best_agree:.3f}, reaches_dest={best_reaches}')
        else:
            best_epoch = max_epochs; best_agree = agree0; best_reaches = False; best_path = [origin]

        # Clear edge cache and freeze GCN after warm-start
        self._freeze_gcn_and_anchor()
        print(f'    [GCN] Frozen. Cached embeddings: '
              f'mu={self._cached_embs.mean().item():.4f} '
              f'sig={self._cached_embs.std().item():.4f}')

        #  Anchor: frozen reference actor copy 
        print(f'    [Anchor] Reference actor frozen. KL weight={self.kl_coef:g}, '
              f'RL actor lr={self.lr_actor:g}')

        #  Warm-start post-mortem (using restored best model) 
        with torch.no_grad():
            embs_final = self._get_node_embeddings()
            agree_final = _fast_agree(embs_final)
            self._diag_dijkstra_agreement_after = float(agree_final)
            ws_path, ws_reaches = _greedy_path(embs_final)

            # Policy stats along Dijkstra path
            entropies, max_probs_ws = [], []
            ws_b2 = self.env.budget; ws_n2 = origin
            for i in range(len(path) - 1):
                ws_acts2 = successors.get(ws_n2, [])
                if len(ws_acts2) > 1:
                    _, ws_p = self._neural_probs(ws_n2, ws_b2, ws_acts2, embs_final)
                    ent = -sum(p * np.log(p + 1e-8) for p in ws_p)
                    entropies.append(ent); max_probs_ws.append(float(np.max(ws_p)))
                mean_t, _ = edges[(ws_n2, path[i+1])]
                ws_b2 -= max(1, round(mean_t)); ws_n2 = path[i+1]

        print(f'    [GCN warm-start] Final (best epoch {best_epoch}): '
              f'agreement={agree_final:.3f}, CE loss={ce_loss_val:.4f}')
        if entropies:
            print(f'    [GCN warm-start] Policy entropy: {np.mean(entropies):.4f}  '
                  f'max prob: {np.mean(max_probs_ws):.4f}')
        print(f'    [GCN warm-start] Greedy path reaches dest: {ws_reaches}  '
              f'(len={len(ws_path)}): {"->".join(str(n) for n in ws_path[:15])}'
              + ('...' if len(ws_path) > 15 else ''))

    #  Policy helpers 
    def get_policy(self, state):
        node, budget = state
        acts = self.env.get_actions(node)
        if not acts:
            return {}, []
        with torch.no_grad():
            node_embs = self._get_node_embeddings()
            _, probs = self._neural_probs(node, budget, acts, node_embs)
        return dict(zip(acts, probs)), acts

    def select_action(self, state, exclude=None):
        node, budget = state
        acts = self.env.get_actions(node)
        if not acts:
            return None
        if exclude:
            filtered = [a for a in acts if a not in exclude]
            if filtered:
                acts = filtered
            elif len(acts) == 1:
                return acts[0]
        with torch.no_grad():
            node_embs = self._get_node_embeddings()
            _, probs = self._neural_probs(node, budget, acts, node_embs)
        return int(np.random.choice(acts, p=probs))

    #  Value functions (same as eu_rac2) 
    def _compute_V_pi(self, state):
        node, budget = state
        if node == self.env.dest:
            return 1.0 if budget >= 0 else 0.0
        if budget < 0:
            return 0.0
        policy, actions = self.get_policy(state)
        if not actions:
            return 0.0
        return sum(policy[a] * self.Q_d[(state, a)] for a in actions)

    def _compute_V_phi(self, state, use_cache=True):
        node, budget = state
        if node == self.env.dest:
            return 1.0 if budget >= 0 else 0.0
        if budget < 0:
            return 0.0
        key = (node, budget)
        if use_cache and key in self._v_cache:
            return self._v_cache[key]
        with torch.no_grad():
            n = torch.tensor([node], device=self._device)
            b = torch.tensor([budget], device=self._device)
            let = torch.tensor([self._let_to_go.get(node, 0)], device=self._device)
            deg = torch.tensor([self._out_degree.get(node, 0)], device=self._device)
            v = float(self.v_net(n, b, let, self._budget_max, deg).item())
        if use_cache:
            self._v_cache[key] = v
        return v

    def _compute_V_target(self, state):
        node, budget = state
        if node == self.env.dest:
            return 1.0 if budget >= 0 else 0.0
        if budget < 0:
            return 0.0
        key = (node, budget)
        if key in self._v_target_cache:
            return self._v_target_cache[key]
        with torch.no_grad():
            n = torch.tensor([node], device=self._device)
            b = torch.tensor([budget], device=self._device)
            let = torch.tensor([self._let_to_go.get(node, 0)], device=self._device)
            deg = torch.tensor([self._out_degree.get(node, 0)], device=self._device)
            v = float(self.v_target(n, b, let, self._budget_max, deg).item())
        self._v_target_cache[key] = v
        return v

    def compute_V(self, state):
        return self._compute_V_phi(state)

    #  V training (same as eu_rac2) 
    def _train_V_step(self, state, y_e_stopgrad):
        node, budget = state
        n = torch.tensor([node], device=self._device)
        b = torch.tensor([budget], device=self._device)
        let = torch.tensor([self._let_to_go.get(node, 0)], device=self._device)
        deg = torch.tensor([self._out_degree.get(node, 0)], device=self._device)

        V_phi_val = self.v_net(n, b, let, self._budget_max, deg)
        v_Q = self._compute_V_pi(state)

        target1 = torch.tensor([y_e_stopgrad], device=self._device)
        target2 = torch.tensor([v_Q], device=self._device)
        loss = (V_phi_val - target1).pow(2).mean() + \
               0.1 * (V_phi_val - target2).pow(2).mean()

        self.v_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.v_net.parameters(), 5.0)
        self.v_optimizer.step()

        with torch.no_grad():
            for p_tgt, p_online in zip(self.v_target.parameters(), self.v_net.parameters()):
                p_tgt.data = 0.005 * p_online.data + 0.995 * p_tgt.data

    #  Environment interaction (same) 
    def env_step(self, state, intended_action):
        node, budget = state
        actual = self.env.sample_executed_action(node, intended_action, self.p_intended)
        travel_time = self.env.sample_travel_time(node, actual)
        next_state = (actual, budget - travel_time)
        return actual, next_state

    #  EU-RAC update with TD(lambda) + neural actor gradient 
    def update(self, state, intended_action, actual_action, next_state):
        next_node, next_budget = next_state

        # Execution-stage target y_e
        if next_node == self.env.dest and next_budget >= 0:
            y_e = 1.0
        elif next_budget < 0:
            y_e = 0.0
        else:
            y_e = self._compute_V_target(next_state)

        key_e = (state, actual_action)
        key_d = (state, intended_action)

        # --- Q_e update with TD(lambda) ---
        delta_e = y_e - self.Q_e[key_e]
        self.e_Qe[key_e] = 1.0
        for key, e_val in list(self.e_Qe.items()):
            if abs(e_val) > 1e-8:
                self.Q_e[key] += self.lr_e * delta_e * e_val
                self.e_Qe[key] = e_val * self.lambda_td

        # --- Q_d update with TD(lambda) ---
        y_d = self.Q_e[key_e]
        delta_d = y_d - self.Q_d[key_d]
        self.e_Qd[key_d] = 1.0
        for key, e_val in list(self.e_Qd.items()):
            if abs(e_val) > 1e-8:
                self.Q_d[key] += self.lr_d * delta_d * e_val
                self.e_Qd[key] = e_val * self.lambda_td

        # --- Train V_phi ---
        self._train_V_step(state, y_e)

        # --- Neural Actor update (policy gradient + KL anchor) ---
        node, budget = state
        V_s = self._compute_V_phi(state)
        A = float(self.Q_d[key_d] - V_s)  # Python float; no gradient to V_phi or Q_d.

        acts = self.env.get_actions(node)
        if len(acts) <= 1:
            return
        if intended_action not in acts:
            return

        self.pi_optimizer.zero_grad()
        node_embs = self._get_node_embeddings()
        raw, clamped = self._action_logits_clamped(node, budget, acts, node_embs)

        # Track both raw and clamped stats from same forward (rolling)
        self._diag_raw_logit_means.append(float(raw.mean().item()))
        self._diag_raw_logit_stds.append(float(raw.std().item()))
        self._diag_logit_means.append(float(clamped.mean().item()))
        self._diag_logit_stds.append(float(clamped.std().item()))

        # Use clamped logits for stable softmax with gradients through the clamp.
        log_probs = clamped - torch.logsumexp(clamped, dim=0)
        idx = acts.index(intended_action)
        chosen_log_prob = log_probs[idx]

        # Entropy bonus (using clamped logits for stability)
        probs = F.softmax(clamped, dim=0)
        entropy_val = -(probs * torch.log(probs + 1e-8)).sum()

        # Actor update. When frozen during RL, only the warm-start policy is used.
        # Conservative actor improvement: keep the warm-start policy as a KL anchor.
        ref_probs = self._reference_probs(state, acts)
        if ref_probs is None:
            kl_loss = torch.tensor(0.0, device=self._device)
            kl_cur_ref = torch.tensor(0.0, device=self._device)
        else:
            ref_probs = ref_probs.detach()
            kl_loss = torch.sum(ref_probs * (
                torch.log(ref_probs + 1e-8) - torch.log(probs + 1e-8)))
            kl_cur_ref = torch.sum(probs.detach() * (
                torch.log(probs.detach() + 1e-8) - torch.log(ref_probs + 1e-8)))
        for group in self.pi_optimizer.param_groups:
            group['lr'] = self.lr_actor
        actor_loss = -A * chosen_log_prob - self.entropy_coef * entropy_val + self.kl_coef * kl_loss
        actor_loss.backward()
        train_params = (self.action_scorer.parameters() if self._gcn_frozen
                        else list(self.gcn.parameters()) + list(self.action_scorer.parameters()))
        nn.utils.clip_grad_norm_(train_params, self.actor_grad_clip)
        self.pi_optimizer.step()
        self._diag_actor_updates += 1
        # Track diagnostics (rolling, deque maxlen auto-evicts oldest)
        self._diag_advantages.append(float(A))
        self._diag_max_probs.append(float(probs.max().item()))
        self._diag_entropies.append(float(entropy_val.item()))
        self._diag_kl_ref_cur.append(float(kl_loss.item()))
        self._diag_kl_cur_ref.append(float(kl_cur_ref.item()))
        self._diag_actor_losses.append(float(actor_loss.item()))

    #  Episode runners 
    def run_episode(self, train=True):
        if train:
            self.e_Qe.clear()
            self.e_Qd.clear()
        if train and len(self._v_cache) > 100000:
            self._v_cache.clear()
            self._v_target_cache.clear()
        state = (self.env.origin, self.env.budget)
        max_steps = len(self.env.nodes) * 2
        visited = {self.env.origin}
        step_counter = 0
        for _ in range(max_steps):
            node, budget = state
            if node == self.env.dest or budget < 0:
                outcome = 1.0 if (node == self.env.dest and budget >= 0) else 0.0
                return self._finish_episode(outcome, train)

            intended = self.select_action(state,
                                           exclude=visited if train else None)
            if intended is None:
                return self._finish_episode(0.0, train)

            actual, next_state = self.env_step(state, intended)
            if train and step_counter % self.update_freq == 0:
                self.update(state, intended, actual, next_state)
            visited.add(actual)
            state = next_state
            step_counter += 1
        return self._finish_episode(0.0, train)

    def train(self, n_episodes=30000, checkpoint_episodes=None, checkpoint_dir=None,
              env_factory=None, val_episodes=200, val_seed=12345,
              auto_checkpoint_interval=10000, resume=True):
        checkpoint_set = set(checkpoint_episodes or [])
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        latest_path = ckpt_dir / 'eurac_latest.pt' if ckpt_dir is not None else None
        start_episode = 0
        if resume and latest_path is not None and latest_path.exists():
            start_episode = self.load_checkpoint(latest_path)
            if start_episode >= n_episodes:
                print(f'    [EU-RAC] latest checkpoint already reached '
                      f'{start_episode}/{n_episodes} episodes')
                return []
            print(f'    [EU-RAC] resumed from {latest_path} '
                  f'({start_episode}/{n_episodes} episodes complete)')

        results = []
        self.training_checkpoints = []
        best_mc = -float('inf')
        for ep in range(start_episode + 1, n_episodes + 1):
            results.append(self.run_episode(train=True))

            should_checkpoint = ep in checkpoint_set
            if auto_checkpoint_interval and ep % auto_checkpoint_interval == 0:
                should_checkpoint = True
            if ep == n_episodes:
                should_checkpoint = True
            if not should_checkpoint:
                continue

            mc = None
            if env_factory is not None:
                mc = self.validate_mc(env_factory, n_episodes=val_episodes,
                                      val_seed=val_seed + ep)

            path = None
            if ckpt_dir is not None:
                if ep in checkpoint_set:
                    path = ckpt_dir / f'eurac_ep{ep}.pt'
                    self.save_checkpoint(path, episode=ep, n_episodes=n_episodes)
                self.save_checkpoint(latest_path, episode=ep, n_episodes=n_episodes)
                if mc is not None and mc > best_mc:
                    best_mc = mc
                    self.save_checkpoint(ckpt_dir / 'eurac_best.pt',
                                         episode=ep, n_episodes=n_episodes)

            gp = self.get_greedy_path()
            self.training_checkpoints.append({
                'episode': ep,
                'mc': mc,
                'dijkstra_agreement': self.compute_dijkstra_agreement(),
                'greedy_path_reaches': bool(gp and gp[-1] == self.env.dest),
                'greedy_path': gp,
                'diagnostics': self.get_diagnostics(),
                'path': str(path or latest_path) if ckpt_dir is not None else None,
            })
        return results

    def evaluate(self, n_episodes=10000):
        results = [self.run_episode(train=False) for _ in range(n_episodes)]
        return float(np.mean(results))

    def q_stats(self):
        if self.Q_e:
            avg_q_e = float(np.mean(list(self.Q_e.values())))
        else:
            avg_q_e = 0.0
        if self.Q_d:
            avg_q_d = float(np.mean(list(self.Q_d.values())))
        else:
            avg_q_d = 0.0
        return avg_q_e, avg_q_d

    def actor_parameter_drift(self):
        """L2 distance from the frozen warm-start action scorer."""
        if self._warm_actor_state is None:
            return 0.0
        d2 = 0.0
        for k, v in self.action_scorer.state_dict().items():
            d2 += (v.detach() - self._warm_actor_state[k]).pow(2).sum().item()
        return float(np.sqrt(d2))

    def policy_kl_to_reference(self, node_list=None, budget=None):
        """Mean KL(pi_current || pi_warm) over selected nodes."""
        if self.ref_scorer is None:
            return float('nan')
        nodes = node_list if node_list is not None else [self.env.origin]
        b = self.env.budget if budget is None else budget
        kls = []
        with torch.no_grad():
            for node in nodes:
                acts = self.env.successors.get(node, [])
                if len(acts) <= 1:
                    continue
                cur_probs = self.action_probs((node, b), acts)
                ref_probs = self._reference_probs((node, b), acts)
                if ref_probs is None:
                    continue
                kl = torch.sum(cur_probs * (
                    torch.log(cur_probs + 1e-8) - torch.log(ref_probs + 1e-8)))
                kls.append(float(kl.item()))
        return float(np.mean(kls)) if kls else float('nan')

    def get_diagnostics(self):
        diag = {}
        if self._diag_advantages:
            diag['adv_mean'] = float(np.mean(self._diag_advantages))
            diag['adv_std'] = float(np.std(self._diag_advantages))
        else:
            diag['adv_mean'] = diag['adv_std'] = float('nan')
        if self._diag_entropies:
            diag['entropy_mean'] = float(np.mean(self._diag_entropies))
        else:
            diag['entropy_mean'] = float('nan')
        if self._diag_max_probs:
            diag['max_prob_mean'] = float(np.mean(self._diag_max_probs))
        else:
            diag['max_prob_mean'] = float('nan')
        if self._diag_kl_ref_cur:
            diag['kl_ref_cur_mean'] = float(np.mean(self._diag_kl_ref_cur))
            diag['kl_cur_ref_mean'] = float(np.mean(self._diag_kl_cur_ref))
        else:
            diag['kl_ref_cur_mean'] = diag['kl_cur_ref_mean'] = float('nan')
        if self._diag_actor_losses:
            diag['actor_loss_mean'] = float(np.mean(self._diag_actor_losses))
        else:
            diag['actor_loss_mean'] = float('nan')
        diag['actor_parameter_drift'] = self.actor_parameter_drift()
        diag['actor_updates'] = self._diag_actor_updates
        if self._diag_logit_means:
            diag['logit_mean'] = float(np.mean(self._diag_logit_means))
            diag['logit_std'] = float(np.mean(self._diag_logit_stds))
        else:
            diag['logit_mean'] = diag['logit_std'] = float('nan')
        if self._diag_raw_logit_means:
            diag['raw_logit_mean'] = float(np.mean(self._diag_raw_logit_means))
            diag['raw_logit_std'] = float(np.mean(self._diag_raw_logit_stds))
            diag['raw_logit_min'] = float(np.min(self._diag_raw_logit_means))
            diag['raw_logit_max'] = float(np.max(self._diag_raw_logit_means))
        else:
            diag['raw_logit_mean'] = diag['raw_logit_std'] = float('nan')
            diag['raw_logit_min'] = diag['raw_logit_max'] = float('nan')
        diag['n_params'] = (sum(p.numel() for p in self.v_net.parameters()) +
                            sum(p.numel() for p in self.gcn.parameters()) +
                            sum(p.numel() for p in self.action_scorer.parameters()))
        diag['dijkstra_agree_before'] = self._diag_dijkstra_agreement_before
        diag['dijkstra_agree_after'] = self._diag_dijkstra_agreement_after

        # Fresh V_phi stats (use_cache=False to bypass stale cache)
        gp = self.get_greedy_path()
        if gp and len(gp) >= 2:
            v_fresh = []
            remaining = self.env.budget
            edges = self.env.edges
            for i, node in enumerate(gp):
                v_fresh.append(self._compute_V_phi((node, max(0, remaining)),
                                                    use_cache=False))
                if node == self.env.dest:
                    break
                if i < len(gp) - 1:
                    next_n = gp[i + 1]
                    remaining -= max(1, round(edges[(node, next_n)][0]))
            diag['V_phi_mean'] = float(np.mean(v_fresh))
            diag['V_phi_std'] = float(np.std(v_fresh))
        else:
            diag['V_phi_mean'] = float('nan')
            diag['V_phi_std'] = float('nan')
        return diag

    def compute_dijkstra_agreement(self):
        edges = self.env.edges
        successors = self.env.successors
        origin = self.env.origin; dest = self.env.dest

        dist = {origin: 0.0}; prev = {}; pq = [(0.0, origin)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')): continue
            if u == dest: break
            for v in successors.get(u, []):
                mean_t, _ = edges[(u, v)]
                nd = d + max(1, round(mean_t))
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if dest not in prev:
            return 0.0

        path = [dest]; n = dest
        while n in prev: path.append(prev[n]); n = prev[n]
        path.reverse()

        budget = self.env.budget
        agree = 0; total = 0
        with torch.no_grad():
            node_embs = self._get_node_embeddings()
            for i in range(len(path) - 1):
                node = path[i]; dijkstra_action = path[i + 1]
                total += 1
                acts = successors.get(node, [])
                if not acts: continue
                current_b = budget - sum(
                    max(1, round(edges[(path[j], path[j+1])][0]))
                    for j in range(i))
                _, probs = self._neural_probs(node, max(0, current_b), acts, node_embs)
                my_action = acts[int(np.argmax(probs))]
                if my_action == dijkstra_action:
                    agree += 1
        return agree / max(total, 1)

    #  Checkpoint & Validation 
    def save_checkpoint(self, path, episode=None, n_episodes=None):
        """Save full agent state."""
        state = {
            'episode': episode,
            'n_episodes': n_episodes,
            'gcn': self.gcn.state_dict(),
            'action_scorer': self.action_scorer.state_dict(),
            'ref_scorer': self.ref_scorer.state_dict() if self.ref_scorer is not None else None,
            'ref_node_embs': self._ref_node_embs,
            'warm_actor_state': self._warm_actor_state,
            'gcn_frozen': self._gcn_frozen,
            'v_net': self.v_net.state_dict(),
            'v_target': self.v_target.state_dict(),
            'pi_opt': self.pi_optimizer.state_dict(),
            'v_opt': self.v_optimizer.state_dict(),
            'Q_e': dict(self.Q_e),
            'Q_d': dict(self.Q_d),
            'e_Qe': dict(self.e_Qe),
            'e_Qd': dict(self.e_Qd),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'env_rng_state': self.env.rng.bit_generator.state if hasattr(self.env, 'rng') else None,
            'theta_keys': list(self.Q_d.keys())[:10],  # sanity check
            'python_rng_state': random.getstate(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        torch.save(state, tmp_path)
        tmp_path.replace(path)

    def load_checkpoint(self, path):
        """Restore full agent state."""
        state = torch.load(path, map_location=self._device, weights_only=False)
        self.gcn.load_state_dict(state['gcn'])
        self.action_scorer.load_state_dict(state['action_scorer'])
        self._gcn_frozen = bool(state.get('gcn_frozen', False))
        if self._gcn_frozen:
            for p in self.gcn.parameters():
                p.requires_grad_(False)
            self._cached_embs = self.gcn(self._static_node_feats, self._adj_norm).detach()
            self.pi_optimizer = optim.Adam(self.action_scorer.parameters(), lr=self.lr_actor)
        else:
            for p in self.gcn.parameters():
                p.requires_grad_(True)
            self._cached_embs = None
            self.pi_optimizer = optim.Adam(
                list(self.gcn.parameters()) + list(self.action_scorer.parameters()),
                lr=self.warm_start_lr)
        if state.get('ref_scorer') is not None:
            self.ref_scorer = copy.deepcopy(self.action_scorer)
            self.ref_scorer.load_state_dict(state['ref_scorer'])
            self.ref_scorer.eval()
            for p in self.ref_scorer.parameters():
                p.requires_grad_(False)
        self._ref_node_embs = state.get('ref_node_embs')
        self._warm_actor_state = state.get('warm_actor_state')
        self.v_net.load_state_dict(state['v_net'])
        self.v_target.load_state_dict(state['v_target'])
        self.pi_optimizer.load_state_dict(state['pi_opt'])
        self.v_optimizer.load_state_dict(state['v_opt'])
        self.Q_e.update(state['Q_e'])
        self.Q_d.update(state['Q_d'])
        self.e_Qe.update(state.get('e_Qe', {}))
        self.e_Qd.update(state.get('e_Qd', {}))
        if state.get('python_rng_state') is not None:
            random.setstate(state['python_rng_state'])
        if state.get('numpy_rng_state') is not None:
            np.random.set_state(state['numpy_rng_state'])
        if state.get('torch_rng_state') is not None:
            torch.set_rng_state(state['torch_rng_state'])
        if torch.cuda.is_available() and state.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state_all'])
        if state.get('env_rng_state') is not None and hasattr(self.env, 'rng'):
            self.env.rng.bit_generator.state = state['env_rng_state']
        if self._gcn_frozen and self._cached_embs is None:
            self._cached_embs = self._get_node_embeddings().detach()
        self._episodes_trained = int(state.get('episode') or 0)
        return self._episodes_trained

    def validate_mc(self, env_factory, n_episodes=200, val_seed=12345):
        """Quick MC eval with a fixed independent seed."""
        from evaluator import evaluate_policy
        val_env = env_factory(val_seed)
        def _pol(node, bud):
            s = (node, bud); pol, acts = self.get_policy(s)
            if not acts: return None
            return max(pol, key=pol.get)
        return evaluate_policy(val_env, _pol, episodes=n_episodes)

    def get_greedy_path(self):
        successors = self.env.successors
        edges = self.env.edges
        origin = self.env.origin; dest = self.env.dest
        budget = self.env.budget
        path = [origin]; node = origin; b = budget; visited = {node}
        max_steps = len(self.env.nodes) * 2
        with torch.no_grad():
            node_embs = self._get_node_embeddings()
            for _ in range(max_steps):
                if node == dest or b < 0: break
                acts = [a for a in successors.get(node, []) if a not in visited]
                if not acts: break
                _, probs = self._neural_probs(node, b, acts, node_embs)
                action = acts[int(np.argmax(probs))]
                mean_t, _ = edges[(node, action)]
                b -= max(1, round(mean_t))
                visited.add(action); path.append(action); node = action
        return path













