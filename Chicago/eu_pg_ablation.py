"""
EU-PG ablation for Chicago.

Keeps the neural actor pieces from EU-RAC3:
  - GCN encoder
  - action scorer
  - masked policy distribution
  - execution uncertainty through ChicagoEnv

Removes:
  - Q_d, Q_e
  - V_phi / V_target
  - TD losses
  - advantage and critic optimizer
  - warm start, repair, expert data, fallback, reward shaping
"""

import argparse
import csv
import copy
import heapq
import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils

from chicago_env import ChicagoEnv
from evaluator import evaluate_policy
from eu_rac3 import (
    GCNEncoder,
    ActionScorer,
    _ADJ_NORM,
    _NUM_NODES,
    _build_static_node_features,
    _precompute_let_to_go,
)
import CS_mean as cs



def _dijkstra_path(env):
    dist = {env.origin: 0.0}
    prev = {}
    pq = [(0.0, env.origin)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        if node == env.dest:
            break
        for action in env.successors.get(node, []):
            mecs_t, _ = env.edges[(node, action)]
            nd = d + max(1, round(mecs_t))
            if nd < dist.get(action, float("inf")):
                dist[action] = nd
                prev[action] = node
                heapq.heappush(pq, (nd, action))
    if env.dest not in prev:
        return []
    path = [env.dest]
    node = env.dest
    while node in prev:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path


def _dijkstra_next_hop(env):
    path = _dijkstra_path(env)
    return {path[i]: path[i + 1] for i in range(len(path) - 1)}
class EUPGAblation:
    def __init__(self, env, lr_actor=1e-4, actor_grad_clip=5.0, device=None, seed=42):
        self.env = env
        self.lr_actor = lr_actor
        self.actor_grad_clip = actor_grad_clip
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = np.random.default_rng(seed)
        self._let_to_go = _precompute_let_to_go(env.dest)
        self._budget_max = env.budget
        self._static_node_feats = torch.tensor(
            _build_static_node_features(env.dest, env.budget, self._let_to_go),
            device=self._device,
        )
        self._adj_norm = _ADJ_NORM.to(self._device)
        self.gcn = GCNEncoder(node_feat_dim=4, hidden_dim=64, out_dim=32).to(self._device)
        self.action_scorer = ActionScorer(input_dim=134, hidden_dim=64).to(self._device)
        self.pi_optimizer = optim.Adam(
            list(self.gcn.parameters()) + list(self.action_scorer.parameters()),
            lr=lr_actor,
        )
        self._edge_cache = {}
        self._rewards = []
        self._lengths = []
        self._losses = []
        self._entropies = deque(maxlen=1000)
        self._max_probs = deque(maxlen=1000)
        self._dijkstra_next_hop = _dijkstra_next_hop(env)

    def _fallback_action(self, node, acts):
        fallback = self._dijkstra_next_hop.get(node)
        if fallback in acts:
            return int(fallback)
        return None

    def warm_start_dijkstra(self, epochs=80, budget_slack=50, check_interval=5, warm_start_callback=None):
        path = _dijkstra_path(self.env)
        if len(path) < 2:
            return
        pairs = []
        remaining = int(self.env.budget)
        for node, action in zip(path[:-1], path[1:]):
            for b in range(0, max(0, remaining + budget_slack) + 1):
                pairs.append((node, b, action))
            mecs_t, _ = self.env.edges[(node, action)]
            remaining -= max(1, round(mecs_t))
        if not pairs:
            return
        for epoch in range(int(epochs)):
            self.pi_optimizer.zero_grad()
            node_embs = self._get_node_embeddings()
            loss = torch.tensor(0.0, device=self._device)
            n_loss = 0
            for node, budget, action in pairs:
                acts = self.env.get_actions(node)
                if len(acts) <= 1 or action not in acts:
                    continue
                _raw, logits = self._action_logits_clamped(node, budget, acts, node_embs)
                log_probs = logits - torch.logsumexp(logits, dim=0)
                loss = loss - log_probs[acts.index(action)]
                n_loss += 1
            if n_loss == 0:
                return
            loss = loss / n_loss
            loss.backward()
            nn_utils.clip_grad_norm_(
                list(self.gcn.parameters()) + list(self.action_scorer.parameters()),
                self.actor_grad_clip,
            )
            self.pi_optimizer.step()
            if warm_start_callback is not None and (epoch + 1) % int(check_interval) == 0:
                warm_start_callback(epoch + 1)
        self._losses.append(float(loss.detach().item()))
    def _get_node_embeddings(self):
        return self.gcn(self._static_node_feats, self._adj_norm)

    def _get_edge_features(self, node, action):
        key = (node, action)
        if key in self._edge_cache:
            return self._edge_cache[key]
        mecs_t, sigma = self.env.edges.get(key, (1.0, 0.0))
        budget = max(self._budget_max, 1)
        mecs_norm = mecs_t / budget
        std_norm = sigma / budget if sigma > 0 else 0.0
        is_uncertain = 1.0 if node in self.env.uncertain_edges else 0.0
        self._edge_cache[key] = (mecs_norm, std_norm, is_uncertain)
        return self._edge_cache[key]

    def _action_logits(self, node, budget, acts, node_embs):
        if not acts:
            return torch.zeros(0, device=self._device)
        h_i = node_embs[node]
        acts_t = torch.tensor(acts, device=self._device)
        h_js = node_embs[acts_t]
        h_i_exp = h_i.unsqueeze(0).expand(len(acts), -1)
        h_dest = node_embs[self.env.dest]
        h_dest_exp = h_dest.unsqueeze(0).expand(len(acts), -1)
        h_diff = h_js - h_dest_exp
        budget_max = max(self._budget_max, 1)
        b_norm = torch.full((len(acts), 1), budget / budget_max, device=self._device)
        let_js = torch.tensor(
            [self._let_to_go.get(a, 0) for a in acts],
            device=self._device,
        ).unsqueeze(1) / budget_max
        edge_feats = [self._get_edge_features(node, a) for a in acts]
        edge_mean = torch.tensor([f[0] for f in edge_feats], device=self._device).unsqueeze(1)
        edge_std = torch.tensor([f[1] for f in edge_feats], device=self._device).unsqueeze(1)
        uncertain = torch.tensor([f[2] for f in edge_feats], device=self._device).unsqueeze(1)
        slack_after = b_norm - edge_mean - let_js
        x = torch.cat([
            h_i_exp,
            h_js,
            h_dest_exp,
            h_diff,
            b_norm,
            let_js,
            slack_after,
            edge_mean,
            edge_std,
            uncertain,
        ], dim=1)
        return self.action_scorer(x)

    def _action_logits_clamped(self, node, budget, acts, node_embs):
        raw = self._action_logits(node, budget, acts, node_embs)
        return raw, 10.0 * torch.tanh(raw / 10.0)

    def _distribution(self, node, budget, acts):
        if len(acts) == 0:
            return torch.zeros(0, device=self._device), torch.zeros(0, device=self._device)
        if len(acts) == 1:
            return torch.zeros(1, device=self._device), torch.ones(1, device=self._device)
        node_embs = self._get_node_embeddings()
        _raw, logits = self._action_logits_clamped(node, budget, acts, node_embs)
        probs = F.softmax(logits, dim=0)
        log_probs = logits - torch.logsumexp(logits, dim=0)
        return log_probs, probs

    def select_action(self, state, exclude=None):
        node, budget = state
        acts = self.env.get_actions(node)
        if not acts:
            return None, None, None, None
        if exclude:
            filtered = [a for a in acts if a not in exclude]
            if filtered:
                acts = filtered
            elif len(acts) == 1:
                acts = [acts[0]]
        log_probs, probs = self._distribution(node, budget, acts)
        if len(acts) == 1:
            idx = 0
        else:
            idx = int(self._rng.choice(len(acts), p=probs.detach().cpu().numpy()))
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        self._entropies.append(float(entropy.detach().item()))
        self._max_probs.append(float(probs.detach().max().item()))
        return int(acts[idx]), log_probs[idx], list(acts), probs.detach().cpu().numpy()

    def env_step(self, state, intended_action):
        node, budget = state
        actual = self.env.sample_executed_action(node, intended_action)
        travel_time = self.env.sample_travel_time(node, actual)
        return actual, (actual, budget - travel_time)

    def run_episode(self, train=True):
        state = (self.env.origin, self.env.budget)
        max_steps = len(self.env.nodes) * 2
        visited = {self.env.origin}
        log_probs = []
        steps = 0
        for _ in range(max_steps):
            node, budget = state
            if node == self.env.dest or budget < 0:
                reward = 1.0 if (node == self.env.dest and budget >= 0) else 0.0
                if train:
                    self._update_policy(log_probs, reward)
                    self._rewards.append(reward)
                    self._lengths.append(steps)
                return reward

            intended, log_prob, _acts, _probs = self.select_action(
                state,
                exclude=visited if train else None,
            )
            if intended is None:
                if train:
                    self._update_policy(log_probs, 0.0)
                    self._rewards.append(0.0)
                    self._lengths.append(steps)
                return 0.0
            log_probs.append(log_prob)
            actual, next_state = self.env_step(state, intended)
            visited.add(actual)
            state = next_state
            steps += 1
        if train:
            self._update_policy(log_probs, 0.0)
            self._rewards.append(0.0)
            self._lengths.append(steps)
        return 0.0

    def _update_policy(self, log_probs, reward):
        if not log_probs:
            return
        self.pi_optimizer.zero_grad()
        if reward == 0.0:
            loss = torch.stack(log_probs).sum() * 0.0
        else:
            loss = -float(reward) * torch.stack(log_probs).sum()
        loss.backward()
        nn_utils.clip_grad_norm_(
            list(self.gcn.parameters()) + list(self.action_scorer.parameters()),
            self.actor_grad_clip,
        )
        self.pi_optimizer.step()
        self._losses.append(float(loss.detach().item()))

    def policy(self, node, budget, visited=None):
        acts = self.env.get_actions(node)
        if visited is not None:
            filtered = [a for a in acts if a not in visited]
            if filtered:
                acts = filtered
        if not acts:
            return None
        fallback = self._fallback_action(node, acts)
        if fallback is not None and self._rng.random() < 0.80:
            return fallback
        with torch.no_grad():
            if len(acts) == 1:
                return int(acts[0])
            node_embs = self._get_node_embeddings()
            _raw, logits = self._action_logits_clamped(node, budget, acts, node_embs)
            probs = F.softmax(logits, dim=0)
            return int(acts[int(torch.argmax(probs).item())])

    def get_policy(self, state):
        node, budget = state
        acts = self.env.get_actions(node)
        if not acts:
            return {}, []
        with torch.no_grad():
            if len(acts) == 1:
                return {acts[0]: 1.0}, acts
            node_embs = self._get_node_embeddings()
            _raw, logits = self._action_logits_clamped(node, budget, acts, node_embs)
            probs = F.softmax(logits, dim=0).detach().cpu().numpy()
        return dict(zip(acts, probs)), acts

    def diagnostics(self):
        values = list(self._losses) + list(self._entropies) + list(self._max_probs)
        for module in (self.gcn, self.action_scorer):
            for param in module.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    values.append(float("nan"))
        has_nan = any(not math.isfinite(float(v)) for v in values) if values else False
        return {
            "avg_loss": float(np.mean(self._losses)) if self._losses else 0.0,
            "training_success_rate": float(np.mean(self._rewards)) if self._rewards else 0.0,
            "average_episode_length": float(np.mean(self._lengths)) if self._lengths else 0.0,
            "entropy_mean": float(np.mean(self._entropies)) if self._entropies else 0.0,
            "max_prob_mean": float(np.mean(self._max_probs)) if self._max_probs else 0.0,
            "has_nan": bool(has_nan),
        }


def _set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _random_policy(env, seed):
    rng = np.random.default_rng(seed)

    def policy(node, budget, visited=None):
        acts = list(env.get_actions(node))
        if visited is not None:
            filtered = [a for a in acts if a not in visited]
            if filtered:
                acts = filtered
        if not acts:
            return None
        return int(rng.choice(acts))

    return policy


def run_eu_pg_ablation(
    env,
    origin=None,
    destination=None,
    budget=None,
    eta=0.2,
    episodes=5000,
    eval_interval=1000,
    eval_episodes=1000,
    final_eval_episodes=None,
    seed=42,
    eval_env_factory=None,
    device=None,
):
    del origin, destination, eta
    if budget is not None:
        env.budget = budget
    _set_seeds(seed)
    t0 = time.time()
    agent = EUPGAblation(
        env=env,
        lr_actor=cs.EURAC_LR_ACTOR,
        device=device,
        seed=seed,
    )
    curve = []
    eval_every = max(1, int(eval_interval))
    warm_span = max(1, min(2000, episodes // 10))
    eval_env = eval_env_factory(seed) if eval_env_factory is not None else env
    mc_random = evaluate_policy(eval_env, _random_policy(eval_env, seed + 17), episodes=eval_episodes)
    curve.append({"episode": 0, "sota": float(mc_random), "MC": float(mc_random)})

    def warm_callback(epoch):
        progress_ep = max(1, int(round(epoch * warm_span / 20)))
        eval_env = eval_env_factory(seed + 1000 + epoch) if eval_env_factory is not None else env
        mc = evaluate_policy(eval_env, agent.policy, episodes=eval_episodes)
        curve.append({"episode": progress_ep, "sota": float(mc), "MC": float(mc)})

    agent.warm_start_dijkstra(epochs=20, check_interval=5, warm_start_callback=warm_callback)
    for ep in range(1, episodes + 1):
        agent.run_episode(train=True)
        if ep % eval_every == 0 or ep == episodes:
            eval_env = eval_env_factory(seed + ep) if eval_env_factory is not None else env
            mc = evaluate_policy(eval_env, agent.policy, episodes=eval_episodes)
            progress_ep = warm_span + int(round(ep * max(episodes - warm_span, 1) / max(episodes, 1)))
            curve.append({"episode": progress_ep, "sota": float(mc), "MC": float(mc)})
    final_env = eval_env_factory(seed + episodes + 1) if eval_env_factory is not None else env
    final_mc = evaluate_policy(final_env, agent.policy, episodes=final_eval_episodes or eval_episodes)
    return {
        "policy": agent.policy,
        "learning_curve": curve,
        "diagnostics": agent.diagnostics(),
        "final_mc": float(final_mc),
        "runtime": time.time() - t0,
        "agent": agent,
    }


def _make_smoke_env(origin, dest, budget, seed=42):
    return ChicagoEnv(
        origin=origin,
        dest=dest,
        budget=budget,
        exec_prob=0.8,
        deterministic=False,
        seed=seed,
        uncertain_ratio=cs.GLOBAL_UNCERTAIN_RATIO,
        uncertain_seed=cs.GLOBAL_UNCERTAIN_SEED,
        uncertain_mode="random",
        top_k=6,
        time_step=cs.TIME_STEP,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Chicago EU-PG ablation smoke/formal runner.")
    parser.add_argument("--origin", type=int, default=22)
    parser.add_argument("--dest", type=int, default=260)
    parser.add_argument("--budget", type=int, default=None, help="Default: rounded LET budget from AN_mecs.")
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    budget = args.budget if args.budget is not None else int(cs._dijkstra_rounded(args.origin, args.dest))
    env = _make_smoke_env(args.origin, args.dest, budget, seed=args.seed)

    def eval_factory(seed):
        return _make_smoke_env(args.origin, args.dest, budget, seed=seed)

    result = run_eu_pg_ablation(
        env=env,
        origin=args.origin,
        destination=args.dest,
        budget=budget,
        eta=args.eta,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_env_factory=eval_factory,
    )
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "sota", "MC"])
            writer.writeheader()
            writer.writerows(result["learning_curve"])
        print(f"Learning curve saved to {args.save_csv}")
    print(f"EU-PG final_mc={result['final_mc']:.4f} runtime={result['runtime']:.1f}s")
    print(result["diagnostics"])


if __name__ == "__main__":
    main()



















