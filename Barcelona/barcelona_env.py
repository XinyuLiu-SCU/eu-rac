"""
BarcelonaEnv  SiouxEnv-compatible environment for the Barcelona network.

Same interface as ChicagoEnv but loads Barcelona network data.
Sigma is extracted from the diagonal of Barcelona_cov.npy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import heapq

NETWORK_DIR = Path(__file__).parent / 'network'


class BarcelonaEnv:
    def __init__(self, origin, dest, budget,
                 exec_prob=0.8, uncertain_ratio=0.2,
                 deterministic=False, seed=42, uncertain_seed=42,
                 uncertain_mode='random', top_k=6,
                 uncertain_nodes=None,
                 uncertain_edges_explicit=None,
                 use_od_uncertainty=False,
                 od_target_uncertain_ratio=0.15,
                 od_max_uncertain_ratio=0.20,
                 od_uncertain_seed=42):
        self.origin = origin
        self.dest = dest
        self.budget = budget
        self.exec_prob = exec_prob
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

        self._load_network()
        if uncertain_edges_explicit is not None:
            self._setup_explicit_uncertainty(uncertain_edges_explicit)
        elif uncertain_nodes is not None:
            self._setup_fixed_uncertainty(uncertain_nodes)
        else:
            self._setup_execution_uncertainty(
                ratio=uncertain_ratio,
                seed=uncertain_seed,
                mode=uncertain_mode,
                top_k=top_k,
            )
            if use_od_uncertainty:
                self._calibrate_od_uncertainty(
                    target_ratio=od_target_uncertain_ratio,
                    max_ratio=od_max_uncertain_ratio,
                    seed=od_uncertain_seed,
                )

    def _load_network(self):
        net_df = pd.read_csv(NETWORK_DIR / 'Barcelona_network.csv')
        # Extract per-edge sigma from the diagonal of the full covariance matrix
        # Covariance is in hours; convert to minutes then sqrt  sigma in minutes
        cov_diag = np.diag(np.load(NETWORK_DIR / 'Barcelona_cov.npy', mmap_mode='r'))
        sigma_arr = np.sqrt(np.maximum(cov_diag, 0.0)) * 60

        self.edges = {}
        self.successors = defaultdict(list)

        for i, row in net_df.iterrows():
            u, v = int(row['From']), int(row['To'])
            mean_t = float(row['Cost']) * 60   # hours  minutes
            sigma = float(sigma_arr[i])
            self.edges[(u, v)] = (mean_t, sigma)
            self.successors[u].append(v)

        self.nodes = set()
        for u, v in self.edges:
            self.nodes.add(u)
            self.nodes.add(v)

    def _setup_explicit_uncertainty(self, edges_dict):
        self.uncertain_mode = 'explicit'
        self.uncertain_edges = {}
        for node, spec in edges_dict.items():
            node = int(node)
            if node not in self.nodes:
                raise ValueError(f"uncertain_edges_explicit: node {node} not in network")
            if isinstance(spec, dict):
                intended = int(spec['intended'])
                if intended not in self.successors.get(node, []):
                    raise ValueError(f"uncertain_edges_explicit: edge {node}->{intended} does not exist")
                self.uncertain_edges[node] = {'intended': intended}
            else:
                next_node = int(spec)
                if next_node not in self.successors.get(node, []):
                    raise ValueError(f"uncertain_edges_explicit: edge {node}->{next_node} does not exist")
                self.uncertain_edges[node] = next_node

    def _setup_fixed_uncertainty(self, uncertain_nodes):
        rng = np.random.default_rng(0)
        self.uncertain_mode = 'fixed'
        self.uncertain_edges = {}
        for node in uncertain_nodes:
            if node not in self.nodes:
                continue
            succs = self.successors[node]
            if succs:
                self.uncertain_edges[node] = int(rng.choice(succs))

    def _setup_execution_uncertainty(self, ratio, seed, mode='random', top_k=6):
        rng = np.random.default_rng(seed)
        if mode == 'important':
            chosen = self._important_nodes(top_k)
        else:
            candidates = sorted(n for n in self.nodes if len(self.successors[n]) >= 2)
            n_uncertain = max(1, int(ratio * len(candidates)))
            chosen = rng.choice(candidates, size=n_uncertain, replace=False).tolist()
        self.uncertain_mode = mode
        self.uncertain_edges = {}
        for node in chosen:
            succs = self.successors[node]
            if succs:
                self.uncertain_edges[node] = int(rng.choice(succs))

    def _dijkstra_mean_path(self):
        dist = {self.origin: 0.0}
        prev = {}
        pq = [(0.0, self.origin)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            if u == self.dest:
                break
            for v in self.successors.get(u, []):
                mean_t, _ = self.edges[(u, v)]
                nd = d + mean_t
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if self.dest not in dist:
            return []

        path = [self.dest]
        node = self.dest
        while node != self.origin:
            node = prev.get(node)
            if node is None:
                return []
            path.append(node)
        path.reverse()
        return path

    def _let_uncertainty_exposure(self, path):
        transitions = max(0, len(path) - 1)
        if transitions == 0:
            return 0, 0.0
        risky = sum(
            1
            for u, v in zip(path[:-1], path[1:])
            if self.uncertain_edges.get(u) == v
        )
        return risky, risky / transitions

    def _calibrate_od_uncertainty(self, target_ratio, max_ratio, seed):
        """Add OD-specific LET risky edges without removing global uncertainty."""
        path = self._dijkstra_mean_path()
        transitions = max(0, len(path) - 1)
        if transitions == 0:
            self.od_uncertainty_report = {
                'path_length': 0,
                'risky_transitions': 0,
                'coverage_ratio': 0.0,
                'added_transitions': 0,
            }
            return

        risky_before, coverage_before = self._let_uncertainty_exposure(path)
        target_count = int(np.ceil(float(target_ratio) * transitions))
        max_count = int(np.floor(float(max_ratio) * transitions))
        target_count = max(1, target_count)
        if max_count > 0:
            target_count = min(target_count, max_count)

        added = 0
        if coverage_before < target_ratio and risky_before < target_count:
            candidates = [
                (idx, u, v)
                for idx, (u, v) in enumerate(zip(path[:-1], path[1:]))
                if u != self.dest
                and len(self.successors.get(u, [])) >= 2
                and u not in self.uncertain_edges
            ]
            rng = np.random.default_rng(seed + self.origin * 1000003 + self.dest)
            if candidates:
                order = rng.permutation(len(candidates))
                need = target_count - risky_before
                for pos in order[:need]:
                    _, u, v = candidates[int(pos)]
                    self.uncertain_edges[u] = v
                    added += 1

        risky_after, coverage_after = self._let_uncertainty_exposure(path)
        self.od_uncertainty_report = {
            'path_length': len(path),
            'risky_transitions': risky_after,
            'coverage_ratio': coverage_after,
            'added_transitions': added,
        }

    def _important_nodes(self, top_k):
        degree = defaultdict(int)
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
        ranked = sorted(
            (n for n in self.nodes if n != self.origin and n != self.dest),
            key=lambda n: degree[n], reverse=True
        )
        return ranked[:top_k]

    def is_terminal(self, node, budget):
        return node == self.dest or budget < 0

    def get_actions(self, node):
        return list(self.successors.get(node, []))

    def sample_travel_time(self, u, v):
        mean, sigma = self.edges[(u, v)]
        if self.deterministic or sigma <= 0:
            return max(1, round(mean))
        cv2 = (sigma / mean) ** 2
        mu_ln = np.log(mean) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        t = self.rng.lognormal(mu_ln, sigma_ln)
        return max(1, round(t))

    def sample_executed_action(self, node, intended_action, p_intended=None):
        prob = p_intended if p_intended is not None else self.exec_prob
        if node not in self.uncertain_edges:
            return intended_action
        spec = self.uncertain_edges.get(node, None)
        if isinstance(spec, dict):
            risky_intended = spec['intended']
            if intended_action != risky_intended:
                return intended_action
            if self.rng.random() < prob:
                return intended_action
            alternatives = [n for n in self.successors[node] if n != intended_action]
            if not alternatives:
                return intended_action
            return int(self.rng.choice(alternatives))
        if intended_action != spec:
            return intended_action
        if self.rng.random() < prob:
            return intended_action
        alternatives = [n for n in self.successors[node] if n != intended_action]
        if not alternatives:
            return intended_action
        return int(self.rng.choice(alternatives))
