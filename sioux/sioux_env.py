import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

NETWORK_DIR = Path(__file__).parent.parent / 'Networks' / 'Networks' / 'SiouxFalls'


class SiouxEnv:
    def __init__(self, origin=2, dest=15, budget=60,
                 exec_prob=0.8, uncertain_ratio=0.2,
                 deterministic=False, seed=42, uncertain_seed=42,
                 uncertain_mode='random', top_k=6,
                 uncertain_nodes=None,
                 uncertain_edges_explicit=None):
        """
        uncertain_nodes : list[int] | None
            If provided, use exactly these nodes as uncertain nodes (fixed mode).
            Overrides uncertain_ratio, uncertain_seed, uncertain_mode, top_k.

        uncertain_edges_explicit : dict[int, int] | None
            Directly specify {node: risky_next_node} pairs.
            E.g. {3: 4, 9: 10} means: at node 3 the edge 3->4 is risky,
            at node 9 the edge 9->10 is risky.
            Takes highest priority  overrides all other uncertainty settings.
        """
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

    # ------------------------------------------------------------------
    # Network loading
    # ------------------------------------------------------------------

    def _load_network(self):
        net_df = pd.read_csv(NETWORK_DIR / 'SiouxFalls_network.csv')
        sigma_arr = np.load(NETWORK_DIR / 'SiouxFalls_0.4_random_sigma.npy')

        self.edges = {}          # (u, v) -> (mean_time, sigma)
        self.successors = defaultdict(list)

        for i, row in net_df.iterrows():
            u, v = int(row['From']), int(row['To'])
            mean_t = float(row['Cost'])
            sigma = float(sigma_arr[i])
            self.edges[(u, v)] = (mean_t, sigma)
            self.successors[u].append(v)

        self.nodes = set()
        for u, v in self.edges:
            self.nodes.add(u)
            self.nodes.add(v)

    # ------------------------------------------------------------------
    # Execution uncertainty setup
    # ------------------------------------------------------------------

    def _important_nodes(self, top_k):
        """Return top_k nodes ranked by approximate betweenness (in+out degree)."""
        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_edges_from(self.edges.keys())
            bc = nx.betweenness_centrality(G, normalized=True)
            # exclude origin and dest so uncertainty doesn't trivially block the path
            ranked = sorted(
                (n for n in bc if n != self.origin and n != self.dest),
                key=lambda n: bc[n], reverse=True
            )
            return ranked[:top_k]
        except ImportError:
            # fallback: rank by in+out degree
            degree = defaultdict(int)
            for u, v in self.edges:
                degree[u] += 1
                degree[v] += 1
            ranked = sorted(
                (n for n in self.nodes if n != self.origin and n != self.dest),
                key=lambda n: degree[n], reverse=True
            )
            return ranked[:top_k]

    def _setup_explicit_uncertainty(self, edges_dict):
        """Directly use caller-specified {node: risky_next_node} pairs."""
        self.uncertain_mode = 'explicit'
        self.uncertain_edges = {}
        for node, next_node in edges_dict.items():
            if node not in self.nodes:
                raise ValueError(f"uncertain_edges_explicit: node {node} not in network")
            if next_node not in self.successors.get(node, []):
                raise ValueError(f"uncertain_edges_explicit: edge {node}->{next_node} does not exist")
            self.uncertain_edges[node] = next_node

    def _setup_fixed_uncertainty(self, uncertain_nodes):
        """Use a caller-specified list of nodes as uncertain nodes."""
        rng = np.random.default_rng(0)  # deterministic risky-edge assignment
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
        self.uncertain_edges = {}   # node -> risky_next_node
        for node in chosen:
            succs = self.successors[node]
            if succs:
                self.uncertain_edges[node] = int(rng.choice(succs))

    # ------------------------------------------------------------------
    # Public interface (mirrors simple/network.py)
    # ------------------------------------------------------------------

    def is_terminal(self, node, budget):
        return node == self.dest or budget < 0

    def get_actions(self, node):
        return list(self.successors.get(node, []))

    def sample_travel_time(self, u, v):
        mean, sigma = self.edges[(u, v)]
        if self.deterministic or sigma <= 0:
            return max(1, round(mean))
        # Lognormal parameterisation from (mean, sigma)
        cv2 = (sigma / mean) ** 2
        mu_ln = np.log(mean) - 0.5 * np.log(1.0 + cv2)
        sigma_ln = np.sqrt(np.log(1.0 + cv2))
        t = self.rng.lognormal(mu_ln, sigma_ln)
        return max(1, round(t))

    def sample_executed_action(self, node, intended_action, p_intended=None):
        # 
        prob = p_intended if p_intended is not None else self.exec_prob

        # 
        if node not in self.uncertain_edges:
            return intended_action

        # 
        if intended_action != self.uncertain_edges.get(node, None):
            return intended_action

        #  prob 
        if self.rng.random() < prob:
            return intended_action

        # 
        alternatives = [n for n in self.successors[node] if n != intended_action]
        if not alternatives:
            return intended_action

        return int(self.rng.choice(alternatives))

    # ------------------------------------------------------------------
    # Helper: default OD (first row of OD file  no demand column present)
    # ------------------------------------------------------------------

    @classmethod
    def default_od(cls):
        public_od = Path(__file__).resolve().parent.parent / 'data' / 'sioux' / 'od_pairs.csv'
        od_df = pd.read_csv(public_od)
        row = od_df.iloc[0]
        return int(row['origin']), int(row['destination'])

