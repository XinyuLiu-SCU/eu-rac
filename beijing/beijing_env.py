"""BeijingEnv: Beijing period-specific environment using raw-second data."""

from __future__ import annotations

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NETWORK_DIR = Path(__file__).parent.parent / "network" / "Beijing"

PERIOD_FILES = {
    "weekday_peak": "Beijing_Weekday_peak.csv",
    "weekday_offpeak": "Beijing_Weekday_Offpeak.csv",
    "weekend_peak": "Beijing_Weekend_Peak.csv",
    "weekend_offpeak": "Beijing_Weekend_Offpeak.csv",
}

REQUIRED_COLUMNS = {"From", "To", "Cost", "Var"}


def normalize_period(period: str) -> str:
    key = str(period).strip().lower()
    if key not in PERIOD_FILES:
        valid = ", ".join(PERIOD_FILES)
        raise ValueError(f"Unsupported Beijing period {period!r}. Supported periods: {valid}")
    return key


class BeijingEnv:
    """Directed Beijing road environment.

    Time convention:
    - Cost is raw seconds.
    - Var is raw seconds squared.
    - Sigma is sqrt(max(Var, 0)).
    - Budgets are integer seconds.

    Phase 1 intentionally does not create random uncertainty placements.
    Use eta=0, or pass explicit uncertain edges for later controlled tests.
    """

    def __init__(
        self,
        origin: int | None = None,
        destination: int | None = None,
        budget: int | None = None,
        eta: float = 0.0,
        period: str = "weekday_peak",
        seed: int | None = None,
        *,
        dest: int | None = None,
        exec_prob: float | None = None,
        deterministic: bool = False,
        uncertain_edges_explicit: dict | None = None,
        uncertain_nodes: Iterable[int] | None = None,
        **_unused,
    ):
        if destination is None and dest is not None:
            destination = dest
        self.origin = None if origin is None else int(origin)
        self.dest = None if destination is None else int(destination)
        self.destination = self.dest
        self.budget = None if budget is None else int(budget)
        self.remaining_budget = self.budget
        self.eta = float(eta)
        self.exec_prob = float(exec_prob) if exec_prob is not None else 1.0 - self.eta
        self.period = normalize_period(period)
        self.deterministic = bool(deterministic)
        self.rng = np.random.default_rng(seed)

        self._load_network()
        self.od_pairs = self._load_od_pairs()

        if uncertain_edges_explicit is not None:
            self._setup_explicit_uncertainty(uncertain_edges_explicit)
        elif uncertain_nodes is not None:
            self._setup_fixed_uncertainty(uncertain_nodes)
        elif self.eta == 0:
            self.uncertain_mode = "none"
            self.uncertain_edges = {}
        else:
            raise ValueError(
                "BeijingEnv Phase 1 does not fabricate random uncertain edges. "
                "Use eta=0 or pass uncertain_edges_explicit."
            )

    def _load_network(self) -> None:
        path = NETWORK_DIR / PERIOD_FILES[self.period]
        df = pd.read_csv(path)
        df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        self.dataframe = df.copy()
        self.edges = {}
        self.mean_time = {}
        self.sigma = {}
        self.variance = {}
        self.successors = defaultdict(list)
        self.adj = self.successors
        self.edge_list = []

        seen = set()
        self.self_loops = 0
        self.duplicate_edges = 0
        for row in df.itertuples(index=False):
            u, v = int(row.From), int(row.To)
            if u == v:
                self.self_loops += 1
            if (u, v) in seen:
                self.duplicate_edges += 1
            seen.add((u, v))
            mean_t = float(row.Cost)
            var = max(float(row.Var), 0.0)
            sigma = float(np.sqrt(var))
            self.edge_list.append((u, v))
            self.mean_time[(u, v)] = mean_t
            self.variance[(u, v)] = var
            self.sigma[(u, v)] = sigma
            self.edges[(u, v)] = (mean_t, sigma)
            self.successors[u].append(v)

        self.nodes = set()
        for u, v in self.edge_list:
            self.nodes.add(u)
            self.nodes.add(v)

    def _load_od_pairs(self) -> np.ndarray:
        pairs = np.load(NETWORK_DIR / "Beijing_Pairs.npy", allow_pickle=True)
        pairs = np.asarray(pairs, dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"Beijing_Pairs.npy must have shape (n, 2), got {pairs.shape}")
        return pairs

    def validate_od_pairs(self) -> list[dict]:
        rows = []
        for origin, dest in self.od_pairs:
            origin = int(origin)
            dest = int(dest)
            cost, path = self.shortest_path(origin, dest)
            rows.append(
                {
                    "origin": origin,
                    "dest": dest,
                    "origin_exists": origin in self.nodes,
                    "dest_exists": dest in self.nodes,
                    "reachable": cost is not None,
                    "let": cost,
                    "hop_count": None if not path else len(path) - 1,
                    "path": path,
                }
            )
        return rows

    def _setup_explicit_uncertainty(self, edges_dict: dict) -> None:
        self.uncertain_mode = "explicit"
        self.uncertain_edges = {}
        self.uncertain_alternatives = {}
        self.uncertain_deviation_sets = {}
        self.execution_failure_delays = {}
        for node, spec in edges_dict.items():
            node = int(node)
            intended = int(spec["intended"]) if isinstance(spec, dict) else int(spec)
            if node not in self.nodes:
                raise ValueError(f"uncertain_edges_explicit: node {node} not in network")
            if intended not in self.successors.get(node, []):
                raise ValueError(f"uncertain_edges_explicit: edge {node}->{intended} does not exist")
            self.uncertain_edges[node] = intended
            alternative = spec.get("alternative") if isinstance(spec, dict) else None
            if alternative is not None:
                alternative = int(alternative)
                if alternative not in self.successors.get(node, []):
                    raise ValueError(f"uncertain_edges_explicit: alternative edge {node}->{alternative} does not exist")
                if alternative == intended:
                    raise ValueError("uncertain_edges_explicit: alternative must differ from intended")
                self.uncertain_alternatives[(node, intended)] = alternative
            deviation_successors = spec.get('deviation_successors') if isinstance(spec, dict) else None
            if deviation_successors is not None:
                choices = [int(v) for v in deviation_successors]
                if not choices or intended in choices or any(v not in self.successors.get(node, []) for v in choices):
                    raise ValueError('uncertain_edges_explicit: invalid deviation_successors')
                self.uncertain_deviation_sets[(node, intended)] = tuple(dict.fromkeys(choices))
            failure_delay = float(spec.get('failure_delay', 0.0)) if isinstance(spec, dict) else 0.0
            if failure_delay < 0:
                raise ValueError('uncertain_edges_explicit: failure_delay must be nonnegative')
            self.execution_failure_delays[(node, intended)] = failure_delay

    def _setup_fixed_uncertainty(self, uncertain_nodes: Iterable[int]) -> None:
        self.uncertain_mode = "fixed"
        self.uncertain_edges = {}
        self.uncertain_alternatives = {}
        self.uncertain_deviation_sets = {}
        self.execution_failure_delays = {}
        for node in uncertain_nodes:
            node = int(node)
            succs = self.successors.get(node, [])
            if succs:
                self.uncertain_edges[node] = int(succs[0])

    def get_actions(self, node: int) -> list[int]:
        return list(self.successors.get(int(node), []))

    def get_outgoing_edges(self, node: int) -> list[tuple[int, int]]:
        node = int(node)
        return [(node, v) for v in self.successors.get(node, [])]

    def is_terminal(self, node: int, budget: float) -> bool:
        return int(node) == self.dest or budget < 0

    def reset(self, origin: int | None = None, budget: int | None = None) -> tuple[int | None, int | None]:
        if origin is not None:
            self.origin = int(origin)
        if budget is not None:
            self.budget = int(budget)
        self.remaining_budget = self.budget
        return self.origin, self.remaining_budget

    def step(self, intended_action: int):
        if self.origin is None or self.remaining_budget is None:
            raise ValueError("step requires origin and budget")
        actual, execution_delay = self.sample_execution_transition(self.origin, intended_action)
        travel_time = self.sample_travel_time(self.origin, actual) + execution_delay
        self.remaining_budget -= travel_time
        self.origin = actual
        done = self.dest is not None and (self.origin == self.dest or self.remaining_budget < 0)
        reward = 1.0 if done and self.origin == self.dest and self.remaining_budget >= 0 else 0.0
        return self.origin, self.remaining_budget, reward, done, {"actual_action": actual, "travel_time": travel_time}

    def sample_travel_time(self, u: int, v: int) -> float:
        mean_t, sigma = self.edges[(int(u), int(v))]
        if self.deterministic or sigma <= 0:
            return max(1.0, mean_t)
        sampled = self.rng.normal(mean_t, sigma)
        return float(max(1.0, sampled))

    def sample_execution_transition(self, node: int, intended_action: int,
                                    p_intended: float | None = None) -> tuple[int, float]:
        """Sample executed successor and the delay caused by an execution failure."""
        node = int(node)
        intended_action = int(intended_action)
        prob = self.exec_prob if p_intended is None else float(p_intended)
        risky = self.uncertain_edges.get(node)
        if risky is None or intended_action != risky:
            return intended_action, 0.0
        if self.rng.random() < prob:
            return intended_action, 0.0
        deviation_set = self.uncertain_deviation_sets.get((node, intended_action))
        if deviation_set:
            actual = int(self.rng.choice(deviation_set))
        else:
            mapped_alternative = self.uncertain_alternatives.get((node, intended_action))
            if mapped_alternative is not None:
                actual = mapped_alternative
            else:
                alternatives = [v for v in self.successors[node] if v != intended_action]
                actual = int(self.rng.choice(alternatives)) if alternatives else intended_action
        return actual, self.execution_failure_delays.get((node, intended_action), 0.0)

    def sample_executed_action(self, node: int, intended_action: int,
                               p_intended: float | None = None) -> int:
        """Backward-compatible successor-only execution sample."""
        return self.sample_execution_transition(node, intended_action, p_intended)[0]

    def shortest_path(self, origin: int | None = None, dest: int | None = None) -> tuple[float | None, list[int]]:
        origin = self.origin if origin is None else int(origin)
        dest = self.dest if dest is None else int(dest)
        if origin is None or dest is None:
            raise ValueError("shortest_path requires origin and destination")
        dist = {origin: 0.0}
        prev = {}
        pq = [(0.0, origin)]
        while pq:
            cost, node = heapq.heappop(pq)
            if cost != dist.get(node):
                continue
            if node == dest:
                break
            for nxt in self.successors.get(node, []):
                new_cost = cost + self.mean_time[(node, nxt)]
                if new_cost < dist.get(nxt, float("inf")):
                    dist[nxt] = new_cost
                    prev[nxt] = node
                    heapq.heappush(pq, (new_cost, nxt))
        if dest not in dist:
            return None, []
        path = [dest]
        node = dest
        while node != origin:
            node = prev.get(node)
            if node is None:
                return None, []
            path.append(node)
        path.reverse()
        return dist[dest], path

    def least_expected_time(self, origin: int | None = None, dest: int | None = None) -> float | None:
        cost, _ = self.shortest_path(origin, dest)
        return cost

    def least_expected_time_path(self, origin: int | None = None, dest: int | None = None) -> list[int]:
        _, path = self.shortest_path(origin, dest)
        return path



