"""Chengdu network environment adapter for routing algorithms."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import pandas as pd


NETWORK_DIR = Path(__file__).resolve().parent.parent / "network" / "Chengdu"

PERIOD_FILES = {
    "weekday_peak": "Weekday_Peak_network.csv",
    "weekday_offpeak": "Weekday_Offpeak_network.csv",
    "weekend_peak": "Weekend_Peak_network.csv",
    "weekend_offpeak": "Weekend_Offpeak_network.csv",
}


def normalize_period(period: str) -> str:
    """Return a validated canonical Chengdu period key."""
    key = str(period).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PERIOD_FILES:
        raise ValueError(f"Unsupported Chengdu period {period!r}; expected one of {sorted(PERIOD_FILES)}")
    return key


class ChengduEnv:
    """Directed stochastic routing environment backed by Chengdu CSV data.

    Edge means retain their source unit (seconds).  Execution uncertainty is
    opt-in: it applies only at configured uncertain nodes and only when eta is
    positive.  The default contains no uncertain nodes, which is appropriate
    until a later uncertainty-design phase supplies an explicit configuration.
    """

    def __init__(
        self,
        origin: int,
        destination: int | None = None,
        budget: float = 0.0,
        eta: float = 0.0,
        period: str = "weekday_peak",
        seed: int | None = None,
        dest: int | None = None,
        uncertain_nodes: list[int] | None = None,
        uncertain_edges_explicit: dict[int, int] | None = None,
    ) -> None:
        if destination is None:
            destination = dest
        if destination is None:
            raise ValueError("destination (or dest) is required")
        if not 0.0 <= float(eta) <= 1.0:
            raise ValueError("eta must be in [0, 1]")

        self.period = normalize_period(period)
        self.origin = int(origin)
        self.dest = int(destination)
        self.destination = self.dest
        self.budget = float(budget)
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)

        self.edges: dict[tuple[int, int], tuple[float, float]] = {}
        self.successors: defaultdict[int, list[int]] = defaultdict(list)
        self._load_network()
        if self.origin not in self.nodes or self.dest not in self.nodes:
            raise ValueError("origin and destination must both be Chengdu network nodes")

        self.uncertain_edges = self._configure_uncertainty(
            uncertain_nodes=uncertain_nodes,
            explicit=uncertain_edges_explicit,
        )
        self.reset()

    def _load_network(self) -> None:
        frame = pd.read_csv(NETWORK_DIR / PERIOD_FILES[self.period])
        required = {"From", "To", "Cost", "Var"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required Chengdu network columns: {sorted(missing)}")

        for row in frame.itertuples(index=False):
            u, v = int(row.From), int(row.To)
            key = (u, v)
            if key in self.edges:
                raise ValueError(f"Duplicate directed edge in Chengdu network: {u}->{v}")
            mean = float(row.Cost)
            sigma = math.sqrt(max(float(row.Var), 0.0))
            if not math.isfinite(mean) or not math.isfinite(sigma):
                raise ValueError(f"Non-finite edge data for {u}->{v}")
            self.edges[key] = (mean, sigma)
            self.successors[u].append(v)

        self.nodes = {node for edge in self.edges for node in edge}

    def _configure_uncertainty(
        self,
        uncertain_nodes: list[int] | None,
        explicit: dict[int, int] | None,
    ) -> dict[int, int]:
        if explicit is not None:
            result = {int(u): int(v) for u, v in explicit.items()}
            for u, v in result.items():
                if v not in self.successors.get(u, []):
                    raise ValueError(f"Uncertain intended edge is not valid: {u}->{v}")
            return result

        result: dict[int, int] = {}
        for u in uncertain_nodes or []:
            actions = self.successors.get(int(u), [])
            if actions:
                result[int(u)] = int(actions[0])
        return result

    def get_actions(self, node: int) -> list[int]:
        """Return valid successor actions for a node."""
        return list(self.successors.get(int(node), []))
    def shortest_path(self, origin: int | None = None, destination: int | None = None) -> tuple[float | None, list[int]]:
        """Return the minimum-mean travel time and path using Dijkstra's algorithm."""
        import heapq
        start = self.origin if origin is None else int(origin)
        target = self.dest if destination is None else int(destination)
        queue = [(0.0, start, [start])]
        best = {start: 0.0}
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == target:
                return cost, path
            if cost != best.get(node):
                continue
            for nxt in self.successors.get(node, []):
                candidate = cost + self.edges[(node, nxt)][0]
                if candidate < best.get(nxt, float("inf")):
                    best[nxt] = candidate
                    heapq.heappush(queue, (candidate, nxt, path + [nxt]))
        return None, []
    def reset(self) -> tuple[int, float]:
        """Reset the state and return the initial node and remaining budget."""
        self.current_node = self.origin
        self.remaining_budget = self.budget
        self.done = self.current_node == self.dest
        return self.current_node, self.remaining_budget

    def sample_executed_action(
        self,
        node: int,
        intended_action: int,
        intended_probability: float | None = None,
    ) -> int:
        """Sample the executed action after a valid intended action is chosen."""
        node, intended_action = int(node), int(intended_action)
        actions = self.successors.get(node, [])
        if intended_action not in actions:
            raise ValueError(f"Invalid intended action {node}->{intended_action}")
        if self.eta == 0.0 or node not in self.uncertain_edges or self.rng.random() >= self.eta:
            return intended_action

        alternatives = [action for action in actions if action != intended_action]
        if not alternatives:
            return intended_action
        return int(self.rng.choice(alternatives))

    def sample_travel_time(self, node: int, action: int) -> float:
        """Sample a non-negative edge travel time using mean and sigma."""
        mean, sigma = self.edges[(int(node), int(action))]
        return float(max(0.0, self.rng.normal(mean, sigma)))

    def step(self, intended_action: int) -> tuple[tuple[int, float], float, bool]:
        """Advance one transition and return state, sampled travel time, and done."""
        if self.done:
            return (self.current_node, self.remaining_budget), 0.0, True
        actual = self.sample_executed_action(self.current_node, intended_action)
        travel_time = self.sample_travel_time(self.current_node, actual)
        self.current_node = actual
        self.remaining_budget -= travel_time
        self.done = self.current_node == self.dest or self.remaining_budget < 0.0
        return (self.current_node, self.remaining_budget), travel_time, self.done
