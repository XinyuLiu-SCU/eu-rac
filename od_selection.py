"""OD pair screening utility for EU-RAC benchmark experiments."""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path

from env import make_env, normalize_network


DEFAULT_BUDGET = {
    "sioux": 60,
    "anaheim": 120,
    "barcelona": 120,
    "chicago": 120,
    "beijing": 1200,
    "chengdu": 1200,
}


def shortest_path_cost(env, origin: int, destination: int) -> float:
    """Return mean shortest-path cost, or infinity if disconnected."""
    dist = {origin: 0.0}
    pq = [(0.0, origin)]
    while pq:
        cost, node = heapq.heappop(pq)
        if node == destination:
            return cost
        if cost > dist.get(node, float("inf")):
            continue
        for nxt in env.successors.get(node, []):
            mean_t, _ = env.edges[(node, nxt)]
            next_cost = cost + max(1.0, float(mean_t))
            if next_cost < dist.get(nxt, float("inf")):
                dist[nxt] = next_cost
                heapq.heappush(pq, (next_cost, nxt))
    return float("inf")


def screen_od_pairs(network: str, budget: float, limit: int, period: str = "weekday_peak") -> list[dict[str, float]]:
    """Select connected OD pairs with feasible but non-trivial mean travel time."""
    key = normalize_network(network)
    seed_origin, seed_dest = (2, 15) if key == "sioux" else (None, None)
    if seed_origin is None:
        probe = make_env(key, origin=1, destination=2, budget=budget, period=period) if key in {"beijing", "chengdu"} else make_env(key, origin=1, destination=2, budget=budget)
    else:
        probe = make_env(key, origin=seed_origin, destination=seed_dest, budget=budget)

    nodes = sorted(int(n) for n in probe.nodes)
    rows: list[dict[str, float]] = []
    min_cost = 0.2 * budget
    max_cost = 0.95 * budget

    for origin in nodes:
        if len(probe.successors.get(origin, [])) < 1:
            continue
        for destination in nodes:
            if origin == destination:
                continue
            cost = shortest_path_cost(probe, origin, destination)
            if min_cost <= cost <= max_cost:
                rows.append({"origin": origin, "destination": destination, "shortest_path_cost": round(cost, 6)})
            if len(rows) >= limit:
                return rows
    return rows


def write_od_pairs(rows: list[dict[str, float]], output: str | Path) -> None:
    """Write selected OD pairs to CSV."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["origin", "destination", "shortest_path_cost"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen OD pairs for EU-RAC benchmark experiments.")
    parser.add_argument("--network", required=True, help="Benchmark network name.")
    parser.add_argument("--budget", type=float, default=None, help="Budget used for feasibility screening.")
    parser.add_argument("--limit", type=int, default=50, help="Number of OD pairs to export.")
    parser.add_argument("--period", default="weekday_peak", help="Period key for Beijing and Chengdu.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    network = normalize_network(args.network)
    budget = DEFAULT_BUDGET[network] if args.budget is None else args.budget
    rows = screen_od_pairs(network, budget=budget, limit=args.limit, period=args.period)
    write_od_pairs(rows, args.output)
    print(f"wrote {len(rows)} OD pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
