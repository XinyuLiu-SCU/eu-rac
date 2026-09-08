"""Command-line entry point for compact EU-RAC experiments."""

from __future__ import annotations

import argparse

from benchmark import evaluate_agent, train_agent
from env import make_env, normalize_network
from func import read_od_pairs


DEFAULT_OD = {
    "sioux": (2, 15, 60),
    "anaheim": (1, 2, 120),
    "barcelona": (203, 374, 120),
    "chicago": (649, 322, 120),
    "beijing": (311, 529, 1200),
    "chengdu": (1, 2, 1200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact EU-RAC smoke experiment.")
    parser.add_argument("--network", default="sioux", help="Benchmark network name.")
    parser.add_argument("--algorithm", default=None, help="Use eurac or eurac-tabular.")
    parser.add_argument("--origin", type=int, default=None, help="Origin node. Overrides --od-file.")
    parser.add_argument("--destination", "--dest", type=int, default=None, help="Destination node. Overrides --od-file.")
    parser.add_argument("--budget", type=float, default=None, help="Travel-time budget.")
    parser.add_argument("--od-file", default=None, help="CSV file containing selected OD pairs.")
    parser.add_argument("--od-index", type=int, default=0, help="OD row index when --od-file is provided.")
    parser.add_argument("--period", default="weekday_peak", help="Period key for Beijing and Chengdu.")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=1000, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to the environment.")
    parser.add_argument("--eta", type=float, default=0.2, help="Execution uncertainty probability for period networks.")
    parser.add_argument("--exec-prob", type=float, default=0.8, help="Intended-action execution probability.")
    return parser.parse_args()


def choose_od(args: argparse.Namespace, network: str) -> tuple[int, int, float]:
    """Resolve origin, destination, and budget from CLI arguments."""
    default_origin, default_dest, default_budget = DEFAULT_OD[network]
    origin = args.origin
    destination = args.destination
    if args.od_file and (origin is None or destination is None):
        pairs = read_od_pairs(args.od_file)
        if not pairs:
            raise ValueError(f"No OD pairs found in {args.od_file}")
        origin, destination = pairs[args.od_index]
    origin = default_origin if origin is None else origin
    destination = default_dest if destination is None else destination
    budget = default_budget if args.budget is None else args.budget
    return int(origin), int(destination), int(round(budget))


def main() -> int:
    args = parse_args()
    network = normalize_network(args.network)
    algorithm = args.algorithm or ("eurac-tabular" if network == "sioux" else "eurac")
    origin, destination, budget = choose_od(args, network)

    env_kwargs = {"seed": args.seed}
    if network in {"beijing", "chengdu"}:
        env_kwargs.update({"period": args.period, "eta": args.eta})
    else:
        env_kwargs.update({"exec_prob": args.exec_prob})

    train_env = make_env(network, origin, destination, budget, **env_kwargs)
    agent_kwargs = {"p_intended": args.exec_prob if network != "beijing" and network != "chengdu" else max(0.0, 1.0 - args.eta)}
    agent = train_agent(algorithm, train_env, episodes=args.episodes, **agent_kwargs)

    eval_env = make_env(network, origin, destination, budget, **env_kwargs)
    probability = evaluate_agent(agent, eval_env, episodes=args.eval_episodes)
    print(f"network={network} algorithm={algorithm} od={origin}->{destination} budget={budget} probability={probability:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
