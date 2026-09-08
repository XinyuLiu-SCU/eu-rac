"""Algorithm dispatchers for public EU-RAC experiments."""

from __future__ import annotations

from typing import Any

from evaluation import greedy_policy_from_agent, evaluate_policy


def normalize_algorithm(name: str) -> str:
    """Return a canonical public algorithm key."""
    key = str(name).strip().lower().replace("_", "-")
    aliases = {
        "eurac": "eurac",
        "eu-rac": "eurac",
        "neural": "eurac",
        "eurac-tabular": "eurac-tabular",
        "eu-rac-tabular": "eurac-tabular",
        "tabular": "eurac-tabular",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported algorithm {name!r}. This public entry point supports eurac and eurac-tabular.")
    return aliases[key]


def train_agent(algorithm: str, env, episodes: int = 1000, **kwargs: Any):
    """Train a public EU-RAC agent and return it."""
    alg = normalize_algorithm(algorithm)
    if alg == "eurac-tabular":
        from eu_rac_tabular import EURAC
        agent = EURAC(env, **kwargs)
        agent.warm_start()
        agent.train(n_episodes=episodes)
        return agent

    from eu_rac import EURAC
    agent = EURAC(env, **kwargs)
    agent.warm_start()
    agent.train(n_episodes=episodes, resume=False)
    return agent


def evaluate_agent(agent, env, episodes: int = 1000) -> float:
    """Evaluate an agent through its greedy policy."""
    return evaluate_policy(env, greedy_policy_from_agent(agent), episodes=episodes)
