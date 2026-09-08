"""Evaluation helpers for routing policies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Policy = Callable[[int, float], int | None]


def evaluate_policy(env, policy: Policy, episodes: int = 1000) -> float:
    """Estimate on-time arrival probability with Monte Carlo rollouts."""
    successes = []
    max_steps = max(1, len(getattr(env, "nodes", [])) * 2)
    for _ in range(episodes):
        node = env.origin
        budget = env.budget
        success = 0.0
        for _step in range(max_steps):
            if node == env.dest:
                success = 1.0 if budget >= 0 else 0.0
                break
            if budget < 0:
                break
            action = policy(node, budget)
            if action is None:
                break
            actual = env.sample_executed_action(node, action)
            travel_time = env.sample_travel_time(node, actual)
            node = actual
            budget -= travel_time
        successes.append(success)
    return float(np.mean(successes)) if successes else 0.0


def greedy_policy_from_agent(agent):
    """Return a deterministic policy that chooses the highest-probability action."""
    def _policy(node: int, budget: float):
        probs, actions = agent.get_policy((node, budget))
        if not actions:
            return None
        return max(probs, key=probs.get)
    return _policy
