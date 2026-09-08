"""
evaluator.py ,Monte Carlo policy evaluator for SiouxEnv.

Evaluates any deterministic policy under the true environment dynamics,
including execution uncertainty and stochastic travel times.
"""

import inspect
import numpy as np


def evaluate_policy(env, policy_func, episodes=10000):
    """
    Monte Carlo evaluation of a policy in SiouxEnv with execution uncertainty.

    Parameters
    ----------
    env         : SiouxEnv instance (execution uncertainty already configured)
    policy_func : callable(node, budget[, visited]) -> next_node
                  Deterministic policy. Return None if no action available.
                  If the callable accepts a third argument, the evaluator
                  passes the set of already visited nodes in this rollout.
    episodes    : number of Monte Carlo rollouts

    Returns
    -------
    float : empirical on-time arrival probability
    """
    max_steps = len(env.nodes) * 3
    successes = 0
    try:
        n_params = len(inspect.signature(policy_func).parameters)
    except (TypeError, ValueError):
        n_params = 2
    pass_visited = n_params >= 3

    for _ in range(episodes):
        node = env.origin
        budget = env.budget
        visited = {node}

        for _ in range(max_steps):
            if node == env.dest:
                if budget >= 0:
                    successes += 1
                break
            if budget < 0:
                break

            if pass_visited:
                intended = policy_func(node, budget, visited)
            else:
                intended = policy_func(node, budget)
            if intended is None:
                break

            # Apply execution uncertainty via env
            actual = env.sample_executed_action(node, intended)
            travel_time = env.sample_travel_time(node, actual)
            budget -= travel_time
            node = actual
            visited.add(node)
        else:
            # max_steps exceeded - treat as failure
            pass

    return successes / episodes