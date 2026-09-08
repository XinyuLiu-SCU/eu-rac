"""
evaluator.py  Monte Carlo policy evaluator for SiouxEnv.

Evaluates any deterministic policy under the true environment dynamics,
including execution uncertainty and stochastic travel times.
"""

import numpy as np


def evaluate_policy(env, policy_func, episodes=10000):
    """
    Monte Carlo evaluation of a policy in SiouxEnv with execution uncertainty.

    Parameters
    ----------
    env         : SiouxEnv instance (execution uncertainty already configured)
    policy_func : callable(node, budget) -> next_node
                  Deterministic policy. Return None if no action available.
    episodes    : number of Monte Carlo rollouts

    Returns
    -------
    float : empirical on-time arrival probability
    """
    max_steps = len(env.nodes) * 3
    successes = 0

    for _ in range(episodes):
        node = env.origin
        budget = env.budget

        for _ in range(max_steps):
            if node == env.dest:
                if budget >= 0:
                    successes += 1
                break
            if budget < 0:
                break

            intended = policy_func(node, budget)
            if intended is None:
                break

            # Apply execution uncertainty via env
            actual, execution_delay = env.sample_execution_transition(node, intended)
            travel_time = env.sample_travel_time(node, actual) + execution_delay
            budget -= travel_time
            node = actual
        else:
            # max_steps exceeded  treat as failure
            pass

    return successes / episodes
