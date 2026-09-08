import numpy as np
from collections import defaultdict


class EURAC:
    def __init__(self, env, p_intended=0.8, lr_e=0.2, lr_d=0.2, lr_actor=0.1, entropy_coef=0.02):
        self.env = env
        self.p_intended = p_intended
        self.lr_e = lr_e
        self.lr_d = lr_d
        self.lr_actor = lr_actor
        self.entropy_coef = entropy_coef

        self.Q_e = defaultdict(float)
        self.Q_d = defaultdict(float)
        self.theta = defaultdict(float)

    def warm_start(self, logit: float = 4.0):
        """Initialize policy logits with the mean-travel-time Dijkstra path."""
        import heapq
        edges = self.env.edges
        successors = self.env.successors

        dist = {self.env.origin: 0.0}
        prev = {}
        pq = [(0.0, self.env.origin)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            for v in successors.get(u, []):
                mean_t, _ = edges[(u, v)]
                nd = d + mean_t
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if self.env.dest not in prev:
            return

        path = []
        n = self.env.dest
        while n in prev:
            path.append(n)
            n = prev[n]
        path.append(self.env.origin)
        path.reverse()

        remaining = self.env.budget
        for i in range(len(path) - 1):
            node = path[i]
            action = path[i + 1]
            state = (node, remaining)
            self.theta[(state, action)] = logit
            mean_t, _ = edges[(node, action)]
            remaining -= round(mean_t)

    def get_policy(self, state):
        """Return a probability dictionary and feasible actions for the state."""
        actions = self.env.get_actions(state[0])
        if not actions:
            return {}, []
        logits = np.array([self.theta[(state, a)] for a in actions], dtype=float)
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        return dict(zip(actions, probs)), actions

    def select_action(self, state):
        policy, actions = self.get_policy(state)
        if not actions:
            return None
        probs = [policy[a] for a in actions]
        return int(np.random.choice(actions, p=probs))

    def compute_V(self, state):
        node, budget = state
        if node == self.env.dest:
            return 1.0 if budget >= 0 else 0.0
        if budget < 0:
            return 0.0
        policy, actions = self.get_policy(state)
        if not actions:
            return 0.0
        return sum(policy[a] * self.Q_d[(state, a)] for a in actions)

    def env_step(self, state, intended_action):
        node, budget = state
        actual = self.env.sample_executed_action(node, intended_action, self.p_intended)
        travel_time = self.env.sample_travel_time(node, actual)
        next_state = (actual, budget - travel_time)
        return actual, next_state

    def update(self, state, intended_action, actual_action, next_state):
        next_node, next_budget = next_state

        if next_node == self.env.dest and next_budget >= 0:
            y_e = 1.0
        elif next_budget < 0:
            y_e = 0.0
        else:
            y_e = self.compute_V(next_state)

        key_e = (state, actual_action)
        self.Q_e[key_e] += self.lr_e * (y_e - self.Q_e[key_e])

        key_d = (state, intended_action)
        y_d = self.Q_e[key_e]
        self.Q_d[key_d] += self.lr_d * (y_d - self.Q_d[key_d])

        V_s = self.compute_V(state)
        A = self.Q_d[key_d] - V_s
        policy, actions = self.get_policy(state)

        if self.entropy_coef > 0:
            entropy = -sum(p * np.log(p + 1e-8) for p in policy.values())
            for a in actions:
                grad_log_pi = (1.0 - policy[a]) if a == intended_action else (-policy[a])
                entropy_grad = -policy[a] * (np.log(policy[a] + 1e-8) + entropy)
                self.theta[(state, a)] += self.lr_actor * (A * grad_log_pi + self.entropy_coef * entropy_grad)
        else:
            for a in actions:
                grad_log_pi = (1.0 - policy[a]) if a == intended_action else (-policy[a])
                self.theta[(state, a)] += self.lr_actor * A * grad_log_pi

    def run_episode(self, train=True):
        state = (self.env.origin, self.env.budget)
        max_steps = len(self.env.nodes) * 2
        for _ in range(max_steps):
            node, budget = state
            if node == self.env.dest or budget < 0:
                return 1.0 if (node == self.env.dest and budget >= 0) else 0.0

            intended = self.select_action(state)
            if intended is None:
                return 0.0

            actual, next_state = self.env_step(state, intended)
            if train:
                self.update(state, intended, actual, next_state)
            state = next_state
        return 0.0

    def train(self, n_episodes=30000):
        return [self.run_episode(train=True) for _ in range(n_episodes)]

    def evaluate(self, n_episodes=10000):
        results = [self.run_episode(train=False) for _ in range(n_episodes)]
        return float(np.mean(results))

    def q_stats(self):
        avg_q_e = float(np.mean(list(self.Q_e.values()))) if self.Q_e else 0.0
        avg_q_d = float(np.mean(list(self.Q_d.values()))) if self.Q_d else 0.0
        return avg_q_e, avg_q_d
