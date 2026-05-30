import numpy as np

class Agent:
    def __init__(self, name="base_agent"):
        self.name = name
        self.history = []

    def decide(self, step: int) -> str:
        raise NotImplementedError

    def observe(self, action: str, reward: float):
        self.history.append({
            "step": len(self.history),
            "action": action,
            "reward": reward
        })

    def reset(self):
        self.history = []


class ExplorativeAgent(Agent):
    def __init__(self, decay_fn=lambda t: 1 / (t + 1), name="explorative"):
        super().__init__(name)
        self.decay_fn = decay_fn

    def decide(self, step: int) -> str:
        if not self.history:
            return np.random.choice(['S', 'R'])

        avg_s = np.mean([h["reward"] for h in self.history if h["action"] == 'S']) if any(h["action"] == 'S' for h in self.history) else -np.inf
        avg_r = np.mean([h["reward"] for h in self.history if h["action"] == 'R']) if any(h["action"] == 'R' for h in self.history) else -np.inf

        best_action = 'S' if avg_s >= avg_r else 'R'
        explore_prob = self.decay_fn(step)

        if np.random.rand() < explore_prob:
            return 'R' if best_action == 'S' else 'S'
        return best_action


class SmallSampleAgent(Agent):
    def __init__(self, K=4, name="small_sample"):
        super().__init__(name)
        self.K = K

    def decide(self, step: int) -> str:
        if step < self.K:
            return np.random.choice(['S', 'R'])  # no enough history

        recent = self.history[-self.K:]
        s_rewards = [h["reward"] for h in recent if h["action"] == 'S']
        r_rewards = [h["reward"] for h in recent if h["action"] == 'R']

        avg_s = np.mean(s_rewards) if s_rewards else -np.inf
        avg_r = np.mean(r_rewards) if r_rewards else -np.inf

        return 'S' if avg_s >= avg_r else 'R'


class ThresholdAgent(Agent):
    def __init__(self, threshold= 1, patience=2, name="threshold"):
        super().__init__(name)
        self.threshold = threshold
        self.patience = patience
        self.current_action = np.random.choice(['S', 'R'])
        self.bad_streak = 0

    def decide(self, step: int) -> str:
        return self.current_action

    def observe(self, action: str, reward: float):
        self.history.append({
            "step": len(self.history),
            "action": action,
            "reward": reward
        })

        # Update based on whether recent outcomes are too bad
        if reward < self.threshold:
            self.bad_streak += 1
        else:
            self.bad_streak = 0

        if self.bad_streak >= self.patience:
            self.current_action = 'S' if self.current_action == 'R' else 'R'
            self.bad_streak = 0  # reset after switching

    def reset(self):
        super().reset()
        self.current_action = np.random.choice(['S', 'R'])
        self.bad_streak = 0


class MaximinAgent(Agent):
    def __init__(self, name="maximin"):
        super().__init__(name)
        self.min_rewards = {'S': None, 'R': None}

    def decide(self, step: int) -> str:
        # Ensure both actions are tried at least once
        for action in ['S', 'R']:
            if self.min_rewards[action] is None:
                return action

        # Maximin decision rule
        if self.min_rewards['S'] > self.min_rewards['R']:
            return 'S'
        elif self.min_rewards['R'] > self.min_rewards['S']:
            return 'R'
        else:
            return np.random.choice(['S', 'R'])  # tie-breaker

    def observe(self, action: str, reward: float):
        self.history.append({
            "step": len(self.history),
            "action": action,
            "reward": reward
        })

        current_min = self.min_rewards[action]
        if current_min is None or reward < current_min:
            self.min_rewards[action] = reward

    def reset(self):
        super().reset()
        self.min_rewards = {'S': None, 'R': None}