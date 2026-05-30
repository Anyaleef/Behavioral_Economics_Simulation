import numpy as np
from scipy.stats import geom, norm
from typing import Callable


class CableCarEpisode:
    def __init__(self, p_empty: float, reward_for_empty: float, discomfort_mean: float, discomfort_std: float, f: Callable):
        self.p = p_empty
        self.reward_for_empty = reward_for_empty
        self.discomfort_dist = norm(loc=discomfort_mean, scale=discomfort_std)
        # F(t) is the utility function for the risky path, defined as reward_for_empty function of T (the number of skipped cars)
        self.f = f

    def generate_episode(self, steps: int):
        """
        Returns reward_for_empty list of dicts with precomputed outcomes:
        {
            'safe_reward': float,
            'risky_reward': float,
            'risky_skipped': int
        }
        """
        data = []
        for _ in range(steps):
            # Safe path
            is_empty = np.random.rand() < self.p
            if is_empty:
                safe_reward = self.reward_for_empty
            else:
                safe_reward = self.discomfort_dist.rvs()

            # Risky path
            T = geom.rvs(self.p)
            risky_reward = self.reward_for_empty + self.f(T - 1)

            data.append({
                "safe_reward": safe_reward,
                "risky_reward": risky_reward,
                "risky_skipped": T - 1
            })

        return data