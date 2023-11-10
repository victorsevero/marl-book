import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarsRover(gym.Env):
    STATES = {
        0: "Start",
        1: "Site A",
        2: "Site B",
        3: "Base",
        4: "Immobile",
        5: "Destroyed",
    }
    ACTIONS = {0: "Left", 1: "Right"}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(6)
        self.action_space = spaces.Discrete(2)
        self.T = self.get_transition_matrix()
        self.rewards = self.get_rewards_matrix()

    def reset(self):
        super().reset()
        self.state = 0
        return self.state, {}

    def step(self, action):
        previous_state = self.state
        probs = self.T[previous_state, action]
        self.state = self.np_random.choice(self.observation_space.n, p=probs)
        reward = self.rewards[previous_state, action, self.state]
        done = self.is_terminal(self.state)

        return self.state, reward, done, False, {}

    @staticmethod
    def get_transition_matrix():
        # (s, a, s') format
        return np.array(
            [
                [
                    [0.0, 0.9, 0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.0, 0.2, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ]
        )

    @staticmethod
    def get_rewards_matrix():
        # (s, a, s') format
        return np.array(
            [
                [[-1, -1, -1, +10, -3, -10], [-1, -1, -1, +10, -3, -10]],
                [[-1, -1, -1, +10, -3, -10], [-1, -1, -1, +10, -3, -10]],
                [[-1, -1, -1, +10, -3, -10], [-1, -1, -1, +10, -3, -10]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            ]
        )

    @staticmethod
    def is_terminal(state):
        return state in [3, 4, 5]


if __name__ == "__main__":
    env = MarsRover()
    env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, *_ = env.step(action)
        action_str = "left" if not action else "right"
        print(f"Action: {action_str}; State: {env.STATES[obs]}")
