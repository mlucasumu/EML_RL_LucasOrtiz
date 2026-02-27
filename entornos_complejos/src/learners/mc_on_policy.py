import numpy as np
from .base_learner import BaseLearner


class MCOnPolicy(BaseLearner):

    def __init__(self, state_size, action_size, discount_factor):
        super().__init__(state_size, action_size)
        self.discount_factor = discount_factor
        self.n_visits = np.zeros([state_size, action_size])

    def start_episode(self):
        self.episode = []

    def step(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))

    def end_episode(self):
        G = 0
        for (state, action, reward) in reversed(self.episode):
            G = reward + self.discount_factor * G
            self.n_visits[state, action] += 1.0
            alpha = 1.0/self.n_visits[state,action]
            td_error = G - self.qtable[state, action]
            self.qtable[state,action] += alpha * td_error
            self.stats['cum_training_error'] += td_error

    def reset(self):
        super().reset()
        self.stats['cum_training_error'] = 0