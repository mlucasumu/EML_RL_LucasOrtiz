import numpy as np
from .base_learner import BaseLearner


class MCOnPolicy(BaseLearner):

    def __init__(self, state_size, action_size, gamma, first_visit=False):
        super().__init__(state_size, action_size)
        self.gamma = gamma
        self.n_visits = np.zeros([state_size, action_size])
        self.first_visit = first_visit

    def start_episode(self):
        self.episode = []

    def step(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))

    def end_episode(self):
        G = 0
        not_visited = [(state, action) for state, action, reward in self.episode]

        for (state, action, reward) in reversed(self.episode):
            G = reward + self.gamma * G

            not_visited.pop()
            if self.first_visit and (state, action) in not_visited:
                continue

            self.n_visits[state, action] += 1.0
            alpha = 1.0/self.n_visits[state,action]
            td_error = G - self.qtable[state, action]
            self.qtable[state,action] += alpha * td_error
            self.stats['cum_training_error'] += td_error

    def reset(self):
        super().reset()
        self.stats['cum_training_error'] = 0
        self.n_visits = np.zeros([self.state_size, self.action_size])