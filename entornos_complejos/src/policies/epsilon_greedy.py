import numpy as np
from .base_policy import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, state, qtable):
        
        if np.random.random() < self.epsilon: # Exploración
            n_actions = len(qtable[state])
            action = np.random.choice(n_actions)
        else: # Explotación
            max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
            action = np.random.choice(max_ids)
        return action