import numpy as np
from .epsilon_greedy import EpsilonGreedyPolicy


class EpsilonDecayPolicy(EpsilonGreedyPolicy):

    def __init__(self, init_epsilon = 1.0, min_epsilon = 0.01, decay_rate = 0.999):
        super().__init__(init_epsilon)
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_action(self, state, qtable):
        action = super().select_action(state, qtable)
        
        # Decaer epsilon en cada paso
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

        return action