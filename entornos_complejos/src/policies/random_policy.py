from .base_policy import BasePolicy
import numpy as np

class RandomPolicy(BasePolicy):
    """
    Política completamente aleatoria para exploración pura.
    """

    def select_action(self, state, qtable):
        if callable(qtable):
            n_actions = len(qtable(state))
        else:
            n_actions = len(qtable[state])
        return np.random.choice(n_actions)

    def action_probability(self, state, action, qtable):
        if callable(qtable):
            n_actions = len(qtable(state))
        else:
            n_actions = len(qtable[state])
        return 1.0 / n_actions
