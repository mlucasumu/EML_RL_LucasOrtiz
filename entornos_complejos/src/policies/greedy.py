import numpy as np
from base_policy import BasePolicy


class GreedyPolicy(BasePolicy):

    def select_action(self, state, qtable):
        # Rompemos empates aleatoriamente para evitar sesgos
        max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
        action = np.random.choice(max_ids)
        return action