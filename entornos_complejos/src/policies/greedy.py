import numpy as np
from .base_policy import BasePolicy


class GreedyPolicy(BasePolicy):

    def select_action(self, state, qtable):
        # Rompemos empates aleatoriamente para evitar sesgos
        max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
        action = np.random.choice(max_ids)
        return action
    
    def action_probabilities(self, state, qtable):
        # Todas las acciones tienen prob. 0 menos la codiciosa, que tiene prob. 1
        probs = np.zeros(len(qtable[state]))
        greedy_action = np.argmax(qtable[state])
        probs[greedy_action] = 1
        return probs
