import numpy as np
from .base_policy import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    """
    def select_action(self, state, qtable):
        if np.random.random() < self.epsilon: # Exploración
            n_actions = len(qtable[state])
            action = np.random.choice(n_actions)
        else: # Explotación
            max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
            action = np.random.choice(max_ids)
        return action
    """

    def select_action(self, state, qvalues):
        # Distinguir si me pasan la tabla entera o solo la fila de la acción
        if len(np.shape(qvalues)) > 1:
            q_vals = qvalues[state]
        else:
            q_vals = qvalues
        n_actions = len(q_vals)

        if np.random.random() < self.epsilon:  # Exploración
            action = np.random.choice(n_actions)
        else:                                   # Explotación
            max_ids = np.where(q_vals == np.max(q_vals))[0]
            action  = np.random.choice(max_ids)

        return action
    
    def action_probabilities(self, state, qtable):
        n_actions = len(qtable[state])
        probs = np.ones(n_actions) * (self.epsilon / n_actions)
        greedy_action = np.argmax(qtable[state])
        probs[greedy_action] += (1.0 - self.epsilon)
        return probs