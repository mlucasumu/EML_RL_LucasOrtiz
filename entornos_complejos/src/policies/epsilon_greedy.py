import numpy as np
from .base_policy import BasePolicy


class EpsilonGreedyPolicy(BasePolicy):

    def __init__(self, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super().__init__()
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, qtable):
        # Support both normal Q-tables and function approximators
        if callable(qtable):
            q_values = qtable(state)
        else:
            q_values = qtable[state]
            
        if np.random.random() < self.epsilon: # Exploración
            n_actions = len(q_values)
            action = np.random.choice(n_actions)
        else: # Explotación
            max_ids = np.where(q_values == np.max(q_values))[0]
            action = np.random.choice(max_ids)
        return action

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.epsilon = self.epsilon_start

    def action_probability(self, state, action, qtable):
        """
        Devuelve la probabilidad de elegir `action` en `state`.
        - Epsilon/N para todas las acciones.
        - + (1-Epsilon)/Max_N para las acciones codiciosas.
        """
        if callable(qtable):
            q_values = qtable(state)
        else:
            q_values = qtable[state]
            
        n_actions = len(q_values)
        max_q = np.max(q_values)
        max_actions = np.where(q_values == max_q)[0]
        n_max_actions = len(max_actions)
        
        prob = self.epsilon / n_actions
        if action in max_actions:
            prob += (1.0 - self.epsilon) / n_max_actions
        return prob