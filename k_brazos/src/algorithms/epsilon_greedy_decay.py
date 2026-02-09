"""
Module: algorithms/epsilon_greedy_decay.py
Description: Implementación del algoritmo epsilon-greedy con decaimiento para el problema de los k-brazos.

Author: Lucas Ortiz
Date: 2026/02/09

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np
from algorithms import Algorithm

class EpsilonGreedyDecay(Algorithm):

    def __init__(self, k: int, initial_epsilon: float = 1.0, min_epsilon: float = 0.01, decay_rate: float = 0.999, initial_value: float = 0.0):
        """
        Inicializa el algoritmo epsilon-greedy con decaimiento.

        :param k: Número de brazos.
        :param initial_epsilon: Valor inicial de epsilon (probabilidad de exploración).
        :param min_epsilon: Valor mínimo que puede alcanzar epsilon.
        :param decay_rate: Factor de decaimiento por paso (epsilon = epsilon * decay_rate).
        :param initial_value: Valor inicial para las estimaciones de recompensa (Optimistic Initialization).
        """
        super().__init__(k, initial_value)
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy con decaimiento.
        
        :return: índice del brazo seleccionado.
        """
        # Explotación vs Exploración
        if np.random.random() < self.epsilon:
            chosen_arm = np.random.choice(self.k)
        else:
            # Romper empates aleatoriamente para evitar sesgos
            max_value = np.max(self.values)
            candidates = np.where(self.values == max_value)[0]
            chosen_arm = np.random.choice(candidates)
            
        # Decaer epsilon después de cada selección
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        return chosen_arm
