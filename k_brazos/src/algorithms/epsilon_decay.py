import numpy as np
from algorithms import Algorithm

class EpsilonGreedyDecay(Algorithm):

    def __init__(self, k: int, initial_epsilon: float = 1.0, min_epsilon: float = 0.01, decay_rate: float = 0.999):
        """
        Inicializa el algoritmo epsilon-greedy con decaimiento.

        :param k: Número de brazos.
        :param initial_epsilon: Valor inicial de epsilon (probabilidad de exploración).
        :param min_epsilon: Valor mínimo que puede alcanzar epsilon.
        :param decay_rate: Factor de decaimiento por paso (epsilon = epsilon * decay_rate).
        """
        super().__init__(k)
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy con decaimiento.
        
        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
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
