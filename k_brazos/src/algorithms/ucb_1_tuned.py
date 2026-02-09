import numpy as np
from algorithms.algorithm import Algorithm

class UCB1Tuned(Algorithm):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1-Tuned.
        :param k: Número de brazos.
        """
        super().__init__(k)
        # Almacenamos la suma de los cuadrados de las recompensas para un cálculo eficiente de la varianza
        self.sum_squared_rewards = np.zeros(k)
    
    def empiric_variance(self):
        """
        Calcula la varianza empírica de las recompensas usando fórmula incremental
        :return: vector con las varianzas
        """
        # Evitar la división entre 0
        safe_counts = np.maximum(self.counts, 1)
        # Varianza = E[X^2] - E[X]^2
        # Necesitamos sum((r - mean)^2) / n = (sum(r^2) - n*mean^2) / n
        variance = (self.sum_squared_rewards - self.counts * self.values**2) / safe_counts
        return variance
    
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1-Tuned.
        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        # Precomputamos los terminos comunes
        total_counts = np.sum(self.counts)
        log_total = np.log(total_counts)
        log_n_over_counts = log_total / self.counts
        
        # Calculamos la varianza
        variance_term = self.empiric_variance() + np.sqrt(2 * log_n_over_counts)
        
        # UCB
        ucb_values = self.values + np.sqrt(log_n_over_counts * np.minimum(0.25, variance_term))
        
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm: int, reward: float):
        super().update(chosen_arm=chosen_arm, reward=reward)
        self.sum_squared_rewards[chosen_arm] += reward ** 2
