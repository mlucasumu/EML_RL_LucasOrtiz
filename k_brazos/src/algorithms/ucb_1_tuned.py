import numpy as np
from algorithms.algorithm import Algorithm

class UCB1Tuned_(Algorithm):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1-Tuned.
        :param k: Número de brazos.
        """
        super().__init__(k)
        self.rewards = [[] for _ in range(k)]
    
    def empiric_variance(self):
        """
        Calcula la varianza empírica de las recompensas
        :return: vector con las varianzas
        """
        sum_total = np.zeros(len(self.counts))
        for i in range(len(self.counts)):
            sum_total[i] = sum((np.array(self.rewards[i]) - self.values[i])**2)

        return (1/self.counts) * sum_total

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1-Tuned.
        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        sum_total = np.zeros(len(self.counts))
        for i in range(len(self.counts)):
            sum_total[i] = sum((np.array(self.rewards[i]) - self.values[i])**2)
        
        ucb_values = self.values + np.sqrt((np.log(sum(self.counts))/self.counts)*np.minimum(0.25,self.empiric_variance()+np.sqrt(2*np.log(sum(self.counts))/self.counts)))

        chosen_arm = np.argmax(ucb_values)

        return chosen_arm
    
    def update(self, chosen_arm: int, reward: float):
        super().update(chosen_arm=chosen_arm, reward=reward)
        self.rewards[chosen_arm].append(reward)

import numpy as np
from algorithms.algorithm import Algorithm

class UCB1Tuned(Algorithm):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1-Tuned.
        :param k: Número de brazos.
        """
        super().__init__(k)
        # Store sum of rewards and sum of squared rewards for efficient variance calculation
        self.sum_rewards = np.zeros(k)
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
        # De forma incremental aumentalos las sumas
        self.sum_rewards[chosen_arm] += reward
        self.sum_squared_rewards[chosen_arm] += reward ** 2
