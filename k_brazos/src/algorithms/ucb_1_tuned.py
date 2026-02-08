import numpy as np
from algorithms.algorithm import Algorithm

class UCB1Tuned(Algorithm):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1-Tuned.
        :param k: Número de brazos.
        """
        super().__init__(k)

    def load_rewards(self, rewards):
        self.rewards = np.array(rewards)
    
    def empiric_variance(self):
        """
        Calcula la varianza empírica de las recompensas
        :return: vector con las varianzas
        """
        return (1/self.counts) * sum((self.rewards - self.values)**2)

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1-Tuned.
        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        ucb_values = self.values + np.sqrt((np.log(sum(self.counts))/self.counts)*np.minimum(0.25,self.empiric_variance()+np.sqrt(2*np.log(sum(self.counts))/self.counts)))

        chosen_arm = np.argmax(ucb_values)

        return chosen_arm