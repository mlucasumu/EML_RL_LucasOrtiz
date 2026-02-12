import numpy as np
from algorithms.algorithm import Algorithm

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1.
        :param k: Número de brazos.
        :param c: Parámetro de exploración (usualmente 1).
        """
        super().__init__(k)
        self.c = c
    
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.
        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        # Calcula el índice UCB1 para cada brazo
        ucb_values = self.values + self.c * np.sqrt(
            2*np.log(sum(self.counts)) / self.counts
        )
        chosen_arm = np.argmax(ucb_values)

        return chosen_arm