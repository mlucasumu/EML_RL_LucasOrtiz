import numpy as np
from algorithms.algorithm import Algorithm

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2.
        :param k: Número de brazos.
        :param alpha: Parámetro que controla la exploración (típicamente entre 0 y 1).
        """
        super().__init__(k)
        self.alpha = alpha
        
        # Número de veces que cada brazo ha sido jugado
        self.counts = np.zeros(k)
        
        # Época actual de cada brazo
        self.r = np.zeros(k, dtype=int)
        
        # Brazo actual seleccionado
        self.current_arm = None
        
        # Número de jugadas restantes para el brazo actual
        self.remaining_plays = 0
        
    def tau(self, r: int) -> int:
        """
        Calcula el número de veces que se debe jugar un brazo en la época r.
        :param r: Época actual.
        :return: Número de jugadas.
        """
        return int(np.ceil((1 + self.alpha) ** r))
    
    def bonus(self, r: int, n: int) -> float:
        """
        Calcula el término de exploración (bonus).
        :param r: Época del brazo.
        :param n: Número total de jugadas hasta ahora.
        :return: Valor del bonus.
        """
        tau_r = self.tau(r)
        if tau_r == 0:
            return float('inf')
        return np.sqrt((1 + self.alpha) * np.log(np.e * n / tau_r) / (2 * tau_r))
    
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB2.
        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        # Si estamos en medio de una época, continuar con el brazo actual
        if self.remaining_plays > 0:
            self.remaining_plays -= 1
            return self.current_arm
        
        # Calcular UCB2 para cada brazo
        n = np.sum(self.counts)
        ucb_values = np.zeros(self.k)
        
        for i in range(self.k):
            ucb_values[i] = self.values[i] + self.bonus(self.r[i], n)
        
        # Seleccionar el brazo con mayor UCB
        self.current_arm = np.argmax(ucb_values)
        
        # Incrementar la época del brazo seleccionado
        self.r[self.current_arm] += 1
        
        # Calcular cuántas veces debemos jugar este brazo en esta época
        self.remaining_plays = self.tau(self.r[self.current_arm]) - 1 # tau(k_a + 1) - tau(k_a) = tau(k_a) - 1
        
        return self.current_arm