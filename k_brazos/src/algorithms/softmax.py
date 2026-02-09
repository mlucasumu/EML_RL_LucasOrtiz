import numpy as np

from algorithms import Algorithm

class Softmax(Algorithm):

    def __init__(self, k: int, temp: float = 1):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param temp: Temperatura: cuanto más alta (>1), más exploración. Cuando más baja (<1), más explotación. 
        :raises ValueError: Si temp no es mayor que 0.
        """
        assert temp > 0, "El parámetro temp debe ser mayor que 0."

        super().__init__(k)
        self.temp = temp

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política softmax.

        :return: índice del brazo seleccionado.
        """
        # Fase de inicialización: jugar cada brazo una vez
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        # Calcular probabilidades
        probs = np.exp(self.values/self.temp)
        probs = probs / np.sum(probs)

        # Seleccionar acción de acuerdo a probabilidades
        chosen_arm = np.random.choice(list(range(self.k)), p=probs)

        return chosen_arm
