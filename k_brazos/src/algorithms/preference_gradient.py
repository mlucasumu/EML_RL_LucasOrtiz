import numpy as np

from algorithms import Algorithm

class PreferenceGradient(Algorithm):

    def __init__(self, k: int, alpha: float = 0.2):
        """
        Inicializa el algoritmo de Gradiente de Preferencias.

        :param k: Número de brazos.
        :param alpha: Tamaño de las actualizaciones de preferencia (tasa de aprendizaje).
        :raises ValueError: Si alpha no es mayor que 0.
        """
        assert alpha > 0, "El parámetro alpha debe ser mayor que 0."

        super().__init__(k)
        self.alpha = alpha
        self.preferences = np.zeros(k, dtype=float)
        self.probs = []

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política de gradiente de preferencias.

        :return: índice del brazo seleccionado.
        """
        # Calcular probabilidades
        probs = []
        for preference in self.preferences:
            numerator = np.exp(preference)
            probs.append(numerator)

        total = np.sum(probs)
        probs = probs/total
        self.probs = probs

        # Seleccionar acción de acuerdo a probabilidades
        chosen_arm = np.random.choice(list(range(self.k)), p=probs)

        return chosen_arm

    def update(self, chosen_arm, reward):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        :param chosen_arm: Índice del brazo que fue tirado.
        :param reward: Recompensa obtenida.
        """
        super().update(chosen_arm, reward)

        # Actualizamos las preferencias (Sutton y Barto 2018, página 37)
        rewards_baseline = np.mean(self.values)
        
        for i in range(len(self.preferences)):
            if i==chosen_arm:
                self.preferences[i] = self.preferences[i] \
                                    + self.alpha*(reward - rewards_baseline) * (1 - self.probs[i])
            else:
                self.preferences[i] = self.preferences[i] \
                                    - self.alpha*(reward - rewards_baseline) * self.probs[i]