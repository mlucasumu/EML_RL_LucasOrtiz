import numpy as np

from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de experimentos.
        :param p: Probabilidad de éxito.
        """
        assert n >= 0, "El número de experimentos debe ser mayor o igual a 0."
        assert p >= 0 and p <= 1, "La probabilidad de éxito debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado de la distribución.
        """

        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 1, n_max: int = 20,
                      p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con números de experimentos en el rango [n_min, n_max]
        y probabilidades de éxito únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo del número de ensayos n.
        :param n_max: Valor máximo del número de ensayos n.
        :param p_min: Valor mínimo de la probabilidad de éxito p.
        :param p_max: Valor máximo de la probabilidad de éxito p.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n_min <= n_max, "El valor de n_min debe ser menor o igual que n_max."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k valores de n
        n_values = []
        while len(n_values) < k:
            n = np.random.random_integers(n_min, n_max)
            n_values.append(n)

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        p_values = list(p_values)

        arms = [cls(n, p) for n,p in zip(n_values, p_values)]

        return arms