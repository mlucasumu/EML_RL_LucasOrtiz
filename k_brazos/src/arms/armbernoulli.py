import numpy as np

from arms import ArmBinomial


class ArmBernoulli(ArmBinomial):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución Bernoulli.

        :param p: Probabilidad de éxito.
        """

        super().__init__(1, p)

    def __str__(self):
        """
        Representación en cadena del brazo Bernoulli.

        :return: Descripción detallada del brazo Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades de éxito únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param p_min: Valor mínimo de la probabilidad de éxito p.
        :param p_max: Valor máximo de la probabilidad de éxito p.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)
        
        p_values = list(p_values)
        arms = [cls(p) for p in p_values]
        
        return arms