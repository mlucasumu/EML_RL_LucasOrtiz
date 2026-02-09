
"""
Module: algorithms/preference_gradient.py
Description: Contiene la implementación del algoritmo Gradient Bandit (Preferencia).

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

# from algorithms import Algorithm # Use this if 'algorithms' is in path, or relative import
from .algorithm import Algorithm # Use relative import to match __init__.py style within the package

class PreferenceGradient(Algorithm):
    def __init__(self, k: int, alpha: float, use_baseline: bool = True):
        """
        Inicializa el algoritmo de Gradient Bandit (Preferencias).
        
        :param k: Número de brazos.
        :param alpha: Tasa de aprendizaje (step-size parameter) > 0.
        :param use_baseline: Si True, usa el promedio de recompensas como baseline. 
                             Si False, usa baseline=0.
        """
        super().__init__(k)
        self.alpha = alpha
        self.use_baseline = use_baseline
        
        # H_t(a): Preferencias numéricas para cada acción a
        self.preferences = np.zeros(k, dtype=float)
        
        # Probabilidades pi_t(a)
        self.probs = np.zeros(k, dtype=float)
        
        # Variables para calcular el baseline (average reward)
        self.average_reward = 0.0
        self.total_reward = 0.0
        self.time_step = 0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la distribución Softmax de las preferencias.
        :return: Índice del brazo seleccionado.
        """
        # Calcular probabilidades pi_t(a) usando softmax
        # Restar max(preferences) para estabilidad numérica (exp(x)/sum(exp(x)) == exp(x-c)/sum(exp(x-c)))
        exp_pref = np.exp(self.preferences - np.max(self.preferences))
        self.probs = exp_pref / np.sum(exp_pref)
        
        # Seleccionar acción de acuerdo a probabilidades
        chosen_arm = np.random.choice(range(self.k), p=self.probs)
        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las preferencias H_t(a) y el baseline.
        :param chosen_arm: Brazo seleccionado A_t.
        :param reward: Recompensa obtenida R_t.
        """
        self.time_step += 1
        
        # Actualizar average reward para el baseline
        # Baseline es R_avg si use_baseline=True, sino 0
        if self.use_baseline:
            # Actualización incremental del promedio: R_avg_new = R_avg_old + (R - R_avg_old)/n
            self.average_reward += (reward - self.average_reward) / self.time_step
            baseline = self.average_reward
        else:
            baseline = 0.0
            
        # Calcular el término común: alpha * (R_t - baseline)
        term = self.alpha * (reward - baseline)
        
        # Crear one-hot vector para la acción seleccionada
        one_hot = np.zeros(self.k)
        one_hot[chosen_arm] = 1
        
        # Actualizar preferencias para todas las acciones:
        # H_{t+1}(a) = H_t(a) + alpha * (R_t - baseline) * (I(a=A_t) - pi_t(a))
        self.preferences += term * (one_hot - self.probs)
        
        # Llamar al método update de la clase base para mantener counts y values actualizados
        # aunque values (Q-estimates) no se usan para la selección en este algoritmo.
        super().update(chosen_arm, reward)

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()
        self.preferences = np.zeros(self.k, dtype=float)
        self.probs = np.zeros(self.k, dtype=float)
        self.average_reward = 0.0
        self.total_reward = 0.0
        self.time_step = 0
