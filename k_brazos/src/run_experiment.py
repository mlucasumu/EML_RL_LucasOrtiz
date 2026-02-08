import numpy as np
from typing import List

from algorithms import Algorithm
from arms import Bandit

def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int, seed: int = 42):

    optimal_arm = bandit.optimal_arm  # Necesario para calcular el porcentaje de selecciones óptimas.
    optimal_expected_reward = bandit.get_expected_rewards()[optimal_arm]

    rewards = np.zeros((len(algorithms), steps)) # Matriz para almacenar las recompensas promedio.

    optimal_selections = np.zeros((len(algorithms), steps))  # Matriz para almacenar el porcentaje de selecciones óptimas.

    cumulative_regret_per_algo = np.zeros((len(algorithms), steps)) # Regret (arrepentimiento) acumulado por algoritmo

    np.random.seed(seed)  # Asegurar reproducibilidad de resultados.

    for run in range(runs):
        current_bandit = Bandit(arms=bandit.arms)

        for algo in algorithms:
            algo.reset() # Reiniciar los valores de los algoritmos.

        total_rewards_per_algo = np.zeros(len(algorithms)) # Acumulador de recompensas por algoritmo. Necesario para calcular el promedio.
        optimal_cumulative_reward = 0 # Recompensa óptima esperada acumulada durante un número de pasos (para el cálculo del regret)
        current_cumulative_reward = np.zeros((len(algorithms))) # Recompensa acumulada para cada algoritmo

        for step in range(steps):
            optimal_cumulative_reward = optimal_cumulative_reward + optimal_expected_reward # Recompensa óptima acumulada para el paso actual
            
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm() # Seleccionar un brazo según la política del algoritmo.
                reward = current_bandit.pull_arm(chosen_arm) # Obtener la recompensa del brazo seleccionado.
                algo.update(chosen_arm, reward) # Actualizar el valor estimado del brazo seleccionado.

                rewards[idx, step] += reward # Acumular la recompensa obtenida en la matriz rewards para el algoritmo idx en el paso step.
                total_rewards_per_algo[idx] += reward # Acumular la recompensa obtenida en total_rewards_per_algo para el algoritmo idx.

                if chosen_arm == optimal_arm:
                    optimal_selections[idx, step] += 1 # Modificar optimal_selections cuando el brazo elegido se corresponda con el brazo óptimo optimal_arm

                # Cálculo del regret
                current_cumulative_reward[idx] = current_cumulative_reward[idx] + reward
                cumulative_regret_per_algo[idx, step] += optimal_cumulative_reward - current_cumulative_reward[idx]

    rewards /= runs
    optimal_selections /= runs
    cumulative_regret_per_algo /= runs

    return rewards, optimal_selections, cumulative_regret_per_algo
